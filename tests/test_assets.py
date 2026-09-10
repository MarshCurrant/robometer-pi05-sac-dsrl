"""Offline asset completeness, provenance, and tokenizer regressions."""

import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from scripts import download_assets


def check_asset(monkeypatch, root, name):
    monkeypatch.setattr(
        download_assets,
        "parse_args",
        lambda: SimpleNamespace(names=[name], all=False, asset_root=root, check=True),
    )
    try:
        download_assets.main()
    except SystemExit as exc:
        code = exc.code
    else:
        code = 0
    return code, json.loads((root / "asset_manifest.json").read_text())["assets"][0]


def indexed_asset(root, name="qwen"):
    destination = root / ("Qwen3-VL-4B-Instruct" if name == "qwen" else "Robometer-4B")
    destination.mkdir()
    (destination / ("config.json" if name == "qwen" else "config.yaml")).write_text("{}")
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"},
            }
        )
    )
    (destination / "model-1.safetensors").write_bytes(b"first shard")
    return destination


@pytest.mark.parametrize("name", ["qwen", "robometer"])
def test_check_rejects_missing_indexed_shard(monkeypatch, tmp_path, name):
    indexed_asset(tmp_path, name)
    code, record = check_asset(monkeypatch, tmp_path, name)
    assert code == 2
    assert "model-2.safetensors" in record["missing"]


def test_unproven_local_revision_is_not_reported_as_verified(monkeypatch, tmp_path):
    destination = indexed_asset(tmp_path)
    (destination / "model-2.safetensors").write_bytes(b"second shard")
    code, record = check_asset(monkeypatch, tmp_path, "qwen")
    assert code == 0
    assert record["expected_revision"]
    assert record["verified_revision"] is None
    assert record["revision_status"] == "unverified"
    assert "revision" not in record


def test_paligemma_uses_local_asset_with_empty_offline_cache(monkeypatch, tmp_path):
    from openpi.models import tokenizer

    monkeypatch.setenv("ASSET_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path / "empty-openpi"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("PALIGEMMA_TOKENIZER_PATH", raising=False)
    proto = b"local tokenizer fixture"
    (tmp_path / "paligemma_tokenizer.model").write_bytes(proto)
    monkeypatch.setattr(
        tokenizer, "_PALIGEMMA_SHA256", hashlib.sha256(proto).hexdigest(), raising=False
    )
    monkeypatch.setattr(
        tokenizer.download, "maybe_download", lambda *a, **kw: pytest.fail("network/cache access")
    )
    seen = []
    monkeypatch.setattr(
        tokenizer.sentencepiece, "SentencePieceProcessor", lambda **kw: seen.append(kw)
    )
    tokenizer.PaligemmaTokenizer()
    assert seen == [{"model_proto": proto}]
    assert not (tmp_path / "empty-openpi").exists()
    assert not (tmp_path / "empty-hf").exists()


def test_fresh_process_tokenizer_startup_without_network_or_caches(tmp_path):
    source = (
        Path(download_assets.__file__).resolve().parents[1] / "assets/paligemma_tokenizer.model"
    )
    if not source.is_file():
        pytest.skip("Provision the pinned tokenizer: scripts/download_assets.py paligemma")
    shutil.copyfile(source, tmp_path / "paligemma_tokenizer.model")
    environment = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "OPENPI_DATA_HOME": str(tmp_path / "openpi-cache"),
        "HF_HOME": str(tmp_path / "hf-cache"),
        "HF_HUB_CACHE": str(tmp_path / "hf-hub-cache"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "ASSET_ROOT": str(tmp_path),
        "PALIGEMMA_TOKENIZER_PATH": str(tmp_path / "paligemma_tokenizer.model"),
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import socket
def no_network(*args, **kwargs):
    raise AssertionError('network access during cold startup')
socket.socket.connect = no_network
from openpi.models.tokenizer import PaligemmaTokenizer
import numpy as np
tokens, mask = PaligemmaTokenizer().tokenize('pick_up the cup', np.zeros(8))
assert tokens.shape == mask.shape == (48,)
assert mask.any()
print('cold-cache tokenizer startup passed')
""",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert "cold-cache tokenizer startup passed" in result.stdout
    assert not (tmp_path / "openpi-cache").exists()


def test_manifest_includes_digest_pinned_paligemma():
    cfg = OmegaConf.load(
        Path(download_assets.__file__).resolve().parents[1] / "configs/assets.yaml"
    )
    assert (
        cfg.assets.paligemma.sha256
        == "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
    )
    assert "generation=" in cfg.assets.paligemma.url


@pytest.mark.parametrize(
    "contents",
    [
        "{",
        "[]",
        "{}",
        '{"weight_map": {}}',
        '{"weight_map": []}',
        '{"weight_map": {"a": "../outside.safetensors"}}',
        '{"weight_map": {"a": "/outside.safetensors"}}',
        '{"weight_map": {"a": 42}}',
    ],
)
def test_check_rejects_malformed_indexes(monkeypatch, tmp_path, contents):
    destination = indexed_asset(tmp_path)
    (destination / "model.safetensors.index.json").write_text(contents)
    code, record = check_asset(monkeypatch, tmp_path, "qwen")
    assert code == 2
    assert record["errors"]
    assert record["verified_revision"] is None


@pytest.mark.parametrize("kind", ["empty", "directory"])
def test_index_shard_must_be_nonempty_file(monkeypatch, tmp_path, kind):
    destination = indexed_asset(tmp_path)
    shard = destination / "model-2.safetensors"
    shard.touch() if kind == "empty" else shard.mkdir()
    code, record = check_asset(monkeypatch, tmp_path, "qwen")
    assert code == 2
    assert "model-2.safetensors" in record["missing"]


def test_all_indexes_are_checked_including_nested(monkeypatch, tmp_path):
    destination = indexed_asset(tmp_path)
    (destination / "model-2.safetensors").write_bytes(b"second shard")
    nested = destination / "extra"
    nested.mkdir()
    (nested / "other.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {"x": "missing.safetensors"},
            }
        )
    )
    code, record = check_asset(monkeypatch, tmp_path, "qwen")
    assert code == 2
    assert "extra/missing.safetensors" in record["missing"]


def write_hf_metadata(destination, files, revision):
    for relative in files:
        path = destination / relative
        data = path.read_bytes()
        # Both supported HF ETags: LFS SHA-256 and Git blob SHA-1.
        etag = (
            hashlib.sha256(data).hexdigest()
            if relative.endswith(".safetensors")
            else hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        )
        metadata = destination / ".cache/huggingface/download" / f"{relative}.metadata"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(f"{revision}\n{etag}\n1234.0\n")


@pytest.mark.parametrize(
    "change", [None, "modified", "missing_metadata", "wrong_revision", "mixed_revision"]
)
def test_revision_requires_matching_content_and_metadata(monkeypatch, tmp_path, change):
    destination = indexed_asset(tmp_path)
    (destination / "model-2.safetensors").write_bytes(b"second shard")
    files, _, _ = download_assets.check_snapshot("qwen", destination)
    cfg = OmegaConf.load(
        Path(download_assets.__file__).resolve().parents[1] / "configs/assets.yaml"
    )
    revision = str(cfg.assets.qwen.revision)
    write_hf_metadata(destination, files, "a" * 40 if change == "wrong_revision" else revision)
    if change == "modified":
        (destination / "model-2.safetensors").write_bytes(b"edited weight")
    elif change == "missing_metadata":
        (destination / ".cache/huggingface/download/model-2.safetensors.metadata").unlink()
    elif change == "mixed_revision":
        write_hf_metadata(destination, ["model-2.safetensors"], "b" * 40)
    code, record = check_asset(monkeypatch, tmp_path, "qwen")
    assert code == (0 if change in (None, "missing_metadata") else 2)
    assert record["verified_revision"] == (revision if change is None else None)
    assert record["revision_scope"] == "checked_files"
    assert record["revision_status"] == (
        "verified"
        if change is None
        else "unverified"
        if change == "missing_metadata"
        else "mismatch"
    )


@pytest.mark.parametrize("valid", [False, True])
def test_tokenizer_download_is_digest_checked_and_atomic(monkeypatch, tmp_path, valid):
    data = b"downloaded tokenizer"
    destination = tmp_path / "paligemma_tokenizer.model"
    destination.write_bytes(b"previous file")
    spec = SimpleNamespace(
        url="https://example.invalid/model", sha256=hashlib.sha256(data).hexdigest()
    )
    monkeypatch.setattr(
        download_assets, "urlopen", lambda *a, **kw: io.BytesIO(data if valid else b"wrong file")
    )
    if valid:
        download_assets.download_tokenizer(spec, destination)
        assert destination.read_bytes() == data
    else:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            download_assets.download_tokenizer(spec, destination)
        assert destination.read_bytes() == b"previous file"
    assert list(tmp_path.iterdir()) == [destination]


def test_tokenizer_download_reuses_verified_local_file(monkeypatch, tmp_path):
    destination = tmp_path / "model"
    destination.write_bytes(b"already pinned")
    spec = SimpleNamespace(sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    monkeypatch.setattr(download_assets, "urlopen", lambda *a, **kw: pytest.fail("redownload"))
    download_assets.download_tokenizer(spec, destination)


@pytest.mark.parametrize("kind", ["missing", "corrupt", "bad_override"])
def test_tokenizer_fails_closed_without_network(monkeypatch, tmp_path, kind):
    from openpi.models import tokenizer

    monkeypatch.setenv("ASSET_ROOT", str(tmp_path))
    monkeypatch.delenv("PALIGEMMA_TOKENIZER_PATH", raising=False)
    monkeypatch.setattr(
        tokenizer.download, "maybe_download", lambda *a, **kw: pytest.fail("network/cache access")
    )
    if kind == "corrupt":
        (tmp_path / "paligemma_tokenizer.model").write_bytes(b"corrupt")
    elif kind == "bad_override":
        monkeypatch.setenv("PALIGEMMA_TOKENIZER_PATH", str(tmp_path / "not-found.model"))
    with pytest.raises(ValueError if kind == "corrupt" else FileNotFoundError):
        tokenizer.PaligemmaTokenizer()


def test_preflight_helper_is_read_only_and_accepts_plain_spec(monkeypatch, tmp_path):
    destination = indexed_asset(tmp_path)
    (destination / "model-2.safetensors").write_bytes(b"second shard")
    before = set(tmp_path.rglob("*"))
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **kw: pytest.fail("network access"))
    result = download_assets.check_asset(
        "qwen",
        destination,
        {
            "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
            "repo_type": "model",
            "revision": "a" * 40,
        },
    )
    assert result["ok"]
    assert result["verified_revision"] is None
    assert set(tmp_path.rglob("*")) == before


def test_provisioned_tokenizer_is_numerically_identical_with_cold_caches(monkeypatch, tmp_path):
    import numpy as np
    import sentencepiece
    from openpi.models import tokenizer

    source = (
        Path(download_assets.__file__).resolve().parents[1] / "assets/paligemma_tokenizer.model"
    )
    if not source.is_file():
        pytest.skip("Provision the pinned tokenizer: scripts/download_assets.py paligemma")
    shutil.copyfile(source, tmp_path / "paligemma_tokenizer.model")
    monkeypatch.setenv("ASSET_ROOT", str(tmp_path))
    monkeypatch.delenv("PALIGEMMA_TOKENIZER_PATH", raising=False)
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path / "empty-openpi"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **kw: pytest.fail("network access"))
    monkeypatch.setattr(
        tokenizer.download, "maybe_download", lambda *a, **kw: pytest.fail("cache access")
    )
    reference = sentencepiece.SentencePieceProcessor(model_proto=source.read_bytes())
    for max_len in (8, 48, 200):
        local = tokenizer.PaligemmaTokenizer(max_len=max_len)
        for prompt in ("pick_up the cup", "  Open\nthe drawer  ", "move " * 100):
            for state in (None, np.array([-2, -1, -0.1, 0, 0.1, 1, 2, 0.5])):
                cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
                if state is None:
                    expected = reference.encode(cleaned, add_bos=True) + reference.encode("\n")
                else:
                    bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
                    state_str = " ".join(map(str, bins))
                    expected = reference.encode(
                        f"Task: {cleaned}, State: {state_str};\nAction: ", add_bos=True
                    )
                count = min(len(expected), max_len)
                expected = expected[:max_len] + [0] * (max_len - count)
                tokens, mask = local.tokenize(prompt, state)
                np.testing.assert_array_equal(tokens, expected)
                np.testing.assert_array_equal(mask, [True] * count + [False] * (max_len - count))
    assert not (tmp_path / "empty-openpi").exists()
    assert not (tmp_path / "empty-hf").exists()
