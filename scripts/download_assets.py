#!/usr/bin/env python3
"""Provision pinned assets and check local completeness and revision evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="pi05 paligemma robometer qwen dino libero")
    parser.add_argument("--all", action="store_true", help="Download every pinned asset")
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument(
        "--check", action="store_true", help="Check local files without downloading"
    )
    return parser.parse_args()


def required_files(name: str) -> tuple[str, ...]:
    return {
        "pi05": ("model.safetensors", "physical-intelligence/libero/norm_stats.json"),
        "robometer": ("model.safetensors.index.json", "config.yaml"),
        "qwen": ("model.safetensors.index.json", "config.json"),
        "dino": ("model.safetensors", "config.json", "preprocessor_config.json"),
        "libero": (
            "stable_scanned_objects",
            "scenes/libero_tabletop_base_style.xml",
        ),
    }[name]


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    if algorithm == "sha1":
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_tokenizer(spec, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and file_digest(destination) == spec.sha256:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            with urlopen(str(spec.url), timeout=60) as response:
                shutil.copyfileobj(response, stream)
        if file_digest(temporary) != spec.sha256:
            raise ValueError("PaliGemma tokenizer SHA-256 mismatch; refusing to install")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def check_snapshot(name: str, destination: Path) -> tuple[list[str], list[str], list[str]]:
    """Expand required paths and every safetensors index, including nested indexes."""
    required = set(required_files(name))
    errors = []
    for index in sorted(destination.rglob("*.safetensors.index.json")):
        relative = index.relative_to(destination).as_posix()
        required.add(relative)
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            weight_map = data.get("weight_map") if isinstance(data, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("weight_map must be a nonempty object")
            for key, shard in weight_map.items():
                if (
                    not key
                    or not isinstance(shard, str)
                    or not shard
                    or Path(shard).is_absolute()
                    or ".." in Path(shard).parts
                    or "\\" in shard
                    or not shard.endswith(".safetensors")
                ):
                    raise ValueError(f"invalid shard entry: {key!r}: {shard!r}")
                required.add((index.parent / shard).relative_to(destination).as_posix())
        except (OSError, ValueError) as exc:
            errors.append(f"{relative}: {exc}")

    missing = []
    files = set()
    for relative in sorted(required):
        path = destination / relative
        if name == "libero" and relative == "stable_scanned_objects":
            children = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else []
            if not children:
                missing.append(relative)
            for child in children:
                files.add(child.relative_to(destination).as_posix())
                if child.stat().st_size == 0:
                    missing.append(child.relative_to(destination).as_posix())
        elif not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)
        else:
            files.add(relative)
    return sorted(files), missing, errors


def check_revision(destination: Path, files: list[str], expected: str) -> dict:
    """Rehash checked files against local HF commit/ETag evidence, never just YAML."""
    revisions = set()
    unverified = []
    errors = []
    for relative in files:
        metadata = destination / ".cache/huggingface/download" / f"{relative}.metadata"
        try:
            # HF local-dir metadata is a three-line commit, ETag, timestamp record.
            commit, etag, timestamp = metadata.read_text(encoding="utf-8").splitlines()
            float(timestamp)
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError("invalid commit")
            if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", etag):
                raise ValueError("unsupported ETag")
        except (OSError, ValueError):
            unverified.append(relative)
            continue
        revisions.add(commit)
        algorithm = "sha256" if len(etag) == 64 else "sha1"
        if file_digest(destination / relative, algorithm) != etag:
            errors.append(f"{relative}: content does not match HF metadata ETag")
    verified = (
        next(iter(revisions)) if len(revisions) == 1 and not unverified and not errors else None
    )
    if revisions - {expected}:
        errors.append(f"HF metadata revisions {sorted(revisions)} differ from expected {expected}")
    return {
        "verified_revision": verified,
        "revision_status": "mismatch" if errors else "verified" if verified else "unverified",
        "revision_evidence": "local_hf_metadata_and_content_hashes" if revisions else None,
        "revision_scope": "checked_files",
        "observed_revisions": sorted(revisions),
        "unverified_files": unverified,
        "errors": errors,
    }


def check_asset(name: str, destination: Path, spec) -> dict:
    """Read-only, offline preflight API; spec is an assets.yaml entry (dict or DictConfig).

    ``ok`` means required files are present and no integrity/revision mismatch is
    known. It does not imply ``verified_revision`` is available. HF verification
    covers ``checked_files`` only and trusts local HF download metadata.
    """
    destination = Path(destination)
    if name == "paligemma":
        expected = str(spec["sha256"])
        actual = file_digest(destination) if destination.is_file() else None
        matches = actual == expected
        return {
            "name": name,
            "url": str(spec["url"]),
            "destination": str(destination),
            "expected_revision": f"sha256:{expected}",
            "verified_revision": f"sha256:{actual}" if matches else None,
            "revision_status": "verified" if matches else "mismatch" if actual else "unverified",
            "revision_evidence": "sha256" if actual else None,
            "revision_scope": "entire_file",
            "actual_sha256": actual,
            "ok": matches,
            "missing": [] if actual else [destination.name],
            "errors": ["tokenizer SHA-256 mismatch"] if actual and not matches else [],
            "checked_files": [destination.name] if actual else [],
        }
    files, missing, errors = check_snapshot(name, destination)
    evidence = check_revision(destination, files, str(spec["revision"]))
    errors.extend(evidence.pop("errors"))
    if missing or errors:
        evidence["verified_revision"] = None
        if evidence["revision_status"] == "verified":
            evidence["revision_status"] = "unverified"
    return {
        "name": name,
        "repo_id": str(spec["repo_id"]),
        "repo_type": str(spec["repo_type"]),
        "expected_revision": str(spec["revision"]),
        **evidence,
        "destination": str(destination),
        "ok": not missing and not errors,
        "missing": missing,
        "errors": errors,
        "checked_files": files,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    asset_root = (args.asset_root or Path(os.environ.get("ASSET_ROOT", root / "assets"))).resolve()
    cfg = OmegaConf.load(root / "configs/assets.yaml")
    names = list(cfg.assets) if args.all else args.names
    if not names:
        raise SystemExit("Choose asset names or pass --all")
    unknown = sorted(set(names) - set(cfg.assets))
    if unknown:
        raise SystemExit(f"Unknown assets: {', '.join(unknown)}")
    if "pi05" in names and "paligemma" not in names:
        names.append("paligemma")

    if not args.check:
        # Clear offline flags before importing HF: the library caches them at import time.
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        from huggingface_hub import constants, snapshot_download

        constants.HF_HUB_OFFLINE = False

    asset_root.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        spec = cfg.assets[name]
        destination = asset_root / str(spec.destination)
        if name == "paligemma":
            if not args.check:
                print(f"Downloading {name}: {spec.url} -> {destination}")
                download_tokenizer(spec, destination)
        elif not args.check:
            print(f"Downloading {name}: {spec.repo_id}@{spec.revision} -> {destination}")
            snapshot_download(
                repo_id=str(spec.repo_id),
                repo_type=str(spec.repo_type),
                revision=str(spec.revision),
                local_dir=destination,
            )
        record = check_asset(name, destination, spec)
        records.append(record)
        status = (
            "ok" if record["ok"] else "INVALID " + "; ".join(record["missing"] + record["errors"])
        )
        print(f"{name}: {status}; revision {record['revision_status']}")

    manifest = {
        "schema_version": 2,
        "checked_at": datetime.now().astimezone().isoformat(),
        "assets": records,
    }
    (asset_root / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not all(record["ok"] for record in records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
