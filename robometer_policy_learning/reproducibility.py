"""Configuration freezing and provenance utilities for formal runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

HASH_EXCLUDED_KEYS = {"runtime", "logging", "reproduction"}
HASH_EXCLUDED_PATHS = {
    "dsrl.pi05_checkpoint",
    "env.h5_dataset_path",
    "model.dinov2_model",
    "resources.policy_gpu",
    "resources.reward_server.gpu",
    "resources.reward_server.host",
    "resources.reward_server.port",
    "reward_model.eval_server_url",
    "reward_model.eval_server_port",
    "reward_model.model_path",
    "training.load_dir",
}


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonical(item)
            for key, item in sorted(value.items())
            if key not in HASH_EXCLUDED_KEYS
        }
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def scientific_recipe_hash(cfg: DictConfig) -> str:
    """Hash scientific settings while excluding paths and logger metadata."""
    data = OmegaConf.to_container(cfg, resolve=True)
    for dotted_path in HASH_EXCLUDED_PATHS:
        cursor = data
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                cursor = None
                break
            cursor = cursor[part]
        if isinstance(cursor, dict):
            cursor.pop(parts[-1], None)
    payload = json.dumps(_canonical(data), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_experiment_config(path: Path) -> DictConfig:
    """Load one explicit YAML recipe with an optional relative ``extends``."""
    path = path.expanduser().resolve()
    child = OmegaConf.load(path)
    parent_ref = OmegaConf.select(child, "extends", default=None)
    if parent_ref is None:
        return child
    del child["extends"]
    parent = load_experiment_config(path.parent / str(parent_ref))
    return OmegaConf.merge(parent, child)


def _git_revision(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_root), *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _snapshot_source(repo_root: Path, output_dir: Path) -> dict[str, str]:
    """Keep the actual worktree, including nonignored new source, for later audits."""
    try:
        names = subprocess.check_output(
            ["git", "-C", str(repo_root), "ls-files", "-z", "--cached",
             "--others", "--exclude-standard"], stderr=subprocess.DEVNULL,
        ).decode().split("\0")
    except (OSError, subprocess.CalledProcessError):
        return {}
    checksums = {}
    with tarfile.open(output_dir / "source_snapshot.tar.gz", "w:gz") as archive:
        for name in sorted(set(names) - {""}):
            path = repo_root / name
            # Do not archive external asset links or locally generated run outputs.
            if path.is_symlink() or not path.is_file() or output_dir in path.parents:
                continue
            checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=name, recursive=False)
    (output_dir / "source_sha256.json").write_text(json.dumps(checksums, indent=2) + "\n")
    return checksums


def prepare_run(cfg: DictConfig, *, source_config: Path) -> DictConfig:
    """Resolve a YAML recipe, validate it, and persist an immutable run snapshot."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    output_root = Path(
        str(OmegaConf.select(cfg, "runtime.output_root", default="outputs"))
    )
    run_name = str(
        OmegaConf.select(cfg, "runtime.run_name", default=source_config.stem)
    )
    output_dir = OmegaConf.select(cfg, "runtime.output_dir", default=None)
    if not output_dir:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"{run_name}_{timestamp}"
    output_dir = Path(str(output_dir)).expanduser().resolve()
    cfg.runtime.output_dir = str(output_dir)
    cfg.logging.wandb_log_dir_base = str(output_dir / "wandb")
    OmegaConf.resolve(cfg)

    metrics = list(OmegaConf.select(cfg, "logging.metric_allowlist", default=[]))
    metric_count = len(metrics)  # namespace step fields are axes, not data metrics
    if metric_count > 30:
        raise ValueError(
            f"W&B metric schema has {metric_count} data metrics; maximum is 30"
        )
    if len(metrics) != len(set(metrics)):
        raise ValueError("W&B metric allowlist contains duplicate names")
    os.environ["WANDB_METRIC_ALLOWLIST"] = ",".join(metrics)
    os.environ["WANDB_DISABLE_VIDEOS"] = "1"

    output_dir.mkdir(parents=True, exist_ok=False)
    recipe_hash = scientific_recipe_hash(cfg)
    cfg.reproduction.recipe_hash = recipe_hash
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
    (output_dir / "source_config.yaml").write_text(
        source_config.read_text(encoding="utf-8"), encoding="utf-8"
    )
    repo_root = Path(__file__).resolve().parents[1]
    source_checksums = _snapshot_source(repo_root, output_dir)
    from robometer_policy_learning.runtime_metadata import collect_runtime_metadata

    policy_runtime = collect_runtime_metadata(role="policy")
    policy_runtime["lock_sha256"] = {
        name: source_checksums.get(name)
        for name in ("uv.lock", "environments/reward/requirements.lock")
    }
    (output_dir / "policy_runtime.json").write_text(json.dumps(policy_runtime, indent=2) + "\n")
    for env_key, filename in (
        ("REWARD_RUNTIME_MANIFEST", "reward_runtime.json"),
        ("ROBOMETER_MODEL_INFO_PATH", "robometer_model_info.json"),
    ):
        source_path = os.environ.get(env_key)
        if source_path:
            shutil.copyfile(source_path, output_dir / filename)
    asset_root = Path(os.environ.get("ASSET_ROOT", repo_root / "assets"))
    if (asset_root / "asset_manifest.json").is_file():
        shutil.copyfile(asset_root / "asset_manifest.json", output_dir / "asset_manifest.json")
    server_log = os.environ.get("ROBOMETER_SERVER_LOG_PATH")
    if server_log:
        (output_dir / "robometer_server.log").symlink_to(Path(server_log).resolve())
    provenance = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "recipe_hash": recipe_hash,
        "source_config": str(source_config.resolve()),
        "repository": _git_revision(repo_root),
        "source_snapshot_files": len(source_checksums),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "offline": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return cfg
