#!/usr/bin/env python3
"""Download pinned model and simulator assets from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="pi05 robometer qwen dino libero")
    parser.add_argument("--all", action="store_true", help="Download every pinned asset")
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="Check local files without downloading")
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

    asset_root.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        spec = cfg.assets[name]
        destination = asset_root / str(spec.destination)
        if not args.check:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            print(f"Downloading {name}: {spec.repo_id}@{spec.revision} -> {destination}")
            snapshot_download(
                repo_id=str(spec.repo_id),
                repo_type=str(spec.repo_type),
                revision=str(spec.revision),
                local_dir=destination,
            )
        missing = [item for item in required_files(name) if not (destination / item).exists()]
        record = {
            "name": name,
            "repo_id": str(spec.repo_id),
            "repo_type": str(spec.repo_type),
            "revision": str(spec.revision),
            "destination": str(destination),
            "ok": not missing,
            "missing": missing,
        }
        records.append(record)
        print(f"{name}: {'ok' if record['ok'] else 'MISSING ' + ', '.join(missing)}")

    manifest = {
        "schema_version": 1,
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
