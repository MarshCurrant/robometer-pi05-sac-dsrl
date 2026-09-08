#!/usr/bin/env python3
"""Validate assets, imports, configuration, and LIBERO task identity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from robometer_policy_learning.reproducibility import (
    load_experiment_config,
    scientific_recipe_hash,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    if args.set:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))
    OmegaConf.resolve(cfg)

    checks = {}
    required_asset_env = ("ROBOMETER_MODEL", "ROBOMETER_BASE_MODEL", "LIBERO_ASSETS")
    asset_env = {}
    for name in required_asset_env:
        value = os.environ.get(name)
        checks[f"environment/{name}"] = {
            "value": value,
            "ok": bool(value),
            "hint": None if value else "Run: source scripts/load_env.sh",
        }
        if value:
            asset_env[name] = Path(value)

    files = {
        "pi05_weights": Path(str(cfg.dsrl.pi05_checkpoint)) / "model.safetensors",
        "pi05_norm_stats": Path(str(cfg.dsrl.pi05_checkpoint))
        / "physical-intelligence/libero/norm_stats.json",
        "dino_config": Path(str(cfg.model.dinov2_model)) / "config.json",
    }
    if "ROBOMETER_MODEL" in asset_env:
        files["robometer_index"] = (
            asset_env["ROBOMETER_MODEL"] / "model.safetensors.index.json"
        )
    if "ROBOMETER_BASE_MODEL" in asset_env:
        files["qwen_index"] = (
            asset_env["ROBOMETER_BASE_MODEL"] / "model.safetensors.index.json"
        )
    if "LIBERO_ASSETS" in asset_env:
        files["libero_asset"] = (
            asset_env["LIBERO_ASSETS"] / "stable_scanned_objects/akita_black_bowl"
        )
        files["libero_scene"] = (
            asset_env["LIBERO_ASSETS"] / "scenes/libero_tabletop_base_style.xml"
        )
    for name, path in files.items():
        checks[name] = {"path": str(path), "ok": path.exists()}

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[str(cfg.env.env_name)]()
    task = suite.get_task(int(cfg.env.task_id))
    actual_task_name = str(task.name)
    checks["task_identity"] = {
        "expected": str(cfg.env.task_name),
        "actual": actual_task_name,
        "ok": actual_task_name == str(cfg.env.task_name),
    }
    threshold = OmegaConf.select(
        cfg, "reward_model.success_detection_threshold", default=None
    )
    terminal_threshold = OmegaConf.select(
        cfg, "reward_model.terminal_adjacent_delta_threshold", default=None
    )
    checks["detector_calibration"] = {
        "success_detection_threshold": threshold,
        "terminal_adjacent_delta_threshold": terminal_threshold,
        "ok": threshold is not None and terminal_threshold is not None,
    }
    metrics = list(cfg.logging.metric_allowlist)
    effective_metric_count = len(metrics) + 1
    checks["wandb_metric_count"] = {
        "allowlisted_metrics": len(metrics),
        "implicit_axes": ["env_step"],
        "count": effective_metric_count,
        "ok": effective_metric_count <= 30,
    }
    checks["recipe_hash"] = {"value": scientific_recipe_hash(cfg), "ok": True}
    report = {"schema_version": 1, "checks": checks, "ok": all(v["ok"] for v in checks.values())}
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
