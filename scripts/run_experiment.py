#!/usr/bin/env python3
"""Launch the isolated RoboMeter server and the policy learner."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import requests
from omegaconf import OmegaConf

from robometer_policy_learning.reproducibility import load_experiment_config
from robometer_policy_learning.runtime_metadata import write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def validate_model_info(info: dict) -> dict:
    serialized = json.dumps(info).lower()
    if "qwen3" not in serialized:
        raise RuntimeError("RoboMeter server is not backed by Qwen3-VL")
    if "qwen2_5" in serialized or "qwen2.5" in serialized:
        raise RuntimeError("RoboMeter server reports a Qwen2.5/Qwen3 model mismatch")
    return info


def wait_for_server(url: str, process: subprocess.Popen, timeout: float = 900.0) -> dict:
    deadline = time.monotonic() + timeout
    last_error = "not started"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"RoboMeter server exited with code {process.returncode}")
        try:
            response = requests.get(f"{url}/model_info", timeout=5)
            response.raise_for_status()
            return validate_model_info(response.json())
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
            time.sleep(3)
    raise TimeoutError(f"RoboMeter server did not become healthy: {last_error}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_experiment_config(args.config)
    if args.set:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))
    OmegaConf.resolve(cfg)

    policy_python = Path(os.environ.get("PYTHON_BIN", repo_root / ".venv-policy/bin/python"))
    reward_python = Path(os.environ.get("REWARD_PYTHON_BIN", repo_root / ".venv-reward/bin/python"))
    if not policy_python.is_file():
        raise SystemExit(f"Policy Python not found: {policy_python}; run scripts/setup.sh")
    if not reward_python.is_file() and not args.no_start_server:
        raise SystemExit(f"Reward Python not found: {reward_python}; run scripts/setup.sh")

    preflight = [str(policy_python), str(repo_root / "scripts/preflight.py"), "--config", str(args.config)]
    for override in args.set:
        preflight.extend(["--set", override])
    subprocess.run(preflight, check=True, env=os.environ.copy())
    if args.preflight_only:
        return

    service_dir = Path(str(cfg.runtime.output_root)).resolve() / ".services"
    service_dir.mkdir(parents=True, exist_ok=True)
    server = None
    server_log_handle = None
    host = str(cfg.resources.reward_server.host)
    port = int(cfg.resources.reward_server.port)
    server_url = f"http://{host}:{port}"
    service_dir = Path(tempfile.mkdtemp(prefix=f"robometer_{port}_", dir=service_dir))
    runtime_manifest = service_dir / "reward_runtime_manifest.json"
    model_info_path = service_dir / "model_info.json"
    server_log = None
    try:
        if not args.no_start_server:
            server_log = service_dir / "server.log"
            server_log_handle = server_log.open("x", encoding="utf-8")
            reward_env = os.environ.copy()
            reward_env["CUDA_VISIBLE_DEVICES"] = str(cfg.resources.reward_server.gpu)
            reward_env["ROBOMETER_BASE_MODEL_ID"] = os.environ["ROBOMETER_BASE_MODEL"]
            reward_env["HF_HUB_OFFLINE"] = "1"
            reward_env["TRANSFORMERS_OFFLINE"] = "1"
            subprocess.run(
                [
                    str(reward_python),
                    str(repo_root / "robometer_policy_learning/runtime_metadata.py"),
                    "--output",
                    str(runtime_manifest),
                ],
                cwd=repo_root,
                env=reward_env,
                check=True,
                timeout=120,
                stdout=server_log_handle,
                stderr=subprocess.STDOUT,
            )
            command = [
                str(reward_python),
                "-m",
                "robometer.evals.eval_server",
                f"model_path={os.environ['ROBOMETER_MODEL']}",
                "num_gpus=1",
                f"max_workers={int(cfg.resources.reward_server.max_workers)}",
                f"batch_size={int(cfg.resources.reward_server.batch_size)}",
                f"server_url={host}",
                f"server_port={port}",
                f"autocast_dtype={cfg.resources.reward_server.autocast_dtype}",
                f"use_unsloth={str(bool(cfg.resources.reward_server.use_unsloth)).lower()}",
            ]
            server = subprocess.Popen(
                command,
                cwd=repo_root,
                env=reward_env,
                stdout=server_log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            info = wait_for_server(server_url, server)
        else:
            response = requests.get(f"{server_url}/model_info", timeout=10)
            response.raise_for_status()
            info = validate_model_info(response.json())
            write_manifest(
                runtime_manifest,
                {
                    "schema_version": 1,
                    "status": "unavailable",
                    "source": "external_server",
                    "reason": "Launcher did not start this server; its interpreter is unknown.",
                },
            )
        write_manifest(model_info_path, info)

        policy_env = os.environ.copy()
        policy_env["CUDA_VISIBLE_DEVICES"] = str(cfg.resources.policy_gpu)
        policy_env["HF_HUB_OFFLINE"] = "1"
        policy_env["TRANSFORMERS_OFFLINE"] = "1"
        policy_env["REWARD_RUNTIME_MANIFEST"] = str(runtime_manifest)
        policy_env["ROBOMETER_MODEL_INFO_PATH"] = str(model_info_path)
        policy_env.pop("ROBOMETER_SERVER_LOG_PATH", None)
        if server_log is not None:
            policy_env["ROBOMETER_SERVER_LOG_PATH"] = str(server_log)
        command = [str(policy_python), str(repo_root / "scripts/train.py"), "--config", str(args.config)]
        for override in args.set:
            command.extend(["--set", override])
        raise SystemExit(
            subprocess.run(command, cwd=repo_root, env=policy_env, check=False).returncode
        )
    finally:
        stop_process(server)
        if server_log_handle is not None:
            server_log_handle.close()


if __name__ == "__main__":
    main()
