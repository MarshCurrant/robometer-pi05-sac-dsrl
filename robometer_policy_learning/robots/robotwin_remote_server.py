#!/usr/bin/env python3
"""RoboTwin socket environment for the official RoboMeter DSRL runner.

The transport and episode lifecycle match ``droid_remote_server.py``.  This
adapter only translates RoboTwin's bimanual ALOHA observations/actions; reward
relabeling, success detection, replay, and SAC remain in the upstream policy
learning implementation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from robometer_policy_learning.robots.remote_server_utils import recv_msg, send_msg


def _load_manifest(path: Path, task_name: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("task", task_name) == task_name]
    if not rows:
        raise ValueError(f"No rows for task={task_name} in {path}")
    for row in rows:
        if "seed" not in row or not row.get("instruction"):
            raise ValueError(f"Manifest row lacks seed/instruction: {row}")
    return rows


def _load_robotwin_args(root: Path, task_name: str, task_config: str) -> dict[str, Any]:
    config_path = root / "task_config" / f"{task_config}.yml"
    args = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    args.update(task_name=task_name, task_config=task_config, eval_mode=True, policy_name="robometer_dsrl")

    from envs import CONFIGS_PATH

    embodiment_types = yaml.safe_load((Path(CONFIGS_PATH) / "_embodiment_config.yml").read_text(encoding="utf-8"))
    camera_types = yaml.safe_load((Path(CONFIGS_PATH) / "_camera_config.yml").read_text(encoding="utf-8"))
    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        left_file = right_file = embodiment_types[embodiment[0]]["file_path"]
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        left_file = embodiment_types[embodiment[0]]["file_path"]
        right_file = embodiment_types[embodiment[1]]["file_path"]
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"Unsupported embodiment config: {embodiment}")

    def embodiment_config(path: str) -> dict[str, Any]:
        return yaml.safe_load((Path(path) / "config.yml").read_text(encoding="utf-8"))

    args["left_robot_file"] = left_file
    args["right_robot_file"] = right_file
    args["left_embodiment_config"] = embodiment_config(left_file)
    args["right_embodiment_config"] = embodiment_config(right_file)
    head_camera = camera_types[args["camera"]["head_camera_type"]]
    args["head_camera_h"] = head_camera["h"]
    args["head_camera_w"] = head_camera["w"]
    args["render_freq"] = 0
    args["eval_video_log"] = False
    args["eval_video_save_dir"] = None
    args["save_data"] = False
    return args


class RoboTwinSession:
    def __init__(
        self,
        root: Path,
        task_name: str,
        task_config: str,
        manifest: Path,
        max_steps: int,
        *,
        success_mode: str = "robometer_enhanced",
        seed_mode: str = "manifest_cycle",
        seed_min: int = 0,
        seed_max: int = 1_000_000,
        sampling_seed: int = 0,
        random_pool_size: int = 20,
        seed_log: Path | None = None,
        max_reset_attempts: int = 20,
    ):
        self.root = root
        self.task_name = task_name
        self.args = _load_robotwin_args(root, task_name, task_config)
        self.rows = _load_manifest(manifest, task_name)
        self.max_steps = max_steps
        if success_mode not in {"robometer_enhanced", "environment"}:
            raise ValueError(f"Unsupported success_mode={success_mode!r}")
        if seed_mode not in {"manifest_cycle", "manifest_shuffle", "random", "random_pool"}:
            raise ValueError(f"Unsupported seed_mode={seed_mode!r}")
        if seed_max <= seed_min:
            raise ValueError("seed_max must be greater than seed_min")
        if random_pool_size <= 0:
            raise ValueError("random_pool_size must be positive")
        self.success_mode = success_mode
        self.seed_mode = seed_mode
        self.seed_min = seed_min
        self.seed_max = seed_max
        self.sampling_seed = sampling_seed
        self.random_pool_size = random_pool_size
        self.max_reset_attempts = max_reset_attempts
        self.seed_log = seed_log
        self.seed_rng = random.Random(sampling_seed)
        self.seed_draw_index = 0
        self.used_random_seeds: set[int] = set()
        self.random_seed_pool: list[int] = []
        if self.seed_mode == "manifest_shuffle":
            self.seed_rng.shuffle(self.rows)
        elif self.seed_mode == "random_pool":
            self.random_seed_pool = self.seed_rng.sample(
                range(self.seed_min, self.seed_max), self.random_pool_size
            )
        if self.seed_log is not None:
            self.seed_log.parent.mkdir(parents=True, exist_ok=True)
        self.episode_index = 0
        module = importlib.import_module(f"envs.{self.task_name}")
        # RoboTwin's official evaluator reuses one TASK_ENV object across all
        # episodes. Creating a fresh object per reset retains SAPIEN renderer
        # reference cycles and exhausts the Vulkan device after about 10 resets.
        self.env = getattr(module, self.task_name)()
        self.episode_active = False
        self.clear_cache_freq = int(self.args.get("clear_cache_freq", 5))
        self.prompt = ""
        self.seed = -1
        self.step_count = 0
        self.actual_success = False

    def _next_seed_and_instruction(self) -> tuple[int, str]:
        row = self.rows[self.episode_index % len(self.rows)]
        instruction = str(row["instruction"])
        if self.seed_mode in {"manifest_cycle", "manifest_shuffle"}:
            return int(row["seed"]), instruction
        if self.seed_mode == "random_pool":
            seed = self.random_seed_pool[self.seed_draw_index % len(self.random_seed_pool)]
            self.seed_draw_index += 1
            return seed, instruction
        if len(self.used_random_seeds) >= self.seed_max - self.seed_min:
            self.used_random_seeds.clear()
        while True:
            seed = self.seed_rng.randrange(self.seed_min, self.seed_max)
            if seed not in self.used_random_seeds:
                self.used_random_seeds.add(seed)
                return seed, instruction

    def _record_seed(self) -> None:
        if self.seed_log is None:
            return
        row = {
            "episode_index": self.episode_index,
            "seed": self.seed,
            "task": self.task_name,
            "instruction": self.prompt,
            "seed_mode": self.seed_mode,
            "sampling_seed": self.sampling_seed,
            "success_mode": self.success_mode,
            "timestamp": time.time(),
        }
        with self.seed_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def close_episode(self) -> None:
        if self.episode_active:
            clear_cache = self.clear_cache_freq > 0 and self.episode_index % self.clear_cache_freq == 0
            self.env.close_env(clear_cache=clear_cache)
            self.episode_active = False

    def reset(self) -> dict[str, Any]:
        self.close_episode()
        last_error = None
        for attempt in range(self.max_reset_attempts):
            self.seed, self.prompt = self._next_seed_and_instruction()
            try:
                self.env.setup_demo(
                    now_ep_num=self.episode_index,
                    seed=self.seed,
                    is_test=True,
                    **self.args,
                )
                break
            except Exception as exc:
                last_error = exc
                try:
                    self.env.close_env(clear_cache=True)
                except Exception:
                    pass
                if self.seed_mode in {"manifest_cycle", "manifest_shuffle"}:
                    raise
                print(
                    f"RoboTwin reset rejected seed={self.seed} "
                    f"(attempt {attempt + 1}/{self.max_reset_attempts}): {exc!r}",
                    flush=True,
                )
        else:
            raise RuntimeError(
                f"Unable to initialize RoboTwin after {self.max_reset_attempts} random seeds"
            ) from last_error
        self.episode_index += 1
        self.episode_active = True
        self.env.step_lim = self.max_steps
        self.env.set_instruction(instruction=self.prompt)
        self.step_count = 0
        self.actual_success = False
        self._record_seed()
        return self.observation(
            info={
                "supports_action_chunking": True,
                "robotwin_task": self.task_name,
                "robotwin_seed": self.seed,
                "paper_timeout_steps": self.max_steps,
                "seed_mode": self.seed_mode,
                "success_mode": self.success_mode,
            }
        )

    @staticmethod
    def _wrist_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_img = Image.fromarray(left).resize((112, 224), Image.Resampling.BILINEAR)
        right_img = Image.fromarray(right).resize((112, 224), Image.Resampling.BILINEAR)
        return np.concatenate([np.asarray(left_img), np.asarray(right_img)], axis=1)

    def observation(self, *, info: dict[str, Any] | None = None, **episode_fields: Any) -> dict[str, Any]:
        raw = self.env.get_obs()
        cameras = raw["observation"]
        head = np.asarray(cameras["head_camera"]["rgb"], dtype=np.uint8)
        left = np.asarray(cameras["left_camera"]["rgb"], dtype=np.uint8)
        right = np.asarray(cameras["right_camera"]["rgb"], dtype=np.uint8)
        state = np.asarray(raw["joint_action"]["vector"], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"Expected RoboTwin ALOHA state (14,), got {state.shape}")
        return {
            "observation.images.cam_high": head,
            "observation.images.cam_left_wrist": left,
            "observation.images.cam_right_wrist": right,
            "observation.images.wrist_pair": self._wrist_pair(left, right),
            "observation.state": state,
            "prompt": self.prompt,
            "info": info or {},
            **episode_fields,
        }

    def step(self, action: np.ndarray) -> dict[str, Any]:
        actions = np.asarray(action, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"Expected action chunk (N,14), got {actions.shape}")

        executed = 0
        for single_action in actions:
            if self.step_count >= self.max_steps:
                break
            self.env.take_action(single_action)
            self.step_count += 1
            executed += 1
            self.actual_success = self.actual_success or bool(self.env.eval_success)

        environment_done = self.success_mode == "environment" and self.actual_success
        truncated = self.step_count >= self.max_steps and not environment_done
        return self.observation(
            reward=0.0,
            done=environment_done,
            truncated=truncated,
            success=environment_done,
            num_steps=executed,
            info={
                "actual_env_success": self.actual_success,
                "robotwin_seed": self.seed,
                "robotwin_step_count": self.step_count,
                "success_mode": self.success_mode,
                "termination_source": "environment" if environment_done else "timeout" if truncated else None,
                "timeout": truncated,
            },
        )


def serve(args: argparse.Namespace) -> None:
    root = Path(args.robotwin_root).expanduser().resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "script"))
    from test_render import Sapien_TEST

    Sapien_TEST()
    session = RoboTwinSession(
        root,
        args.task_name,
        args.task_config,
        Path(args.manifest),
        args.max_steps,
        success_mode=args.success_mode,
        seed_mode=args.seed_mode,
        seed_min=args.seed_min,
        seed_max=args.seed_max,
        sampling_seed=args.sampling_seed,
        random_pool_size=args.random_pool_size,
        seed_log=Path(args.seed_log) if args.seed_log else None,
        max_reset_attempts=args.max_reset_attempts,
    )
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"RoboTwin DSRL server listening on {args.host}:{args.port}", flush=True)
    try:
        while True:
            conn, address = server.accept()
            print(f"DSRL client connected from {address}", flush=True)
            with conn:
                while True:
                    command = recv_msg(conn)
                    if command is None or command.get("type") == "CLOSE":
                        break
                    try:
                        if command["type"] == "RESET":
                            send_msg(conn, session.reset())
                        elif command["type"] == "STEP":
                            send_msg(conn, session.step(np.asarray(command["action"], dtype=np.float32)))
                        elif command["type"] == "SUCCESS_CHECK":
                            robometer_done = args.success_mode == "robometer_enhanced"
                            send_msg(
                                conn,
                                {
                                    "done": robometer_done or session.actual_success,
                                    "blocked": False,
                                    "info": {
                                        "actual_env_success": session.actual_success,
                                        "robotwin_step_count": session.step_count,
                                        "success_mode": args.success_mode,
                                        "termination_source": (
                                            "robometer_enhanced"
                                            if robometer_done
                                            else "environment"
                                            if session.actual_success
                                            else None
                                        ),
                                    },
                                },
                            )
                        else:
                            raise ValueError(f"Unknown command: {command}")
                    except Exception as exc:
                        send_msg(conn, {"error": repr(exc)})
                        raise
    finally:
        session.close_episode()
        server.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--task-name", default="beat_block_hammer")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument(
        "--success-mode",
        choices=["robometer_enhanced", "environment"],
        default="robometer_enhanced",
    )
    parser.add_argument(
        "--seed-mode",
        choices=["manifest_cycle", "manifest_shuffle", "random", "random_pool"],
        default="manifest_cycle",
    )
    parser.add_argument("--seed-min", type=int, default=0)
    parser.add_argument("--seed-max", type=int, default=1_000_000)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--random-pool-size", type=int, default=20)
    parser.add_argument("--seed-log", default=None)
    parser.add_argument("--max-reset-attempts", type=int, default=20)
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
