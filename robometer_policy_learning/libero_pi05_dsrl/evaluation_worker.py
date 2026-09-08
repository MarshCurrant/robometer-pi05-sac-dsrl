"""Evaluation worker for LIBERO MLP-SAC noise policies decoded by Pi0.5."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robometer_policy_learning.rollouts.evaluation_worker import EvaluationWorker
from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device


class LiberoPi05EvaluationWorker(EvaluationWorker):
    def __init__(self, *, pi05_policy, action_exec_len: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pi05 = pi05_policy
        self.action_exec_len = int(action_exec_len)

    def _instruction(self) -> str:
        instruction = self.eval_env.get_language_instruction()
        if isinstance(instruction, (list, tuple, np.ndarray)):
            return str(instruction[0])
        return str(instruction)

    def _decode(self, actor, obs: dict, deterministic: bool) -> np.ndarray:
        obs_device = move_to_device(convert_to_tensor(obs), self.device)
        noise, _ = actor.act(obs_device, deterministic=deterministic)
        noise_np = noise.detach().float().cpu().numpy()
        return self.pi05.infer(obs, noise_np, self._instruction())

    @staticmethod
    def _capture_observation(obs: dict, instruction: str) -> dict[str, Any]:
        """Store the official RoboMeter main-camera input without policy features."""
        frame = obs["observation/image"]
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim != 3:
            raise ValueError(f"Expected an HWC main image, got {frame.shape}")
        return {
            "main_images": np.ascontiguousarray(frame, dtype=np.uint8),
            "task_descriptions": instruction,
        }

    @staticmethod
    def _terminal_observation(infos: Any, env_index: int) -> dict | None:
        """Extract Gymnasium's pre-autoreset terminal observation when present."""
        if not isinstance(infos, dict) or "final_observation" not in infos:
            return None
        mask = infos.get("_final_observation")
        if mask is not None:
            mask_array = np.asarray(mask).reshape(-1)
            if env_index >= mask_array.size or not bool(mask_array[env_index]):
                return None
        final_observations = infos["final_observation"]
        if isinstance(final_observations, dict):
            return {
                key: (np.asarray(value)[env_index] if np.asarray(value).ndim else value)
                for key, value in final_observations.items()
            }
        values = np.asarray(final_observations, dtype=object).reshape(-1)
        if env_index >= values.size or not isinstance(values[env_index], dict):
            return None
        return values[env_index]

    @staticmethod
    def _trajectory_output_dir() -> Path | None:
        enabled = os.environ.get("SAVE_STEP0_TRAJECTORIES", "0").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return None
        configured = os.environ.get("STEP0_TRAJECTORY_DIR")
        run_root = os.environ.get("RUN_ROOT")
        if configured:
            return Path(configured)
        if run_root:
            return Path(run_root) / "step0_trajectories"
        raise RuntimeError(
            "SAVE_STEP0_TRAJECTORIES requires STEP0_TRAJECTORY_DIR or RUN_ROOT"
        )

    def _run_evaluations(self, actor, num_episodes: int = 10):
        rewards_all = []
        steps_all = []
        success_all = []
        episode_rows = []
        trajectory_dir = self._trajectory_output_dir()
        if trajectory_dir is not None:
            trajectory_dir.mkdir(parents=True, exist_ok=True)
        trajectory_tag = os.environ.get("STEP0_TRAJECTORY_TAG", "step0")
        trajectory_env_index = int(os.environ.get("STEP0_ENV_INDEX", "0"))
        actor_deterministic = os.environ.get(
            "STEP0_ACTOR_DETERMINISTIC", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        autoreset_obs = None
        autoreset_infos = None

        for episode_index in range(num_episodes):
            if autoreset_obs is None:
                obs, reset_infos = self.eval_env.reset()
            else:
                # Gymnasium 0.29 SyncVectorEnv resets a completed sub-environment
                # inside step(). Reuse that observation instead of resetting again,
                # which would consume and skip the next procedural LIBERO state.
                obs, reset_infos = autoreset_obs, autoreset_infos
                autoreset_obs = None
                autoreset_infos = None
            reset_info = self._extract_info(reset_infos, 0)
            init_fingerprint = reset_info.get("initial_observation_fingerprint")
            if init_fingerprint is not None:
                init_fingerprint = int(np.asarray(init_fingerprint).reshape(-1)[0])
            total_reward = 0.0
            total_steps = 0
            success = False
            done = False
            action_hasher = hashlib.sha256()
            instruction = self._instruction() if trajectory_dir is not None else ""
            captured_observations = []
            captured_actions = []
            captured_infos = []
            if trajectory_dir is not None:
                captured_observations.append(
                    self._capture_observation(obs, instruction)
                )
                captured_infos.append({"success_once": False})

            while not done:
                actions = self._decode(
                    actor, obs, deterministic=actor_deterministic
                )
                for action in actions[0, : self.action_exec_len]:
                    action_hasher.update(
                        np.ascontiguousarray(action, dtype=np.float32).tobytes()
                    )
                    obs, rewards, dones, truncateds, infos = self.eval_env.step(action[None])
                    reward = float(np.asarray(rewards).reshape(-1)[0])
                    terminated = bool(np.asarray(dones).reshape(-1)[0])
                    truncated = bool(np.asarray(truncateds).reshape(-1)[0])
                    info = self._extract_info(infos, 0)
                    total_reward += reward
                    total_steps += 1
                    # Native-termination eval exposes success via ``terminated``;
                    # no-early-stop eval exposes the sticky simulator label via
                    # ``sim_success_once``. Keep both because Gymnasium vector
                    # auto-reset can move terminal info under ``final_info``.
                    success = success or terminated or bool(
                        info.get("is_success", False)
                        or info.get("success", False)
                        or info.get("sim_success_once", False)
                    )
                    done = terminated or truncated
                    if trajectory_dir is not None:
                        captured_actions.append(
                            np.ascontiguousarray(action, dtype=np.float32)
                        )
                        transition_obs = (
                            self._terminal_observation(infos, 0) if done else None
                        )
                        captured_observations.append(
                            self._capture_observation(
                                transition_obs if transition_obs is not None else obs,
                                instruction,
                            )
                        )
                        captured_infos.append({"success_once": bool(success)})
                    if done:
                        if self._contains_autoreset_transition(infos, 0):
                            autoreset_obs = obs
                            autoreset_infos = infos
                        break

            rewards_all.append(total_reward)
            steps_all.append(total_steps)
            success_all.append(success)
            episode_rows.append(
                {
                    "episode_index": episode_index,
                    "initial_observation_fingerprint": init_fingerprint,
                    "success": bool(success),
                    "steps": int(total_steps),
                    "reward": float(total_reward),
                    "action_sha256": action_hasher.hexdigest(),
                }
            )
            if trajectory_dir is not None:
                if len(captured_observations) != len(captured_actions) + 1:
                    raise RuntimeError("Captured trajectory state/action alignment failed")
                if len(captured_infos) != len(captured_observations):
                    raise RuntimeError("Captured trajectory info/state alignment failed")
                suffix = "success" if success else "fail"
                trajectory_path = trajectory_dir / (
                    f"{trajectory_tag}_env_{trajectory_env_index}_episode_"
                    f"{episode_index:03d}_{suffix}.pkl"
                )
                with trajectory_path.open("wb") as handle:
                    pickle.dump(
                        {
                            "observations": captured_observations,
                            "actions": captured_actions,
                            "infos": captured_infos,
                            "success": bool(success),
                            "env_idx": trajectory_env_index,
                            "episode_id": episode_index,
                            "actor_deterministic": actor_deterministic,
                        },
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

        run_root = os.environ.get("RUN_ROOT")
        if run_root:
            manifest_path = Path(run_root) / "step0_eval_episodes.jsonl"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("w", encoding="utf-8") as handle:
                for row in episode_rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")

        return {
            "avg_reward": float(np.mean(rewards_all)),
            "std_reward": float(np.std(rewards_all)),
            "min_reward": float(np.min(rewards_all)),
            "max_reward": float(np.max(rewards_all)),
            "avg_steps": float(np.mean(steps_all)),
            "success_rate": float(np.mean(success_all)),
            "num_eval_episodes": num_episodes,
            "unique_init_states": len(
                {
                    row["initial_observation_fingerprint"]
                    for row in episode_rows
                    if row["initial_observation_fingerprint"] is not None
                }
            ),
        }

    @staticmethod
    def _contains_autoreset_transition(infos: Any, env_index: int) -> bool:
        """Return whether vector ``step`` already reset this environment."""
        if not isinstance(infos, dict) or "final_observation" not in infos:
            return False
        mask = infos.get("_final_observation")
        if mask is None:
            return True
        mask_array = np.asarray(mask).reshape(-1)
        return env_index < mask_array.size and bool(mask_array[env_index])

    def _record_evaluation_video(self, actor):
        # Keep the MLP logger's metric schema stable. Formal success metrics are
        # produced by _run_evaluations; video recording is intentionally left to
        # the existing LIBERO recorder until it supports macro-action adapters.
        return {
            "video_reward": 0.0,
            "video_steps": 0,
            "video_saved": False,
            "video_success": False,
        }
