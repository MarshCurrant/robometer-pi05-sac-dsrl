"""LIBERO MLP-SAC rollout worker with a frozen Pi0.5 action decoder."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from robometer_policy_learning.rollouts.rollout_worker import RolloutWorker
from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device


class LiberoPi05RobometerRolloutWorker(RolloutWorker):
    """Collect one replay transition per Pi0.5 macro action.

    RoboMeter receives all rendered low-level frames accumulated through the
    current macro endpoint. The replay action remains the 32D SAC noise, never
    the decoded 7D environment action.
    """

    def __init__(
        self,
        *,
        pi05_policy,
        action_exec_len: int,
        macro_environment_reward: float,
        reward_relabeling_keys: List[str],
        reward_autocast_dtype: str = "bf16",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if self.num_envs != 1:
            raise ValueError("The serial LIBERO Pi0.5 DSRL worker currently requires num_envs=1")
        self.pi05 = pi05_policy
        self.action_exec_len = int(action_exec_len)
        self.macro_environment_reward = float(macro_environment_reward)
        self.reward_relabeling_keys = list(reward_relabeling_keys)
        self.reward_autocast_dtype = str(reward_autocast_dtype).lower()
        if self.reward_autocast_dtype not in {"none", "bf16", "fp16"}:
            raise ValueError(
                "reward_autocast_dtype must be one of: none, bf16, fp16"
            )
        self._episode_frames = {key: [] for key in self.reward_relabeling_keys}
        self._episode_dino_embeddings: list[Any] = []
        self._episode_text_embedding = None
        self._macro_step_in_episode = 0
        self._first_transition_audit_written = False

    @staticmethod
    def official_dsrl_macro_reward(*, step_reward: float, num_steps: int) -> float:
        """Use the final-step sparse reward for one DSRL macro transition.

        This matches ``DSRLRolloutWorker.env_step_pi0()``, which intentionally
        stores only the final low-level reward while advancing the environment
        counter by every executed action.
        """
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        return float(step_reward) if num_steps else 0.0

    def _instruction(self) -> str:
        instruction = self.env.get_language_instruction()
        if isinstance(instruction, (list, tuple, np.ndarray)):
            return str(instruction[0])
        return str(instruction)

    @staticmethod
    def _info(infos: Any) -> dict:
        if isinstance(infos, list):
            return infos[0] if infos and infos[0] is not None else {}
        return infos if isinstance(infos, dict) else {}

    def _append_frame(self, obs: Dict[str, Any]) -> None:
        for key in self.reward_relabeling_keys:
            self._episode_frames[key].append(obs[key])
        if "dino_embedding" in obs:
            self._episode_dino_embeddings.append(obs["dino_embedding"])
        if self._episode_text_embedding is None and "language" in obs:
            self._episode_text_embedding = obs["language"]

    def _reset_episode_cache(self) -> None:
        self._episode_frames = {key: [] for key in self.reward_relabeling_keys}
        self._episode_dino_embeddings = []
        self._episode_text_embedding = None
        self._macro_step_in_episode = 0

    def _reward_forward_context(self):
        """Match RoboMeter eval-server mixed precision for local scoring only."""
        has_local_reward_model = getattr(self.buffer, "reward_model", None) is not None
        if (
            not has_local_reward_model
            or self.device.type != "cuda"
            or self.reward_autocast_dtype == "none"
        ):
            return nullcontext()
        dtype = (
            torch.bfloat16
            if self.reward_autocast_dtype == "bf16"
            else torch.float16
        )
        return torch.autocast(device_type="cuda", dtype=dtype)

    def run(self, can_train: bool = True) -> Dict[str, float]:
        del can_train
        num_steps = 0
        num_episodes = 0
        last_audit_metrics: Dict[str, float] = {}

        while self.should_continue(num_steps, num_episodes):
            if self.recent_obs is None:
                obs, _ = self.env.reset()
                self.actor_state = self.get_initial_actor_state()
                self._reset_episode_cache()
            else:
                obs = self.recent_obs

            obs_i = self.extract_env_data(obs, 0)
            if not self._episode_frames[self.reward_relabeling_keys[0]]:
                self._append_frame(obs_i)

            obs_tensor = move_to_device(convert_to_tensor(obs), self.device)
            with torch.inference_mode():
                noise, self.actor_state = self.actor.act(
                    obs_tensor,
                    actor_state=self.actor_state,
                    deterministic=False,
                )
            noise_np = noise.detach().float().cpu().numpy()
            decoded_actions = self.pi05.infer(obs, noise_np, self._instruction())

            last_next_obs = obs
            last_rewards = np.zeros((1,), dtype=np.float32)
            last_dones = np.zeros((1,), dtype=bool)
            last_truncateds = np.zeros((1,), dtype=bool)
            last_infos: Any = [{}]
            actual_env_steps = 0

            for action in decoded_actions[0, : self.action_exec_len]:
                next_obs, rewards, dones, truncateds, infos = self.env.step(action[None])
                actual_env_steps += 1
                next_obs_i = self.extract_env_data(next_obs, 0)
                self._append_frame(next_obs_i)
                last_next_obs = next_obs
                last_rewards = rewards
                last_dones = dones
                last_truncateds = truncateds
                last_infos = infos
                if bool(np.asarray(dones).reshape(-1)[0]) or bool(
                    np.asarray(truncateds).reshape(-1)[0]
                ):
                    break

            next_obs_i = self.extract_env_data(last_next_obs, 0)
            # Simulator success remains hidden. Match the official DSRL worker:
            # one replay transition stores the final low-level sparse reward,
            # not a discounted sum over the executed action chunk.
            environment_reward_i = self.official_dsrl_macro_reward(
                step_reward=self.macro_environment_reward,
                num_steps=actual_env_steps,
            )
            done_i = bool(np.asarray(last_dones).reshape(-1)[0])
            truncated_i = bool(np.asarray(last_truncateds).reshape(-1)[0])
            info_i = self._info(last_infos)

            with self._reward_forward_context():
                self.buffer.add(
                    obs=obs_i,
                    action=noise_np[0],
                    reward=environment_reward_i,
                    next_obs=next_obs_i,
                    done=done_i,
                    truncated=truncated_i,
                    episode_id=self.total_episodes,
                    step_in_episode=self._macro_step_in_episode,
                    video_frames=self._episode_frames,
                    language_instruction=self._instruction(),
                    dino_embeddings=self._episode_dino_embeddings,
                    text_embedding=self._episode_text_embedding,
                )

            relabel = getattr(self.buffer, "last_relabel_result", None)
            if not relabel:
                raise RuntimeError(
                    "LIBERO Pi0.5 DSRL requires the RoboMeter buffer to return "
                    "the relabel/detection result for every macro transition"
                )
            training_reward_i = float(relabel["reward"])
            done_i = bool(relabel["done"])
            truncated_i = bool(relabel["truncated"])
            info_i.update(
                {
                    "env_reward": float(relabel["environment_reward"]),
                    "training_reward": training_reward_i,
                    "relabeled_reward": float(relabel["estimated_reward"]),
                    "success_prob": float(relabel["success_prob"]),
                    "robometer_progress": float(relabel["progress"]),
                    "robometer_detected": bool(relabel["detected"]),
                    "robometer_success_head_detected": bool(
                        relabel["head_detected"]
                    ),
                    "robometer_terminal_detected": bool(
                        relabel["terminal_detected"]
                    ),
                    "robometer_terminal_adjacent_delta": relabel[
                        "terminal_adjacent_delta"
                    ],
                }
            )
            self.buffer.update_info(
                self.total_episodes,
                self._macro_step_in_episode,
                {
                    "env_reward": float(relabel["environment_reward"]),
                    "relabeled_reward": float(relabel["estimated_reward"]),
                    "success_prob": float(relabel["success_prob"]),
                },
            )
            if not self._first_transition_audit_written and os.environ.get("RUN_ROOT"):
                audit_path = Path(os.environ["RUN_ROOT"]) / "first_transition_reward_audit.json"
                audit_path.write_text(
                    json.dumps(
                        {
                            "actual_env_steps": actual_env_steps,
                            "environment_reward": float(relabel["environment_reward"]),
                            "absolute_robometer_progress": float(
                                relabel["estimated_reward"]
                            ),
                            "training_reward": training_reward_i,
                            "formula": "environment_reward + absolute_robometer_progress",
                            "simulator_success_used_for_training": False,
                            "robometer_detected": bool(relabel["detected"]),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self._first_transition_audit_written = True
            if relabel["detected"]:
                info_i["success"] = True
            last_audit_metrics = {
                "robometer_detected": float(bool(relabel["detected"])),
                "robometer_success_head_detected": float(
                    bool(relabel["head_detected"])
                ),
                "robometer_terminal_detected": float(
                    bool(relabel["terminal_detected"])
                ),
                # This label is never exposed to the actor, critic, replay
                # reward, or training termination. It is logged only to audit
                # RoboMeter detection against the simulator's sticky outcome.
                "sim_success_once": float(bool(info_i.get("sim_success_once", False))),
            }

            self.episode_tracker.add_step(
                0,
                environment_reward_i,
                info_i,
                done_i,
                truncated_i,
                actual_env_steps,
            )
            num_steps += actual_env_steps
            self._macro_step_in_episode += 1
            self.recent_obs = last_next_obs

            if done_i or truncated_i:
                self.episode_tracker.end_episode(0)
                self.reset_actor_state_for_env(0)
                self._reset_episode_cache()
                num_episodes += 1
                self.total_episodes += 1
                # RoboMeter termination is intentionally hidden from the
                # simulator, so reset explicitly instead of relying on vector
                # environment autoreset behavior.
                self.recent_obs = None

        metrics = self.episode_tracker.get_metrics()
        metrics.update(last_audit_metrics)
        return metrics
