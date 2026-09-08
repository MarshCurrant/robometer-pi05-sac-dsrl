import torch
import numpy as np
import gymnasium as gym
from loguru import logger
from typing import Dict, List, Any, Optional, Tuple
from robometer_policy_learning.modules.base.modeling_actor import BaseActor
from robometer_policy_learning.rollouts.episode_data import EpisodeData
from robometer_policy_learning.utils.gpu_utils import move_to_device, convert_to_tensor
from robometer_policy_learning.rollouts.rollout_worker import RolloutWorker
from robometer_policy_learning.rollouts.dsrl_rollout_worker import DSRLRolloutWorker

from robometer_policy_learning.utils.pi0_integration import Pi0Wrapper
from robometer_policy_learning.utils.dsrl_utils import ActionQueue, format_obs_for_storage, resize_images
from robometer_policy_learning.envs.dsrl_env_wrappers import DummyDSRLEnv


class RobometerRolloutWorker(RolloutWorker):
    """
    Rollout worker for Robometer.
    """

    def __init__(self, reward_relabeling_keys: List[str] = ["image"], **kwargs):
        super().__init__(**kwargs)
        self.reward_relabeling_keys = reward_relabeling_keys

    def run(self, can_train=True) -> Dict[str, float]:
        """Main rollout loop."""
        # Reset for new rollout
        num_steps = 0
        num_episodes = 0

        # Process each environment
        video_frames_all_envs = [{key: [] for key in self.reward_relabeling_keys} for _ in range(self.num_envs)]
        dino_embeddings_all_envs = [[] for _ in range(self.num_envs)]
        text_embeddings_all_envs = [None for _ in range(self.num_envs)]
        while self.should_continue(num_steps, num_episodes):
            # Get observations
            if self.recent_obs is None:
                obs, _ = self.env.reset()
                self.actor_state = self.get_initial_actor_state()
            else:
                obs = self.recent_obs

            obs_tensor = convert_to_tensor(obs)
            obs_tensor = move_to_device(obs_tensor, self.device)

            # Ensure initial frame is captured at the start of every episode (including after per-env resets)
            for i in range(self.num_envs):
                if self.should_collect_from_env(i) and len(video_frames_all_envs[i]) == 0:
                    obs_i_initial = self.extract_env_data(obs, i)
                    for key in self.reward_relabeling_keys:
                        video_frames_all_envs[i][key].append(obs_i_initial[key])
                    # Collect initial embeddings if they exist
                    if "dino_embedding" in obs_i_initial:
                        dino_embeddings_all_envs[i].append(obs_i_initial["dino_embedding"])
                    if "language" in obs_i_initial and text_embeddings_all_envs[i] is None:
                        text_embeddings_all_envs[i] = obs_i_initial["language"]

            # If chunked, we need compute actions when the chunk is empty
            if self.is_chunked_rollout:
                if self.env.is_chunk_empty:
                    with torch.inference_mode():
                        actions, self.actor_state = self.actor.act(
                            obs_tensor, actor_state=self.actor_state, deterministic=False
                        )
                    actions = actions.detach().cpu().numpy()

                    # Step environment
                    next_obs, rewards, dones, truncateds, infos = self.env.step(actions)
                else:
                    actions = None
                    # Actions will be None because we are taking actions from the wrapper
                    next_obs, rewards, dones, truncateds, infos = self.env.step(actions)

                # Update actions to grab the actual action that was taken
                actions = self.env._get_last_action()
            else:
                # Otherwise just take the action like normal
                with torch.inference_mode():
                    actions, self.actor_state = self.actor.act(
                        obs_tensor, actor_state=self.actor_state, deterministic=False
                    )
                actions = actions.detach().cpu().numpy()
                next_obs, rewards, dones, truncateds, infos = self.env.step(actions)

            # Process each environment
            for i in range(self.num_envs):
                if not self.should_collect_from_env(i):
                    continue

                # Extract data for this environment
                obs_i = self.extract_env_data(obs, i)
                next_obs_i = self.extract_env_data(next_obs, i)
                for key in self.reward_relabeling_keys:
                    video_frames_all_envs[i][key].append(next_obs_i[key])
                # Collect embeddings if they exist, if there are multiple keys, the dino embeddings will be concatenated.
                if "dino_embedding" in next_obs_i:
                    dino_embeddings_all_envs[i].append(next_obs_i["dino_embedding"])
                if "language" in next_obs_i and text_embeddings_all_envs[i] is None:
                    text_embeddings_all_envs[i] = next_obs_i["language"]

                action_i = self.process_action(actions[i])
                reward_i = float(rewards[i])
                done_i = bool(dones[i])
                truncated_i = bool(truncateds[i])

                # Get info safely
                info_i = {}
                if infos is not None:
                    if isinstance(infos, list) and i < len(infos):
                        info_i = infos[i] if infos[i] is not None else {}
                    elif isinstance(infos, dict):
                        info_i = infos  # Single info dict for all envs

                # Extract success info (if episode is done/truncated)
                success_info = {}
                if done_i or truncated_i:
                    # Pass success indicators from info to buffer
                    if "is_success" in info_i:
                        success_info["is_success"] = info_i["is_success"]
                    elif "success" in info_i:
                        success_info["success"] = info_i["success"]

                # Add to buffer
                language_instruction = self.env.get_language_instruction()
                # Extract step_in_episode from info if available (from async reward relabel wrapper)
                step_in_episode = info_i.get("step_in_episode", None)
                if step_in_episode is not None:
                    # Convert to Python int if it's a numpy array/scalar (numpy types aren't hashable)
                    if hasattr(step_in_episode, "item"):
                        step_in_episode = int(step_in_episode.item())
                    else:
                        step_in_episode = int(step_in_episode)
                self.buffer.add(
                    obs=obs_i,
                    action=action_i,
                    reward=reward_i,
                    next_obs=next_obs_i,
                    done=done_i,
                    truncated=truncated_i,
                    episode_id=self.total_episodes,
                    step_in_episode=step_in_episode,
                    video_frames=video_frames_all_envs[i],
                    language_instruction=language_instruction,
                    dino_embeddings=dino_embeddings_all_envs[i],
                    text_embedding=text_embeddings_all_envs[i],
                    **success_info,
                )

                # Update episode tracking
                self.episode_tracker.add_step(i, reward_i, info_i, done_i, truncated_i)
                num_steps += 1

                if done_i or truncated_i:
                    self.episode_tracker.end_episode(i)
                    video_frames_all_envs[i] = {key: [] for key in self.reward_relabeling_keys}
                    dino_embeddings_all_envs[i] = []
                    text_embeddings_all_envs[i] = None
                    self.reset_actor_state_for_env(i)
                    num_episodes += 1
                    self.total_episodes += 1
            self.recent_obs = next_obs

        return self.episode_tracker.get_metrics()


class DSRLwithRobometerRolloutWorker(DSRLRolloutWorker):
    def __init__(
        self,
        pi0_wrapper: Pi0Wrapper,
        action_exec_len: int,
        gamma: float = 0.99,
        dummy_dsrl_env: DummyDSRLEnv = None,
        reward_relabeling_keys: List[str] = ["image"],
        **kwargs,
    ):
        super().__init__(
            pi0_wrapper=pi0_wrapper,
            action_exec_len=action_exec_len,
            gamma=gamma,
            dummy_dsrl_env=dummy_dsrl_env,
            **kwargs,
        )
        self.reward_relabeling_keys = reward_relabeling_keys

    def run(self, can_train=True) -> Dict[str, float]:
        """Main rollout loop."""
        # Reset for new rollout
        num_steps = 0
        num_episodes = 0

        video_frames_all_envs = [{key: [] for key in self.reward_relabeling_keys} for _ in range(self.num_envs)]
        dino_embeddings_all_envs = [[] for _ in range(self.num_envs)]
        text_embeddings_all_envs = [None for _ in range(self.num_envs)]
        while self.should_continue(num_steps, num_episodes):
            dsrl_obs, dsrl_next_obs, dsrl_actions, rewards, dones, truncateds, infos, actual_env_steps = (
                self.env_step_pi0([0], can_train=can_train)
            )

            # Process each environment
            for i in range(self.num_envs):
                if not self.should_collect_from_env(i):
                    continue

                # Extract data for this environment
                obs_i = self.extract_env_data(dsrl_obs, i)
                if all(len(video_frames_all_envs[i][key]) == 0 for key in self.reward_relabeling_keys):
                    # Add initial frame to video frames
                    for key in self.reward_relabeling_keys:
                        video_frames_all_envs[i][key].append(obs_i[key])
                    # Collect initial embeddings if they exist
                    if "dino_embedding" in obs_i:
                        dino_embeddings_all_envs[i].append(obs_i["dino_embedding"])
                    if "language" in obs_i and text_embeddings_all_envs[i] is None:
                        text_embeddings_all_envs[i] = obs_i["language"]

                next_obs_i = self.extract_env_data(dsrl_next_obs, i)
                for key in self.reward_relabeling_keys:
                    video_frames_all_envs[i][key].append(next_obs_i[key])
                # Collect embeddings if they exist
                if "dino_embedding" in next_obs_i:
                    dino_embeddings_all_envs[i].append(next_obs_i["dino_embedding"])
                if "language" in next_obs_i and text_embeddings_all_envs[i] is None:
                    text_embeddings_all_envs[i] = next_obs_i["language"]

                action_i = self.process_action(dsrl_actions[i])
                reward_i = float(rewards[i])
                done_i = bool(dones[i])
                truncated_i = bool(truncateds[i])

                # Get info safely
                info_i = {}
                if infos is not None:
                    if isinstance(infos, list) and i < len(infos):
                        info_i = infos[i] if infos[i] is not None else {}
                    elif isinstance(infos, dict):
                        info_i = infos  # Single info dict for all envs

                # Check for disconnection - don't add disconnection transitions to buffer
                is_disconnected = info_i.get("disconnected", False)

                if not is_disconnected:
                    # Extract success info (if episode is done/truncated)
                    success_info = {}
                    if done_i or truncated_i:
                        # Pass success indicators from info to buffer
                        if "is_success" in info_i:
                            success_info["is_success"] = info_i["is_success"]
                        elif "success" in info_i:
                            success_info["success"] = info_i["success"]

                    # Add to buffer (skip if disconnected to avoid corrupted data)
                    language_instruction = self.env.get_language_instruction()
                    # Extract step_in_episode from info if available (from async reward relabel wrapper)
                    step_in_episode = info_i.get("step_in_episode", None)
                    if step_in_episode is not None:
                        # Convert to Python int if it's a numpy array/scalar (numpy types aren't hashable)
                        if hasattr(step_in_episode, "item"):
                            step_in_episode = int(step_in_episode.item())
                        else:
                            step_in_episode = int(step_in_episode)
                    self.buffer.add(
                        obs=obs_i,
                        action=action_i,
                        reward=reward_i,
                        next_obs=next_obs_i,
                        done=done_i,
                        truncated=truncated_i,
                        episode_id=self.total_episodes,
                        step_in_episode=step_in_episode,
                        video_frames=video_frames_all_envs[i],
                        language_instruction=language_instruction,
                        dino_embeddings=dino_embeddings_all_envs[i],
                        text_embedding=text_embeddings_all_envs[i],
                        **success_info,
                    )

                    # Update episode tracking
                    self.episode_tracker.add_step(i, reward_i, info_i, done_i, truncated_i, actual_env_steps)
                    # Increment by actual number of environment steps executed
                    num_steps += actual_env_steps

                if done_i or truncated_i:
                    # Log episode end details
                    if is_disconnected:
                        logger.warning(
                            f"Episode {self.total_episodes} interrupted: Robot disconnected. Will retry on next reset."
                        )
                    else:
                        success = info_i.get("is_success", False) or info_i.get("success", False)
                        logger.info(
                            f"Episode {self.total_episodes} ended: done={done_i}, truncated={truncated_i}, success={success}, steps={num_steps}"
                        )

                    # DEBUG: Print success/failure buffer stats
                    from robometer_policy_learning.buffers.success_failure_replay_buffer import SuccessFailureReplayBuffer

                    if isinstance(self.buffer, SuccessFailureReplayBuffer):
                        stats = self.buffer.get_buffer_sizes()
                        print(
                            f"[DEBUG] Buffer Stats - Success: {stats['success_buffer_size']}, Failure: {stats['failure_buffer_size']}, Pending: {stats['pending_episodes']}"
                        )

                    self.episode_tracker.end_episode(i)
                    self.reset_actor_state_for_env(i)
                    video_frames_all_envs[i] = {key: [] for key in self.reward_relabeling_keys}
                    dino_embeddings_all_envs[i] = []
                    text_embeddings_all_envs[i] = None
                    # Environment is already reset in env_step_pi0, recent_obs contains reset obs
                    num_episodes += 1
                    self.total_episodes += 1

        return self.episode_tracker.get_metrics()
