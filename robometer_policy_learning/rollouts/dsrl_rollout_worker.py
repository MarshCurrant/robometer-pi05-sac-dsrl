"""
DSRL Rollout Worker

Handles rollout collection for DSRL:
- Manages action queues (Pi0 predicts chunks)
- Queries Pi0 with noise from SAC actor
- Stores trajectories in replay buffer
"""

import numpy as np
import torch
from typing import Dict, Any, Optional
from PIL import Image
import gymnasium as gym
from loguru import logger
from robometer_policy_learning.utils.pi0_integration import Pi0Wrapper
from robometer_policy_learning.utils.dsrl_utils import ActionQueue, format_obs_for_storage, resize_images

from robometer_policy_learning.rollouts.rollout_worker import RolloutWorker
from robometer_policy_learning.modules.base.modeling_actor import BaseActor
from robometer_policy_learning.envs.dsrl_env_wrappers import DummyDSRLEnv


class DSRLRolloutWorker(RolloutWorker):
    def __init__(
        self,
        env: gym.Env,
        buffer,
        pi0_wrapper: Pi0Wrapper,
        action_exec_len: int,
        gamma: float = 0.99,
        actor: BaseActor = None,
        num_rollouts: int = 1,
        num_envs: int = 1,
        device: torch.device = torch.device("cpu"),
        count_by: str = "episode",  # 'step' or 'episode'
        dummy_dsrl_env: DummyDSRLEnv = None,
        **kwargs,
    ):
        super().__init__(env, buffer, actor, num_rollouts, num_envs, device, count_by)
        self.pi0 = pi0_wrapper
        self.gamma = gamma  # not used for now
        self.action_exec_len = action_exec_len
        self.dummy_dsrl_env = dummy_dsrl_env

        # Action queues (one per environment)
        self.action_queues = ActionQueue(num_envs)

        # Track if environment needs reset
        self.needs_reset = [False] * num_envs

        # Cache for VLM features to avoid recomputation
        # dsrl_next_obs from step N becomes dsrl_obs for step N+1
        self._cached_dsrl_obs = None

        # Track server capabilities (set on first reset)
        self._server_supports_chunking = False

        if hasattr(self.env, "envs"):
            assert hasattr(self.env.envs[0], "dsrl_key_mapping"), "env must have dsrl_key_mapping attribute"
            self.dsrl_key_mapping = self.env.envs[0].dsrl_key_mapping
        else:
            assert hasattr(self.env, "dsrl_key_mapping"), "env must have dsrl_key_mapping attribute"
            self.dsrl_key_mapping = self.env.dsrl_key_mapping

    def run(self, can_train=True) -> Dict[str, float]:
        """Main rollout loop."""
        # Reset for new rollout
        num_steps = 0
        num_episodes = 0

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
                next_obs_i = self.extract_env_data(dsrl_next_obs, i)
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
                    # Extract step_in_episode from info if available (from async reward relabel wrapper)
                    step_in_episode = info_i.get("step_in_episode", None)
                    # logger.debug(
                    #     f"[DSRLRolloutWorker] Before processing step_in_episode: raw={step_in_episode} (type={type(step_in_episode).__name__ if step_in_episode is not None else 'None'}), "
                    #     f"actual_env_steps={actual_env_steps}, episode_id={self.total_episodes}"
                    # )
                    if step_in_episode is not None:
                        # Convert to Python int if it's a numpy array/scalar
                        import numpy as np

                        if isinstance(step_in_episode, np.ndarray):
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
                    # Environment is already reset in env_step_pi0, recent_obs contains reset obs
                    num_episodes += 1
                    self.total_episodes += 1

        return self.episode_tracker.get_metrics()

    def _process_obs_for_dsrl(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Helper to process raw environment observation into DSRL format.
        Extracts VLM features, concatenates with proprio, and formats dict.
        """
        # Ensure prompt is present and in correct format
        if "prompt" not in obs:
            raise ValueError(f"Observation missing 'prompt' key. Available keys: {obs.keys()}")

        # Handle list of prompts (vectorized envs)
        if isinstance(obs["prompt"], (list, tuple)):
            obs["prompt"] = obs["prompt"][0]

        proprio = torch.from_numpy(obs[self.dsrl_key_mapping["state"]]).to(self.device).float()

        # Prepare final DSRL observation dict
        dsrl_obs = {
            self.dsrl_key_mapping["state"]: proprio,
            "language": torch.from_numpy(obs[self.dsrl_key_mapping["language"]]).to(self.device).float(),
        }

        # Extract VLM features from Pi0
        vlm_features = None
        if self.dummy_dsrl_env.use_vlm_features:
            with torch.inference_mode():
                # do this BEFORE resizing the image for impala so that pi0 gets full res original image
                vlm_features = self.pi0.get_features(obs).to(self.device)  # (n_envs, 2048)

            # Clone to make it a normal tensor (detached from inference mode graph if any)
            dsrl_obs["vlm_features"] = vlm_features.clone()


        # use dino features
        if "dino_embedding" in obs:
            dsrl_obs["dino_embedding"] = torch.from_numpy(obs["dino_embedding"]).to(self.device).float()
        else:
            # otherwise using image features
            for img_key in self.dsrl_key_mapping["image"]:
                # resize numpy image to lower resolution 128x128
                img = resize_images(obs[img_key], 128)
                dsrl_obs[img_key] = torch.from_numpy(img).to(self.device)

        return dsrl_obs

    def env_step_pi0(self, env_ids, can_train=True):
        # Reset environment if needed (from previous termination)
        if self.recent_obs is None or self.needs_reset[0]:
            obs, info = self.env.reset()
            # Check if server advertises chunking support
            self._server_supports_chunking = info.get("supports_action_chunking", False)
            if self._server_supports_chunking:
                logger.info("[DSRLRolloutWorker] Server supports action chunking - will use optimized execution")
            else:
                logger.info("[DSRLRolloutWorker] Server does not support chunking - will execute actions one at a time")
            self.actor_state = self.get_initial_actor_state()
            # Ensure queue is clear when resetting
            self.action_queues.clear(0)
            self.needs_reset[0] = False
            # Invalidate cache on reset
            self._cached_dsrl_obs = None
        else:
            obs = self.recent_obs

        # Reuse cached dsrl_obs if available (dsrl_next_obs from previous step)
        # This avoids recomputing Pi0 VLM features for the same observation
        if self._cached_dsrl_obs is not None:
            dsrl_obs = self._cached_dsrl_obs
        else:
            dsrl_obs = self._process_obs_for_dsrl(obs)

        # Predict noise with actor
        # noise_a, self.actor_state = self.actor.act(dsrl_obs.copy(), actor_state=self.actor_state, deterministic=False)
        # print(f"train noise: {noise_a.mean(), noise_a.min(), noise_a.max(), noise_a.shape}")

        # dsrl_actions = noise_a.detach().cpu().numpy()
        if can_train:
            # Predict noise with actor
            noise_a, self.actor_state = self.actor.act(
                dsrl_obs.copy(), actor_state=self.actor_state, deterministic=False
            )

            dsrl_actions = noise_a.detach().cpu().numpy()
        else:
            dsrl_actions = self.dummy_dsrl_env.sample_action()

        action_og_ndims = dsrl_actions.ndim

        # Query Pi0 with noise to get actions (use inference_mode)
        with torch.inference_mode():
            result = self.pi0.infer(
                observations=obs,
                noise=dsrl_actions,
            )

        # Extract actions and add to queues
        pi0_actions = result["actions"]  # (n_envs, horizon, 7)
        if pi0_actions.ndim == 2:
            pi0_actions = np.expand_dims(pi0_actions, axis=0)

        # NOTE: the snippet below assumes there is only one environment (env_id = 0)
        env_id = 0
        action_chunk = pi0_actions[env_id, : self.action_exec_len]

        # Execute actions - use chunked execution if server supports it
        self.action_queues.add(env_id, action_chunk)

        if self._server_supports_chunking:
            # Send entire chunk in one STEP call for better latency
            chunk_actions = []
            while not self.action_queues.is_empty(0):
                chunk_actions.append(self.action_queues.pop(0))
            chunk_actions = np.array(chunk_actions)  # Shape: (N, action_dim)

            # Only last observation needed when chunking
            self._set_need_obs(True)

            # Access the base environment directly to send chunked actions
            # The vectorized wrapper expects (num_envs, action_dim), not (chunk_size, action_dim)
            base_env = self._get_base_env(0)
            if base_env is not None and hasattr(base_env, "step"):
                # Call base env directly with chunk
                next_obs_single, reward_single, done_single, truncated_single, info_single = base_env.step(
                    chunk_actions
                )

                # Wrap back into vectorized format
                next_obs = self._vectorize_obs(next_obs_single)
                rewards = np.array([reward_single])
                dones = np.array([done_single])
                truncateds = np.array([truncated_single])
                infos = [info_single]
            else:
                # Fallback: reshape to add batch dimension
                chunk_actions_batched = chunk_actions[None, :]  # (1, N, action_dim)
                next_obs, rewards, dones, truncateds, infos = self.env.step(chunk_actions_batched)

            # Get actual number of steps executed from server response
            if isinstance(infos, list):
                actual_env_steps = infos[0].get("num_steps", len(chunk_actions))
            else:
                actual_env_steps = infos.get("num_steps", len(chunk_actions))

            if dones.squeeze() or truncateds.squeeze():
                # Mark that environment needs reset on next call
                self.needs_reset[0] = True
        else:
            # Original one-at-a-time execution for servers that don't support chunking
            actual_env_steps = 0
            while not self.action_queues.is_empty(0):
                single_action = self.action_queues.pop(0)[None, :]

                # Only request full observation (including camera) on the last step of the chunk
                # This saves expensive camera captures on intermediate steps
                is_last_action = self.action_queues.is_empty(0)

                # Set need_obs attribute on environment before stepping
                # This works through vectorized wrappers that don't pass kwargs
                self._set_need_obs(is_last_action)

                # Sparse reward: only last step's reward is stored (intentional)
                next_obs, rewards, dones, truncateds, infos = self.env.step(single_action)

                actual_env_steps += 1
                if dones.squeeze() or truncateds.squeeze():
                    # Clear remaining actions from queue since episode ended
                    self.action_queues.clear(0)
                    # Mark that environment needs reset on next call
                    self.needs_reset[0] = True

                    break

        # Process next observation using helper
        dsrl_next_obs = self._process_obs_for_dsrl(next_obs)

        # Cache dsrl_next_obs - it becomes dsrl_obs for the next step
        # This avoids redundant Pi0 VLM feature computation
        if not (dones.squeeze() or truncateds.squeeze()):
            self._cached_dsrl_obs = dsrl_next_obs
        else:
            # Invalidate cache if episode ended (next step will use reset obs)
            self._cached_dsrl_obs = None

        # Store next_obs (which is reset obs if episode terminated, else regular next_obs)
        self.recent_obs = next_obs

        return dsrl_obs, dsrl_next_obs, dsrl_actions, rewards, dones, truncateds, infos, actual_env_steps

    def _set_need_obs(self, need_obs: bool):
        """
        Set need_obs flag on the underlying environment(s).

        This works through vectorized wrappers by finding the base RemoteEnv
        and setting its need_obs attribute directly.
        """
        # Try to find and set need_obs on base environments
        # For SyncVectorEnv, access the individual environments
        if hasattr(self.env, "envs"):
            # Vectorized environment - set on each sub-environment
            for sub_env in self.env.envs:
                self._set_env_need_obs(sub_env, need_obs)
        else:
            # Single environment
            self._set_env_need_obs(self.env, need_obs)

    def _set_env_need_obs(self, env, need_obs: bool):
        """Recursively find and set need_obs on the base environment."""
        # Set on this env if it has the attribute
        if hasattr(env, "need_obs"):
            env.need_obs = need_obs

        # Recursively check wrapped environments
        if hasattr(env, "env"):
            self._set_env_need_obs(env.env, need_obs)

    def _get_base_env(self, env_idx: int):
        """Get the base (unwrapped) environment for direct access.

        This is used for chunked action execution to bypass vectorized wrapper.
        """
        if hasattr(self.env, "envs"):
            # Vectorized environment - get specific sub-environment
            env = self.env.envs[env_idx]
        else:
            # Single environment
            env = self.env

        return env

    def _vectorize_obs(self, obs):
        """Convert single observation to vectorized format.

        Used when we call base env directly and need to wrap result for vectorized interface.
        """
        if isinstance(obs, dict):
            vectorized_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    # Add batch dimension
                    vectorized_obs[key] = value[None, ...]
                else:
                    # Keep scalars/strings as-is
                    vectorized_obs[key] = value
            return vectorized_obs
        else:
            # For non-dict obs, just add batch dimension
            return np.array([obs])

    def _pop_actions(self) -> np.ndarray:
        """Pop one action from each queue"""
        actions = []
        for env_id in range(self.num_envs):
            action = self.action_queues.pop(env_id)
            if action is None:
                # Should not happen if logic is correct
                action = np.zeros(7)
            actions.append(action)
        return np.array(actions)

    def force_reset(self):
        """
        Force reset the rollout worker state.

        Call this after evaluation when eval_env is the same as training env,
        to ensure the rollout worker starts fresh with a new episode.
        """
        logger.info("[DSRLRolloutWorker] Forcing reset after evaluation...")

        # Clear action queues
        for env_id in range(self.num_envs):
            self.action_queues.clear(env_id)

        # Mark all environments as needing reset
        self.needs_reset = [True] * self.num_envs

        # Clear recent observation (will be populated on next run)
        self.recent_obs = None

        # Invalidate VLM feature cache
        self._cached_dsrl_obs = None

        # Reset actor state
        self.actor_state = self.get_initial_actor_state()

        logger.info("[DSRLRolloutWorker] State reset complete. Next run will start fresh.")
