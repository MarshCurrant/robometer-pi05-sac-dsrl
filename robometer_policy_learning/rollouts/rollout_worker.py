import torch
import numpy as np
import gymnasium as gym
from typing import Dict, List, Any, Optional, Tuple
from robometer_policy_learning.modules.base.modeling_actor import BaseActor
from robometer_policy_learning.rollouts.episode_data import EpisodeData
from robometer_policy_learning.utils.gpu_utils import move_to_device, convert_to_tensor


class RolloutWorker:
    """
    Clean, simple rollout worker for collecting environment data.

    Supports:
    - Step-based or episode-based collection
    - Stateful policies (RNNs, etc.)
    - Multi-environment parallelization
    """

    def __init__(
        self,
        env: gym.Env,
        buffer,
        actor: BaseActor = None,
        num_rollouts: int = 1,
        num_envs: int = 1,
        device: torch.device = torch.device("cpu"),
        count_by: str = "episode",  # 'step' or 'episode'
        **kwargs,
    ):
        self.env = env
        self.buffer = buffer
        self.actor = actor
        self.num_rollouts = num_rollouts
        self.num_envs = num_envs
        self.device = device
        self.count_by = count_by
        self.kwargs = kwargs

        # State tracking
        self.recent_obs = None
        self.actor_state = None

        # Episode metrics
        self.episode_tracker = EpisodeTracker(num_envs)
        self.episode_tracker.reset()

        # Check if this is a chunked rollout
        self.is_chunked_rollout = hasattr(self.env, "is_chunk_empty")

        self.total_episodes = 0

    def run(self, can_train=True) -> Dict[str, float]:
        """Main rollout loop."""
        # Reset for new rollout
        num_steps = 0
        num_episodes = 0

        actor_was_training = self.actor.training
        while self.should_continue(num_steps, num_episodes):
            # Get observations
            if self.recent_obs is None:
                obs, _ = self.env.reset()
                self.actor_state = self.get_initial_actor_state()
            else:
                obs = self.recent_obs

            obs_tensor = convert_to_tensor(obs)
            obs_tensor = move_to_device(obs_tensor, self.device)

            # If chunked, we need compute actions when the chunk is empty
            if self.is_chunked_rollout:
                if self.env.is_chunk_empty:
                    with torch.inference_mode():
                        if actor_was_training:
                            self.actor.eval()
                        actions, self.actor_state = self.actor.act(
                            obs_tensor, actor_state=self.actor_state, deterministic=False
                        )
                        if actor_was_training:
                            self.actor.train()
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
                    **success_info,
                )

                # Update episode tracking
                self.episode_tracker.add_step(i, reward_i, info_i, done_i, truncated_i)
                num_steps += 1
                
                if done_i or truncated_i:
                    self.episode_tracker.end_episode(i)
                    self.reset_actor_state_for_env(i)
                    num_episodes += 1
                    self.total_episodes += 1
            self.recent_obs = next_obs

        return self.episode_tracker.get_metrics()

    def update_actor(self, actor: BaseActor):
        """Update actor and reset its state."""
        self.actor = actor
        self.actor_state = self.get_initial_actor_state()

    def should_continue(self, num_steps: int, num_episodes: int) -> bool:
        """Check if we should continue collecting data."""
        if self.num_rollouts == -1:
            return True

        if self.count_by == "step":
            return num_steps < self.num_rollouts
        else:  # episode
            return num_episodes < self.num_rollouts

    def should_collect_from_env(self, env_idx: int) -> bool:
        """Check if we should collect data from this environment."""
        if self.count_by == "step" or self.num_rollouts == -1:
            return True
        return self.episode_tracker.episodes_collected[env_idx] < self.num_rollouts

    def extract_env_data(self, batched_data, env_idx: int):
        """Extract data for a specific environment from batched format."""
        if isinstance(batched_data, dict):
            return {key: value[env_idx] for key, value in batched_data.items()}
        else:
            return batched_data[env_idx]

    def process_action(self, action):
        """Process action with normalization if needed."""
        if not self.actor.normalize_actions:
            return action if isinstance(action, np.ndarray) else action.numpy()

        action_tensor = torch.tensor(action, dtype=torch.float32) if not torch.is_tensor(action) else action
        normalized_action = self.actor.normalize_action(action_tensor)
        return normalized_action.numpy()

    def get_initial_actor_state(self):
        """Get initial actor state if the actor supports it."""
        if self.actor is None:
            raise ValueError("Actor is not set")

        elif hasattr(self.actor, "get_initial_state"):
            return [self.actor.get_initial_state() for _ in range(self.num_envs)]
        else:
            return None

    def reset_actor_state_for_env(self, env_idx: int):
        """Reset actor state for a specific environment."""
        if self.actor_state is None or not hasattr(self.actor, "get_initial_state"):
            return

        if isinstance(self.actor_state, list):
            self.actor_state[env_idx] = self.actor.get_initial_state()


class EpisodeTracker:
    """Episode tracking and metrics calculation."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.reset()
        self.total_steps = 0
        self.episodes_collected = [0] * self.num_envs

    def reset(self):
        """Reset all tracking state."""
        self.current_rewards = [0.0] * self.num_envs
        self.current_successes = [False] * self.num_envs
        self.completed_rewards = []
        self.completed_successes = []
        
        # Track env_reward, progress_reward, and success_prob
        self.env_rewards = []
        self.progress_rewards = []
        self.success_probs = []
        self.current_training_rewards = [0.0] * self.num_envs
        self.completed_training_rewards = []

    def add_step(
        self, env_idx: int, reward: float, info: Dict, done: bool, truncated: bool, actual_env_steps: int = None
    ):
        """Add a step for the given environment. If actual_env_steps is provided, increment total_steps by that amount."""
        self.total_steps += actual_env_steps if actual_env_steps is not None else 1
        self.current_rewards[env_idx] += reward

        # Check for success indicators (check both done and truncated for real robot compatibility)
        if (done or truncated) and self.is_success(info):
            self.current_successes[env_idx] = True

        # Track env_reward, relabeled_reward, and success_prob if available
        if info is not None and "env_reward" in info:
            if isinstance(info["env_reward"], list) or isinstance(info["env_reward"], np.ndarray):
                env_reward = info["env_reward"][env_idx]
                relabeled_reward = info["relabeled_reward"][env_idx]
                success_prob = info["success_prob"][env_idx]
            else:
                env_reward = info["env_reward"]
                relabeled_reward = info["relabeled_reward"]
                success_prob = info["success_prob"]

            if env_reward is not None:
                self.env_rewards.append(float(env_reward))
            if relabeled_reward is not None:
                self.progress_rewards.append(float(relabeled_reward))
            if success_prob is not None:
                self.success_probs.append(float(success_prob))
        if info is not None and info.get("training_reward") is not None:
            self.current_training_rewards[env_idx] += float(info["training_reward"])

    def end_episode(self, env_idx: int):
        """End an episode for the given environment."""
        self.completed_rewards.append(self.current_rewards[env_idx])
        self.completed_successes.append(self.current_successes[env_idx])
        self.completed_training_rewards.append(self.current_training_rewards[env_idx])

        # Reset for next episode
        self.current_rewards[env_idx] = 0.0
        self.current_successes[env_idx] = False
        self.current_training_rewards[env_idx] = 0.0
        self.episodes_collected[env_idx] += 1

    def min_episodes_collected(self) -> int:
        """Get the minimum number of episodes collected across all environments."""
        return min(self.episodes_collected) if self.episodes_collected else 0

    def get_metrics(self) -> Dict[str, float]:
        """Get episode metrics."""
        if not self.completed_rewards:
            return {
                "avg_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "success_rate": 0.0,
                "num_episodes": 0,
                "total_steps": self.total_steps,
            }

        metrics = {
            "ep_avg_reward": np.mean(self.completed_rewards[-self.num_envs :]),
            "ep_min_reward": np.min(self.completed_rewards[-self.num_envs :]),
            "ep_max_reward": np.max(self.completed_rewards[-self.num_envs :]),
            "success_rate": np.mean(self.completed_successes[-self.num_envs :]),
            "ep_overall_success_rate": np.mean(self.completed_successes),
            "ep_overall_reward": np.mean(self.completed_rewards),
            "num_episodes": len(self.completed_rewards),
            "total_steps": self.total_steps,
        }
        
        # Add env_reward statistics if available
        if self.env_rewards:
            metrics["ep_avg_env_reward"] = np.mean(self.env_rewards)
            metrics["ep_min_env_reward"] = np.min(self.env_rewards)
            metrics["ep_max_env_reward"] = np.max(self.env_rewards)
        
        # Add progress_reward statistics if available
        if self.progress_rewards:
            metrics["ep_avg_progress_reward"] = np.mean(self.progress_rewards)
            metrics["ep_min_progress_reward"] = np.min(self.progress_rewards)
            metrics["ep_max_progress_reward"] = np.max(self.progress_rewards)
        
        # Add success_prob statistics if available
        if self.success_probs:
            metrics["ep_avg_success_prob"] = np.mean(self.success_probs)
            metrics["ep_min_success_prob"] = np.min(self.success_probs)
            metrics["ep_max_success_prob"] = np.max(self.success_probs)
        if self.completed_training_rewards:
            metrics["ep_avg_training_reward"] = np.mean(
                self.completed_training_rewards[-self.num_envs :]
            )
            metrics["ep_overall_training_reward"] = np.mean(
                self.completed_training_rewards
            )
        
        return metrics

    def is_success(self, info: Dict) -> bool:
        """Check if the episode was successful based on info."""
        return info.get("is_success", False) or info.get("success", False)
