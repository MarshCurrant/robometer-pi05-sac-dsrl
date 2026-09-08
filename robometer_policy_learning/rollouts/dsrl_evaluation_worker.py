import numpy as np
import cv2
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from robometer_policy_learning.modules.base.modeling_actor import BaseActor
from robometer_policy_learning.utils.gpu_utils import move_to_device, convert_to_tensor
from robometer_policy_learning.loggers.logger import Logger
from loguru import logger
from robometer_policy_learning.rollouts.evaluation_worker import EvaluationWorker
from robometer_policy_learning.utils.pi0_integration import Pi0Wrapper
from robometer_policy_learning.utils.dsrl_utils import resize_images
from robometer.evals.eval_viz_utils import create_combined_progress_success_plot
from robometer_policy_learning.envs.async_reward_relabel_wrapper import AsyncRewardRelabelEnvWrapper


class DSRLEvaluationWorker(EvaluationWorker):
    """
    DSRL Evaluation Worker with Pi0 wrapper.
    Note: Inherits the eval mode handling from EvaluationWorker.run()
    
    Args:
        eval_env: Evaluation environment
        device: PyTorch device
        pi0_wrapper: Pi0Wrapper instance for action generation
        action_exec_len: Number of actions to execute from each chunk
        gamma: Discount factor
        num_episodes: Number of episodes to evaluate
        record_video: Whether to record evaluation videos
        logger: Optional logger for metrics
        image_keys: Keys for image observations in video recording
        use_random_noise: If True, sample random Gaussian noise instead of using actor
        noise_dim: Dimension of noise (required if use_random_noise=True)
        noise_scale: Scale of random noise (default 1.0)
    """

    def __init__(
        self,
        eval_env,
        device,
        pi0_wrapper: Pi0Wrapper,
        action_exec_len: int,
        gamma: float = 0.99,
        num_episodes: int = 10,
        record_video: bool = True,
        logger: Logger = None,
        image_keys: List[str] = None,
        use_random_noise: bool = False,
        noise_dim: int = None,
        noise_scale: float = 1.0,
        results_path: str | None = None,
    ):
        super().__init__(eval_env, device, num_episodes, record_video, logger, image_keys)
        self.pi0 = pi0_wrapper
        self.action_exec_len = action_exec_len
        self.gamma = gamma
        self.use_random_noise = use_random_noise
        self.noise_dim = noise_dim
        self.noise_scale = noise_scale
        self.results_path = results_path
        
        # Validate noise_dim is provided if using random noise
        if self.use_random_noise and self.noise_dim is None:
            raise ValueError("noise_dim must be provided when use_random_noise=True")
        
        self.dsrl_key_mapping = None
        if hasattr(eval_env, "envs"):
            self.dsrl_key_mapping = eval_env.envs[0].dsrl_key_mapping
        else:
            self.dsrl_key_mapping = eval_env.dsrl_key_mapping
        if self.dsrl_key_mapping is None:
            raise ValueError("DSRL key mapping not found in environment")

        # Track server capabilities (set on first reset)
        self._server_supports_chunking = False
        
        # Helper to find AsyncRewardRelabelEnvWrapper in env stack (for extracting relabeled data at episode end)
        self._async_wrapper = self._find_async_wrapper(eval_env)
    
    def _find_async_wrapper(self, env):
        """Recursively find AsyncRewardRelabelEnvWrapper in environment stack."""
        if isinstance(env, AsyncRewardRelabelEnvWrapper):
            return env
        
        # For vectorized envs, check sub-envs
        if hasattr(env, "envs"):
            for sub_env in env.envs:
                wrapper = self._find_async_wrapper(sub_env)
                if wrapper is not None:
                    return wrapper
        
        # Check .env attribute for wrapped environments
        if hasattr(env, "env"):
            return self._find_async_wrapper(env.env)
        
        return None

    def _sample_random_noise(self, length, batch_size: int = 1) -> np.ndarray:
        """
        Sample random Gaussian noise for Pi0 steering.
        
        Args:
            batch_size: Number of noise samples to generate
            
        Returns:
            noise: np.ndarray of shape (batch_size, noise_dim)
        """
        noise = np.random.randn(batch_size, length, self.noise_dim) * self.noise_scale
        return noise.astype(np.float32)

    @staticmethod
    def _environment_success(info: Dict) -> bool:
        """Read simulator success without conflating it with reward-model termination."""
        return bool(
            info.get("actual_env_success", False)
            or info.get("is_success", False)
            or info.get("success", False)
        )

    def run(self, actor: BaseActor = None):
        """
        Run evaluation episodes and optionally record video.
        
        Args:
            actor: Policy actor (optional if use_random_noise=True)
        """
        if actor is None and not self.use_random_noise:
            raise ValueError("actor must be provided when use_random_noise=False")
        
        # Set actor to eval mode if provided
        was_training = False
        if actor is not None:
            was_training = actor.training
            actor.eval()

        try:
            # Run multiple evaluation episodes for statistics
            with torch.inference_mode():
                eval_metrics = self._run_evaluations(actor, num_episodes=self.num_episodes)

            # Record a video if requested
            if self.record_video:
                with torch.inference_mode():
                    video_metrics = self._record_evaluation_video(actor)
                eval_metrics.update(video_metrics)

            return eval_metrics
        finally:
            # Restore original training mode
            if actor is not None and was_training:
                actor.train()

    def _process_obs_for_dsrl(self, obs: Dict, squeeze_vlm: bool = True) -> Dict:
        """
        Helper to process raw environment observation into DSRL format.
        Extracts VLM features, concatenates with proprio, and formats dict.

        Args:
            obs: Raw observation dict from environment (should be batched)
            squeeze_vlm: Whether to squeeze the VLM features (for single env)
        """
        # Handle list of prompts (vectorized envs)
        if isinstance(obs["prompt"], (list, tuple)):
            obs["prompt"] = obs["prompt"][0]

        # Extract VLM features from Pi0
        vlm_features = self.pi0.get_features(obs).to(self.device)  # (n_envs, 2048)

        # Observations are already batched, so no unsqueeze needed
        proprio = torch.from_numpy(obs[self.dsrl_key_mapping["state"]]).to(self.device).float()
        language = torch.from_numpy(obs[self.dsrl_key_mapping["language"]]).to(self.device).float()

        # Squeeze if needed (for single env evaluation)
        if squeeze_vlm:
            vlm_features = vlm_features.squeeze(0)
            proprio = proprio.squeeze(0)
            dino_embedding = dino_embedding.squeeze(0)
            language = language.squeeze(0)

        # Prepare final DSRL observation dict
        dsrl_obs = {
            "vlm_features": vlm_features,
            self.dsrl_key_mapping["state"]: proprio,
            "language": language,
        }
        # use dino features
        if "dino_embedding" in obs:
            dsrl_obs["dino_embedding"] = torch.from_numpy(obs["dino_embedding"]).to(self.device).float()
        else:
            # otherwise using image features
            for img_key in self.dsrl_key_mapping["image"]:
                img = resize_images(obs[img_key], 128)
                dsrl_obs[img_key] = torch.from_numpy(img).to(self.device)

        # for key in obs.keys():
        # if key not in dsrl_obs:
        # dsrl_obs[key] = obs[key]

        return dsrl_obs

    def _get_base_env(self, env_idx: int):
        """Get the base (unwrapped) environment for direct access.

        This is used for chunked action execution to bypass vectorized wrapper.
        """
        if hasattr(self.eval_env, "envs"):
            # Vectorized environment - get specific sub-environment
            env = self.eval_env.envs[env_idx]
        else:
            # Single environment
            env = self.eval_env

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

    def _capture_frame(self, obs: Dict, image_keys: List[str], frames_dict: Dict) -> None:
        """Capture and process a frame from the current observation.

        Args:
            obs: Current observation dict (may be batched)
            image_keys: List of keys containing image data
            frames_dict: Dict to append stacked frames to
        """
        if not isinstance(obs, dict) or not image_keys:
            return

        collected_frames = {}

        for key in image_keys:
            if key in obs:
                frame = obs[key]

                # Convert torch tensor to numpy if needed
                if hasattr(frame, "detach"):
                    frame = frame.detach().cpu().numpy()

                if isinstance(frame, np.ndarray):
                    # Convert to uint8
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

                    # Handle shape formats
                    if len(frame.shape) == 4:  # Remove batch dimension
                        frame = frame[0]
                    elif len(frame.shape) == 3 and frame.shape[0] in [1, 3]:  # CHW to HWC
                        frame = np.transpose(frame, (1, 2, 0))

                    # Convert grayscale to RGB
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                    collected_frames[key] = frame

        # Stack frames if any were collected
        if collected_frames:
            stacked_frame = self._stack_frames(collected_frames, image_keys)
            if stacked_frame is not None:
                frames_dict["stacked"].append(stacked_frame)

    def _run_evaluations(self, actor: BaseActor, num_episodes: int = 10):
        """Run multiple evaluation episodes without video recording for statistics."""
        if num_episodes <= 0:
            return {}
        all_rewards = []
        all_steps = []
        all_success = []
        episode_results = []
        all_final_progress = []  # Final progress value per episode
        all_final_success_prob = []  # Final success probability per episode

        for episode_idx in tqdm(range(num_episodes), desc="Evaluating episodes"):
            sum_reward = 0
            is_done = False
            obs, info = self.eval_env.reset()
            reset_info = self._extract_info(info, 0)

            # Check if server advertises chunking support (on first reset)
            if episode_idx == 0:
                self._server_supports_chunking = bool(reset_info.get("supports_action_chunking", False))
                if self._server_supports_chunking:
                    logger.info("[DSRLEvaluationWorker] Server supports action chunking - will use optimized execution")
                else:
                    logger.info(
                        "[DSRLEvaluationWorker] Server does not support chunking - will execute actions one at a time"
                    )

            step_count = 0

            # Track if success was achieved at any point during the episode
            is_success = False
            
            # Track success labels during episode (for plotting)
            success_labels = []

            while not is_done:
                with torch.inference_mode():
                    # Process observation using helper (keep obs batched for Pi0)
                    dsrl_obs = self._process_obs_for_dsrl(obs, squeeze_vlm=False)
                    # Generate noise: either from actor or random sampling
                    if self.use_random_noise:
                        dsrl_actions = self._sample_random_noise(batch_size=1, length=self.pi0.action_horizon)
                        dsrl_actions = None
                    else:
                        noise_a, _ = actor.act(dsrl_obs, deterministic=True)
                        dsrl_actions = noise_a.detach().cpu().numpy()

                    # Query Pi0 with noise to get actions (Pi0 expects batched observations)
                    result = self.pi0.infer(
                        observations=obs,
                        noise=dsrl_actions,
                    )

                    # Extract actions and add to queues
                    pi0_actions = result["actions"]  # (pi0 horizon, 7)

                    # Take first action_exec_len actions
                    action_chunk = pi0_actions[:self.action_exec_len]  # (action_exec_len, 7)

                    if self._server_supports_chunking:
                        # Send entire chunk in one step call
                        # Access base environment directly to avoid vectorized wrapper issues
                        base_env = self._get_base_env(0)
                        if base_env is not None and hasattr(base_env, "step"):
                            # Call base env directly with chunk
                            next_obs_single, reward_single, done_single, truncated_single, info_single = base_env.step(
                                action_chunk
                            )
                            # Wrap back into vectorized format
                            next_obs = self._vectorize_obs(next_obs_single)
                            rewards = np.array([reward_single])
                            dones = np.array([done_single])
                            truncateds = np.array([truncated_single])
                            infos = [info_single]
                        else:
                            # Fallback: try batched format
                            action_chunk_batched = np.expand_dims(action_chunk, axis=0)  # (1, action_exec_len, 7)
                            next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action_chunk_batched)
                    else:
                        # Execute actions one at a time
                        for action in action_chunk:
                            action = np.expand_dims(action, axis=0)  # (1, 7)
                            next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action)
                            if dones.squeeze() or truncateds.squeeze():
                                break

                # Extract data for first environment (index 0)
                reward = self._extract_scalar(rewards, 0)
                done = self._extract_scalar(dones, 0)
                truncated = self._extract_scalar(truncateds, 0)
                obs = next_obs  # Keep batched format for consistency with rollout worker
                info = self._extract_info(infos, 0)

                # Check for success at every step - once success is achieved, it stays true
                if isinstance(info, dict):
                    if self._environment_success(info):
                        is_success = True
                    
                    # Track success labels for plotting (only for first 5 episodes)
                    if episode_idx < 5:
                        success_labels.append(1.0 if self._environment_success(info) else 0.0)

                sum_reward += reward
                is_done = done or truncated
                step_count += 1

                if step_count > 1000:
                    break

            all_rewards.append(sum_reward)
            all_steps.append(step_count)
            all_success.append(is_success)
            episode_results.append(
                {
                    "episode_idx": episode_idx,
                    "seed": int(reset_info.get("robotwin_seed", episode_idx)),
                    "success": bool(is_success),
                    "policy_steps": int(step_count),
                    "environment_steps": int(info.get("robotwin_step_count", step_count * self.action_exec_len)),
                    "return": float(sum_reward),
                }
            )
            
            # Extract relabeled rewards and success probs after episode ends
            # For plotting: only first 5 episodes
            # For statistics: all episodes
            if self._async_wrapper is not None:
                with self._async_wrapper._lock:
                    # Verify sync_mode is enabled for evaluation
                    if not self._async_wrapper.sync_mode:
                        logger.warning(
                            f"Episode {episode_idx + 1}: AsyncRewardRelabelEnvWrapper is not in sync_mode. "
                            f"Relabeled rewards may not be available immediately."
                        )
                    
                    # Extract all relabeled rewards and success probs from wrapper's internal state
                    # Use the actual keys from the dictionaries (step_in_episode values) rather than assuming sequential
                    progress_pred = []
                    success_probs = []
                    env_rewards = []
                    
                    # Get all step indices that have relabeled rewards (sorted)
                    relabeled_step_indices = sorted(self._async_wrapper._relabeled_rewards.keys())
                    
                    if relabeled_step_indices:
                        # Iterate through all steps that have relabeled rewards
                        for step_idx in relabeled_step_indices:
                            # Get relabeled reward (progress)
                            if step_idx in self._async_wrapper._relabeled_rewards:
                                progress_pred.append(float(self._async_wrapper._relabeled_rewards[step_idx]))
                            else:
                                # This shouldn't happen since we're iterating over keys, but log if it does
                                logger.warning(
                                    f"Episode {episode_idx + 1}, step {step_idx}: relabeled_reward key mismatch. "
                                    f"This may indicate a problem."
                                )
                                progress_pred.append(0.0)
                            
                            # Get success probability
                            if step_idx in self._async_wrapper._success_probs:
                                success_probs.append(float(self._async_wrapper._success_probs[step_idx]))
                            else:
                                # In sync mode, all success probs should be available - log warning if missing
                                logger.warning(
                                    f"Episode {episode_idx + 1}, step {step_idx}: success_prob not found in sync mode. "
                                    f"This may indicate a problem."
                                )
                                success_probs.append(0.0)
                            
                            # Get env reward (only needed for plotting)
                            if episode_idx < 5:
                                if step_idx in self._async_wrapper._env_rewards:
                                    env_rewards.append(float(self._async_wrapper._env_rewards[step_idx]))
                                else:
                                    env_rewards.append(0.0)
                    else:
                        # No relabeled rewards found - log warning in sync mode
                        if self._async_wrapper.sync_mode:
                            logger.warning(
                                f"Episode {episode_idx + 1}: No relabeled rewards found in sync mode. "
                                f"This may indicate that relabeling did not complete or episode had no steps."
                            )
                    
                    # Collect statistics for all episodes
                    if progress_pred:
                        progress_array = np.array(progress_pred)
                        all_final_progress.append(float(progress_array[-1]) if len(progress_array) > 0 else 0.0)
                    
                    if success_probs:
                        success_probs_array = np.array(success_probs)
                        all_final_success_prob.append(float(success_probs_array[-1]) if len(success_probs_array) > 0 else 0.0)
                    
                    # Pad success_labels to match length if needed (only for plotting)
                    if episode_idx < 5:
                        while len(success_labels) < len(progress_pred):
                            success_labels.append(0.0)
                    
                    # Create and log plot if we have predictions (only for first 5 episodes)
                    if episode_idx < 5 and progress_pred:
                        progress_array = np.array(progress_pred)
                        success_probs_array = np.array(success_probs) if success_probs else None
                        success_binary = (success_probs_array > 0.5).astype(float) if success_probs_array is not None else None
                        success_labels_array = np.array(success_labels[:len(progress_pred)]) if success_labels else None
                        
                        fig = create_combined_progress_success_plot(
                            progress_pred=progress_array,
                            num_frames=len(progress_array),
                            success_binary=success_binary,
                            success_probs=success_probs_array,
                            success_labels=success_labels_array,
                            title=f"Episode {episode_idx + 1} - Progress & Success",
                        )
                        
                        # Add env_rewards and combined reward as extra lines on the progress plot (first subplot)
                        if env_rewards and len(env_rewards) == len(progress_array):
                            # Get the first axis (progress plot) - it's always the first subplot
                            progress_ax = fig.axes[0]  # or fig.get_axes()[0]
                            
                            # Create a secondary y-axis for env_rewards and combined reward (since they may have different scale)
                            ax2_env = progress_ax.twinx()
                            env_rewards_array = np.array(env_rewards)
                            
                            # Plot env reward
                            ax2_env.plot(env_rewards_array, linewidth=2, color="orange", linestyle="--", label="Env Reward", alpha=0.7)
                            
                            # Compute and plot combined reward (env_reward + relabeled_reward/progress)
                            combined_rewards_array = env_rewards_array + progress_array
                            ax2_env.plot(combined_rewards_array, linewidth=2, color="red", linestyle="-", label="Combined Reward", alpha=0.7)
                            
                            ax2_env.set_ylabel("Reward", color="orange")
                            ax2_env.tick_params(axis="y", labelcolor="orange")
                            
                            # Combine legends from both axes
                            lines1, labels1 = progress_ax.get_legend_handles_labels()
                            lines2, labels2 = ax2_env.get_legend_handles_labels()
                            progress_ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
                        
                        # Log plot to wandb immediately
                        if self.logger is not None:
                            fig.canvas.draw()
                            # Use buffer_rgba() which returns RGBA, then extract RGB
                            # This replaces deprecated tostring_rgb()
                            buf = fig.canvas.buffer_rgba()
                            img_array = np.asarray(buf)
                            # Extract RGB from RGBA (drop alpha channel)
                            img_array = img_array[:, :, :3]
                            
                            if hasattr(self.logger, "log_dict"):
                                self.logger.log_dict(
                                    {f"progress_success_plot_ep{episode_idx}": wandb.Image(img_array)},
                                    step=episode_idx,
                                    prefix="eval"
                                )
                            
                            plt.close(fig)

        # Compute statistics
        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        avg_steps = np.mean(all_steps)
        success_rate = np.mean(all_success)

        eval_metrics = {
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "min_reward": np.min(all_rewards),
            "max_reward": np.max(all_rewards),
            "avg_steps": avg_steps,
            "success_rate": success_rate,
            "num_eval_episodes": num_episodes,
            "episode_results": episode_results,
        }

        if self.results_path:
            results_path = os.path.abspath(os.path.expanduser(self.results_path))
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w", encoding="utf-8") as handle:
                for row in episode_results:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            logger.info(f"Saved per-episode evaluation results to {results_path}")

        # Add progress statistics if available
        if all_final_progress:
            eval_metrics["progress_final_mean"] = np.mean(all_final_progress)
            eval_metrics["progress_final_std"] = np.std(all_final_progress)
            eval_metrics["progress_final_min"] = np.min(all_final_progress)
            eval_metrics["progress_final_max"] = np.max(all_final_progress)

        # Add success probability statistics if available
        if all_final_success_prob:
            eval_metrics["success_prob_final_mean"] = np.mean(all_final_success_prob)
            eval_metrics["success_prob_final_std"] = np.std(all_final_success_prob)
            eval_metrics["success_prob_final_min"] = np.min(all_final_success_prob)
            eval_metrics["success_prob_final_max"] = np.max(all_final_success_prob)

        logger.info(f"Evaluation over {num_episodes} episodes:")
        logger.info(f"  Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
        logger.info(f"  Success Rate: {success_rate:.1%}")
        logger.info(f"  Average Steps: {avg_steps:.1f}")
        
        if all_final_progress:
            logger.info(f"  Final Progress: {eval_metrics['progress_final_mean']:.3f} ± {eval_metrics['progress_final_std']:.3f}")
            logger.info(f"  Progress Range: [{eval_metrics['progress_final_min']:.3f}, {eval_metrics['progress_final_max']:.3f}]")
        
        if all_final_success_prob:
            logger.info(f"  Final Success Prob: {eval_metrics['success_prob_final_mean']:.3f} ± {eval_metrics['success_prob_final_std']:.3f}")
            logger.info(f"  Success Prob Range: [{eval_metrics['success_prob_final_min']:.3f}, {eval_metrics['success_prob_final_max']:.3f}]")

        return eval_metrics

    def _record_evaluation_video(self, actor: BaseActor):
        """Record a single evaluation episode with video for visualization."""

        # Create videos directory (robust to missing logger or log_dir)
        if self.logger is not None and getattr(self.logger, "log_dir", None):
            base_dir = self.logger.log_dir
            video_dir = os.path.join(base_dir, "evaluation_videos")
        else:
            video_dir = os.path.join(os.getcwd(), "evaluation_videos")
        os.makedirs(video_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sum_reward = 0
        is_done = False
        obs, info = self.eval_env.reset()

        # Check if server advertises chunking support
        self._server_supports_chunking = info.get("supports_action_chunking", False)

        # Track if success was achieved at any point during the episode
        is_success = False
        
        # Track success labels during episode (for plotting)
        success_labels = []

        # Identify image keys and initialize storage (works with batched obs)
        if self.image_keys is not None:
            image_keys = self.image_keys
        else:
            image_keys = (
                [key for key in obs.keys() if "image" in key.lower() or "cam" in key.lower()]
                if isinstance(obs, dict)
                else []
            )
        frames_dict = {"stacked": []}
        step_count = 0
        frame_skip_interval = 1

        while not is_done:
            # Capture frames (skip some for faster video)
            if step_count % frame_skip_interval == 0:
                self._capture_frame(obs, image_keys, frames_dict)

            with torch.inference_mode():
                # Generate noise: either from actor or random sampling
                dsrl_obs = self._process_obs_for_dsrl(obs, squeeze_vlm=False)
                if self.use_random_noise:
                    dsrl_actions = self._sample_random_noise(batch_size=1, length=self.pi0.action_horizon)
                    dsrl_actions = None
                else:
                    # Process observation using helper
                    noise_a, _ = actor.act(dsrl_obs, deterministic=True)
                    dsrl_actions = noise_a.detach().cpu().numpy()

                # Query Pi0 with noise to get actions
                result = self.pi0.infer(
                    observations=obs,
                    noise=dsrl_actions,
                )

                # Extract actions and add to queues
                pi0_actions = result["actions"]  # (pi0 horizon, 7)

                # Take first action_exec_len actions
                action_chunk = pi0_actions[: self.action_exec_len]  # (action_exec_len, 7)

                if self._server_supports_chunking:
                    # Send entire chunk in one step call
                    # Access base environment directly to avoid vectorized wrapper issues
                    base_env = self._get_base_env(0)
                    if base_env is not None and hasattr(base_env, "step"):
                        # Call base env directly with chunk
                        next_obs_single, reward_single, done_single, truncated_single, info_single = base_env.step(
                            action_chunk
                        )
                        # Wrap back into vectorized format
                        next_obs = self._vectorize_obs(next_obs_single)
                        rewards = np.array([reward_single])
                        dones = np.array([done_single])
                        truncateds = np.array([truncated_single])
                        infos = [info_single]
                    else:
                        # Fallback: try batched format
                        action_chunk_batched = np.expand_dims(action_chunk, axis=0)  # (1, action_exec_len, 7)
                        next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action_chunk_batched)
                else:
                    # Execute actions one at a time
                    for action in action_chunk:
                        action = np.expand_dims(action, axis=0)  # (1, 7)
                        next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action)
                        if dones.squeeze() or truncateds.squeeze():
                            break

            # Extract data for first environment (index 0)
            reward = self._extract_scalar(rewards, 0)
            done = self._extract_scalar(dones, 0)
            truncated = self._extract_scalar(truncateds, 0)
            obs = next_obs  # Keep batched format for consistency with rollout worker
            info = self._extract_info(infos, 0)

            # Check for success at every step - once success is achieved, it stays true
            if isinstance(info, dict):
                if info.get("is_success", False) or info.get("success", False):
                    is_success = True
                
                # Track success labels for plotting
                success_labels.append(1.0 if info.get("is_success", False) or info.get("success", False) else 0.0)

            sum_reward += reward
            is_done = done or truncated
            step_count += 1

        # Capture the final frame to ensure last observation is saved
        self._capture_frame(obs, image_keys, frames_dict)
        
        # Extract relabeled rewards and success probs after episode ends
        # In sync mode, these should be available after episode completion
        video_progress_plot = None
        if self._async_wrapper is not None:
            with self._async_wrapper._lock:
                # Verify sync_mode is enabled for evaluation
                if not self._async_wrapper.sync_mode:
                    logger.warning(
                        "Video episode: AsyncRewardRelabelEnvWrapper is not in sync_mode. "
                        "Relabeled rewards may not be available immediately."
                    )
                
                # Extract all relabeled rewards and success probs from wrapper's internal state
                # Use the actual keys from the dictionaries (step_in_episode values) rather than assuming sequential
                progress_pred = []
                success_probs = []
                env_rewards = []
                
                # Get all step indices that have relabeled rewards (sorted)
                relabeled_step_indices = sorted(self._async_wrapper._relabeled_rewards.keys())
                
                if relabeled_step_indices:
                    # Iterate through all steps that have relabeled rewards
                    for step_idx in relabeled_step_indices:
                        # Get relabeled reward (progress)
                        if step_idx in self._async_wrapper._relabeled_rewards:
                            progress_pred.append(float(self._async_wrapper._relabeled_rewards[step_idx]))
                        else:
                            # This shouldn't happen since we're iterating over keys, but log if it does
                            logger.warning(
                                f"Video episode, step {step_idx}: relabeled_reward key mismatch. "
                                f"This may indicate a problem."
                            )
                            progress_pred.append(0.0)
                        
                        # Get success probability
                        if step_idx in self._async_wrapper._success_probs:
                            success_probs.append(float(self._async_wrapper._success_probs[step_idx]))
                        else:
                            # In sync mode, all success probs should be available - log warning if missing
                            logger.warning(
                                f"Video episode, step {step_idx}: success_prob not found in sync mode. "
                                f"This may indicate a problem."
                            )
                            success_probs.append(0.0)
                        
                        # Get env reward
                        if step_idx in self._async_wrapper._env_rewards:
                            env_rewards.append(float(self._async_wrapper._env_rewards[step_idx]))
                        else:
                            env_rewards.append(0.0)
                else:
                    # No relabeled rewards found - log warning in sync mode
                    if self._async_wrapper.sync_mode:
                        logger.warning(
                            "Video episode: No relabeled rewards found in sync mode. "
                            "This may indicate that relabeling did not complete or episode had no steps."
                        )
                
                # Pad success_labels to match length if needed
                while len(success_labels) < len(progress_pred):
                    success_labels.append(0.0)
                
                # Create plot if we have predictions
                if progress_pred:
                    progress_array = np.array(progress_pred)
                    success_probs_array = np.array(success_probs) if success_probs else None
                    success_binary = (success_probs_array > 0.5).astype(float) if success_probs_array is not None else None
                    success_labels_array = np.array(success_labels[:len(progress_pred)]) if success_labels else None
                    
                    video_progress_plot = create_combined_progress_success_plot(
                        progress_pred=progress_array,
                        num_frames=len(progress_array),
                        success_binary=success_binary,
                        success_probs=success_probs_array,
                        success_labels=success_labels_array,
                        title=f"Video Episode - Progress & Success",
                    )
                    
                    # Add env_rewards and combined reward as extra lines on the progress plot (first subplot)
                    if env_rewards and len(env_rewards) == len(progress_array):
                        # Get the first axis (progress plot) - it's always the first subplot
                        progress_ax = video_progress_plot.axes[0]
                        
                        # Create a secondary y-axis for env_rewards and combined reward (since they may have different scale)
                        ax2_env = progress_ax.twinx()
                        env_rewards_array = np.array(env_rewards)
                        
                        # Plot env reward
                        ax2_env.plot(env_rewards_array, linewidth=2, color="orange", linestyle="--", label="Env Reward", alpha=0.7)
                        
                        # Compute and plot combined reward (env_reward + relabeled_reward/progress)
                        combined_rewards_array = env_rewards_array + progress_array
                        ax2_env.plot(combined_rewards_array, linewidth=2, color="red", linestyle="-", label="Combined Reward", alpha=0.7)
                        
                        ax2_env.set_ylabel("Reward", color="orange")
                        ax2_env.tick_params(axis="y", labelcolor="orange")
                        
                        # Combine legends from both axes
                        lines1, labels1 = progress_ax.get_legend_handles_labels()
                        lines2, labels2 = ax2_env.get_legend_handles_labels()
                        progress_ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # Save video and log to TensorBoard
        video_saved = False
        if frames_dict["stacked"]:
            video_path = os.path.join(video_dir, f"eval_{timestamp}_all_cameras.mp4")

            # Log to TensorBoard if logger available
            if self.logger is not None:
                video_frames_rgb = []
                for frame_bgr in frames_dict["stacked"]:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_chw = np.transpose(frame_rgb, (2, 0, 1))
                    video_frames_rgb.append(frame_chw)

                video_tensor = torch.from_numpy(np.stack(video_frames_rgb)).unsqueeze(0)
                self.logger.log_video("eval_video", video=video_tensor, step=step_count, prefix="eval")

            # Save MP4 file
            self._save_video(frames_dict["stacked"], video_path, fps=20)
            video_saved = True
            logger.success(f"📹 Saved evaluation video: {video_path}")
        
        # Save and log progress/success plot for video episode
        if video_progress_plot is not None:
            # Save plot to file
            plot_path = os.path.join(video_dir, f"eval_{timestamp}_progress_success.png")
            video_progress_plot.savefig(plot_path, dpi=150, bbox_inches="tight")
            logger.success(f"📊 Saved progress/success plot: {plot_path}")
            
            # Log to wandb if logger is available
            if self.logger is not None:
                video_progress_plot.canvas.draw()
                # Use buffer_rgba() which returns RGBA, then extract RGB
                # This replaces deprecated tostring_rgb()
                buf = video_progress_plot.canvas.buffer_rgba()
                img_array = np.asarray(buf)
                # Extract RGB from RGBA (drop alpha channel)
                img_array = img_array[:, :, :3]
                
                if hasattr(self.logger, "log_dict"):
                    self.logger.log_dict(
                        {"video_progress_success_plot": wandb.Image(img_array)},
                        step=step_count,
                        prefix="eval"
                    )
            
            plt.close(video_progress_plot)

        return {
            "video_reward": sum_reward,
            "video_steps": step_count,
            "video_saved": video_saved,
            "video_success": is_success,
        }
