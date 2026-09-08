import numpy as np
import cv2
import os
from datetime import datetime
from typing import Dict, List
import torch

from robometer_policy_learning.modules.base.modeling_actor import BaseActor
from robometer_policy_learning.utils.gpu_utils import move_to_device, convert_to_tensor
from robometer_policy_learning.loggers.logger import Logger
from loguru import logger


class EvaluationWorker:
    """Worker class for running evaluations and recording videos."""

    def __init__(
        self,
        eval_env,
        device,
        num_episodes: int = 10,
        record_video: bool = True,
        logger: Logger = None,
        image_keys: List[str] = None,
    ):
        self.eval_env = eval_env
        self.device = device
        self.num_episodes = num_episodes
        self.record_video = record_video
        self.logger = logger
        self.image_keys = image_keys

        # Check if this is a chunked rollout (like in RolloutWorker)
        self.is_chunked_rollout = hasattr(self.eval_env, "is_chunk_empty")

    def run(self, actor: BaseActor):
        """Run evaluation episodes and optionally record video."""
        # Set actor to eval mode to disable dropout/batchnorm randomness
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
            if was_training:
                actor.train()

    def _run_evaluations(self, actor: BaseActor, num_episodes: int = 10):
        """Run multiple evaluation episodes without video recording for statistics."""
        all_rewards = []
        all_steps = []
        all_success = []

        for episode_idx in range(num_episodes):
            sum_reward = 0
            is_done = False
            obs, info = self.eval_env.reset()
            step_count = 0

            # Extract first env data for vectorized environments
            obs = self._extract_env_data(obs, 0)

            # Track if success was achieved at any point during the episode
            is_success = False

            while not is_done:
                # Get action and step environment
                obs_device = convert_to_tensor(obs)
                obs_device = move_to_device(obs_device, self.device)

                # Handle chunked actions (like in RolloutWorker)
                if self.is_chunked_rollout:
                    if self.eval_env.is_chunk_empty:
                        action, _ = actor.act(obs_device, deterministic=True)
                        action = action.detach().cpu().numpy()
                        next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action)
                    else:
                        action = None
                        next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action)
                    # Update action to grab the actual action that was taken
                    action = self.eval_env._get_last_action()
                else:
                    action, _ = actor.act(obs_device, deterministic=True)
                    action = action.detach().cpu().numpy()
                    next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action)

                # Extract data for first environment (index 0)
                reward = self._extract_scalar(rewards, 0)
                done = self._extract_scalar(dones, 0)
                truncated = self._extract_scalar(truncateds, 0)
                obs = self._extract_env_data(next_obs, 0)
                info = self._extract_info(infos, 0)

                # Check for success at every step - once success is achieved, it stays true
                if isinstance(info, dict):
                    if info.get("is_success", False) or info.get("success", False):
                        is_success = True

                sum_reward += reward
                is_done = done or truncated
                step_count += 1

            all_rewards.append(sum_reward)
            all_steps.append(step_count)
            all_success.append(is_success)

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
        }

        logger.info(f"Evaluation over {num_episodes} episodes:")
        logger.info(f"  Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
        logger.info(f"  Success Rate: {success_rate:.1%}")
        logger.info(f"  Average Steps: {avg_steps:.1f}")

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

        # Extract first env data for vectorized environments
        obs = self._extract_env_data(obs, 0)

        # Track if success was achieved at any point during the episode
        is_success = False

        # Identify image keys and initialize storage
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
        frame_skip_interval = 3

        while not is_done:
            # Capture frames (skip some for faster video)
            if step_count % frame_skip_interval == 0 and isinstance(obs, dict) and image_keys:
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
                            elif len(frame.shape) == 3 and frame.shape[0] in [
                                1,
                                3,
                            ]:  # CHW to HWC
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

            # Get action and step environment
            obs_device = convert_to_tensor(obs)
            obs_device = move_to_device(obs_device, self.device)

            # Handle chunked actions (like in RolloutWorker)
            if self.is_chunked_rollout:
                if self.eval_env.is_chunk_empty:
                    action, _ = actor.act(obs_device, deterministic=True)
                    action = action.detach().cpu().numpy()
                    next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action)
                else:
                    action = None
                    next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action)
                # Update action to grab the actual action that was taken
                action = self.eval_env._get_last_action()
            else:
                action, _ = actor.act(obs_device, deterministic=True)
                action = action.detach().cpu().numpy()
                next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action)

            # Extract data for first environment (index 0)
            reward = self._extract_scalar(rewards, 0)
            done = self._extract_scalar(dones, 0)
            truncated = self._extract_scalar(truncateds, 0)
            obs = self._extract_env_data(next_obs, 0)
            info = self._extract_info(infos, 0)

            # Check for success at every step - once success is achieved, it stays true
            if isinstance(info, dict):
                if info.get("is_success", False) or info.get("success", False):
                    is_success = True

            sum_reward += reward
            is_done = done or truncated
            step_count += 1

            if step_count > 1000:
                break

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

        return {
            "video_reward": sum_reward,
            "video_steps": step_count,
            "video_saved": video_saved,
            "video_success": is_success,
        }

    def _stack_frames(self, collected_frames: Dict[str, np.ndarray], image_keys: List[str]) -> np.ndarray:
        """Stack multiple camera frames horizontally."""
        frames_to_stack = []
        for key in sorted(image_keys):
            if key in collected_frames:
                frame = collected_frames[key]
                # Resize to match height if needed
                if frames_to_stack and frame.shape[0] != frames_to_stack[0].shape[0]:
                    target_height = frames_to_stack[0].shape[0]
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    new_width = int(target_height * aspect_ratio)
                    frame = cv2.resize(frame, (new_width, target_height))
                frames_to_stack.append(frame)

        return cv2.cvtColor(np.hstack(frames_to_stack), cv2.COLOR_RGB2BGR) if frames_to_stack else None

    def _save_video(self, frames: List[np.ndarray], video_path: str, fps: int = 20):
        """Save frames as MP4 video."""
        if not frames:
            return

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        try:
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                video_writer.write(frame)
        finally:
            video_writer.release()

    def _extract_env_data(self, batched_data, env_idx: int):
        """Extract data for a specific environment from batched format (like RolloutWorker)."""
        if isinstance(batched_data, dict):
            return {key: value[env_idx] for key, value in batched_data.items()}
        elif isinstance(batched_data, (list, tuple)) and len(batched_data) > 0:
            return batched_data[env_idx]
        else:
            return batched_data

    def _extract_scalar(self, batched_data, env_idx: int):
        """Extract scalar value for a specific environment."""
        if isinstance(batched_data, (list, np.ndarray)):
            return float(batched_data[env_idx])
        else:
            return float(batched_data)

    def _extract_info(self, infos, env_idx: int):
        """Extract info dict for a specific environment."""
        if infos is None:
            return {}
        if isinstance(infos, list) and env_idx < len(infos):
            return infos[env_idx] if infos[env_idx] is not None else {}
        elif isinstance(infos, dict):
            # Could be a dict with batched values or a single info dict
            # Try to extract per-env if possible, otherwise return the whole dict
            return infos
        return {}
