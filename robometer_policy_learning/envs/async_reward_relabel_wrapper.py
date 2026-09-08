"""
Environment wrapper that relabels rewards asynchronously using reward model.

This wrapper intercepts step() calls, accumulates trajectory context, calls the reward
relabeling service asynchronously, and uses success probabilities to determine episode
termination (done=True).
"""

import threading
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from loguru import logger
from robometer_policy_learning.distributed.clients.reward_relabel_client import (
    RewardRelabelClient,
    PendingRelabel,
    PendingRelabelBatch,
)


def terminal_adjacent_progress_delta(progress_by_step: Dict[int, float]) -> Optional[float]:
    """Return endpoint progress change per physical environment step.

    Chunked RoboTwin transitions retain their physical endpoint indices (7, 15,
    ...), so this normalization is invariant to ``action_exec_len`` and matches
    the calibrated RobotTwin enhanced detector.
    """
    if len(progress_by_step) < 2:
        return None
    ordered_steps = sorted(progress_by_step)
    first_step, last_step = ordered_steps[0], ordered_steps[-1]
    denominator = last_step - first_step
    if denominator <= 0:
        return None
    return float(
        (float(progress_by_step[last_step]) - float(progress_by_step[first_step]))
        / denominator
    )


class AsyncRewardRelabelEnvWrapper(gym.Wrapper):
    """
    Wraps environment to relabel rewards asynchronously using reward model.

    Key features:
    - Accumulates trajectory context (0:t) for each episode
    - Calls reward relabeling service asynchronously every N steps or on episode end
    - Uses success probabilities to determine episode termination (done=True)
    - Returns rewards immediately (placeholder or cached), updates async when available
    - Manages success detection with sliding window/voting mechanism

    Args:
        env: The environment to wrap
        reward_relabel_client: RewardRelabelClient instance for async relabeling
        batch_size: Number of transitions to batch before sending (default: 32)
        success_detection_duration: Number of consecutive steps to check for success (default: 2)
        success_detection_threshold: Success probability threshold (default: 0.65)
        use_relative_rewards: Whether to use relative rewards (delta from previous) (default: False)
        sync_mode: If True, relabeling is synchronous (blocks until complete). If False, async (default: False)
        buffer: Optional buffer reference for retroactive reward updates
    """

    def __init__(
        self,
        env: gym.Env,
        reward_relabel_client: RewardRelabelClient,
        batch_size: int = 32,
        success_detection_duration: int = 2,
        success_detection_threshold: float = 0.65,
        use_relative_rewards: bool = False,
        sync_mode: bool = False,
        buffer=None,  # Optional buffer reference for retroactive reward updates
        action_exec_len: Optional[
            int
        ] = None,  # For DSRL mode: number of actions executed per chunk (None means no chunking)
        wait_for_completion_on_episode_end: bool = False,  # If True, wait for all pending batches to complete before episode terminates
        success_detection_mode: str = "standard",
        terminal_adjacent_delta_detection: bool = False,
        terminal_adjacent_delta_threshold: float = float("inf"),
    ):
        super().__init__(env)

        self.client = reward_relabel_client
        self.buffer = buffer  # Buffer reference for retroactive updates
        self.batch_size = batch_size
        self.success_detection_duration = success_detection_duration
        self.success_detection_threshold = success_detection_threshold
        self.use_relative_rewards = use_relative_rewards
        self.sync_mode = sync_mode  # If True, use synchronous relabeling
        self.action_exec_len = action_exec_len  # For DSRL mode: if > 1, only last step of each chunk is added to buffer
        self.wait_for_completion_on_episode_end = (
            wait_for_completion_on_episode_end  # If True, wait for all batches to complete before episode ends
        )
        if success_detection_mode not in {"standard", "enhanced_terminal_delta"}:
            raise ValueError(f"Unsupported success_detection_mode={success_detection_mode!r}")
        self.success_detection_mode = success_detection_mode
        self.terminal_adjacent_delta_detection = bool(terminal_adjacent_delta_detection)
        self.terminal_adjacent_delta_threshold = float(terminal_adjacent_delta_threshold)
        if self.success_detection_mode == "enhanced_terminal_delta":
            if self.sync_mode:
                raise ValueError(
                    "enhanced_terminal_delta currently requires async mode so the terminal "
                    "request can complete before finalization"
                )
            if not self.terminal_adjacent_delta_detection:
                raise ValueError(
                    "enhanced_terminal_delta requires terminal_adjacent_delta_detection=true"
                )
            if not self.wait_for_completion_on_episode_end:
                raise ValueError(
                    "enhanced_terminal_delta requires wait_for_completion_on_episode_end=true"
                )

        # Trajectory state per episode
        self._trajectory: List[PendingRelabel] = []
        self._episode_id: Optional[Any] = None
        self._language_instruction: Optional[str] = None
        self._step_in_episode: int = 0
        self._trajectory_stage_start_idx: int = 0  # For multi-stage tasks: index into _trajectory where the current subtask begins

        # Rewards and success probabilities (indexed by step_in_episode)
        self._relabeled_rewards: Dict[int, float] = {}  # step_in_episode -> relabeled reward (progress)
        self._relabeled_reward_deltas: Dict[int, float] = {}  # step_in_episode -> relabeled reward delta (progress diff)
        self._env_rewards: Dict[int, float] = {}  # step_in_episode -> env_reward (for summing with relabeled)
        self._success_probs: Dict[int, float] = {}  # step_in_episode -> success_prob (aggregated)
        self._success_probs_by_key: Dict[str, Dict[int, float]] = {}  # image_key -> step_in_episode -> success_prob
        self._progress_predictions_by_key: Dict[str, Dict[int, float]] = {}  # image_key -> step_in_episode -> progress
        self._success_probs_window: deque = deque(maxlen=success_detection_duration)
        self._progress_by_episode: Dict[Any, Dict[int, float]] = {}

        # Track which steps have been processed to prevent duplicates
        self._processed_steps: set = set()  # Set of step_in_episode that have been relabeled

        # Store info dicts per step for retroactive updates when relabeled rewards arrive
        self._info_dicts: Dict[int, Dict[str, Any]] = {}  # step_in_episode -> info dict

        # Previous reward for relative reward computation
        self._prev_reward: float = 0.0

        # Pending batches tracking
        self._pending_batch_indices: int = 0  # Next index to send for relabeling

        # For DSRL mode: track which trajectory indices correspond to buffer entries (last step of each chunk)
        # In DSRL mode, only trajectory indices action_exec_len-1, 2*action_exec_len-1, 3*action_exec_len-1, ... are added to buffer
        self._buffer_indices: List[int] = []  # List of trajectory indices that correspond to buffer entries

        # Thread lock for thread-safe access (must be created before Condition)
        self._lock = threading.Lock()

        # Track pending batches for wait_for_completion mode (per episode)
        # Using dict to track batches by episode_id to handle delayed callbacks correctly
        self._pending_batches_by_episode: Dict[Any, int] = {}  # episode_id -> count of pending batches
        self._completed_batches_by_episode: Dict[Any, int] = {}  # episode_id -> count of completed batches
        self._pending_batches_condition = threading.Condition(
            self._lock
        )  # Condition variable for waiting on batch completion

        # Start client background thread only if using async mode
        # In sync mode, we don't need the background thread since we call _send_batch directly
        if not self.sync_mode and not self.client.running:
            self.client.start()

        # Statistics
        self.stats = {
            "batches_sent": 0,
            "transitions_sent": 0,
            "success_detections": 0,
            "success_head_detections": 0,
            "terminal_delta_detections": 0,
            "episodes": 0,
            "buffer_updates": 0,  # Number of buffer reward updates
            "buffer_update_failures": 0,  # Failed buffer updates
        }

        # Internal episode ID counter (increments on each reset)
        self._episode_counter = 0

    def _get_language_instruction(self, obs: Dict[str, Any], info: Dict[str, Any]) -> str:
        # Try multiple sources for language_instruction:
        # 1. From info dict
        language_instruction = info.get("language_instruction") if isinstance(info, dict) else None

        # 2. If not found, try from observation's "prompt" key (if it's a string)
        if language_instruction is None and isinstance(obs, dict):
            prompt_value = obs.get("prompt")
            if isinstance(prompt_value, str):
                language_instruction = prompt_value
            elif isinstance(prompt_value, np.ndarray):
                # Handle numpy string scalar (0D array) or 1D array with single string
                if prompt_value.ndim == 0:
                    # 0D numpy array (scalar)
                    language_instruction = str(prompt_value.item())
                elif prompt_value.ndim == 1 and len(prompt_value) > 0:
                    # 1D array, take first element
                    language_instruction = str(prompt_value.flat[0])
            elif isinstance(prompt_value, (list, tuple)) and len(prompt_value) > 0:
                # Handle vectorized case where prompt might be a list
                first_item = prompt_value[0]
                if isinstance(first_item, str):
                    language_instruction = first_item
                elif isinstance(first_item, np.ndarray) and first_item.ndim == 0:
                    language_instruction = str(first_item.item())

        # 3. If still not found, try from wrapped environment's language_instruction attribute
        if language_instruction is None:
            language_instruction = getattr(self.env, "language_instruction", None)

        return language_instruction

    def reset(self, seed=None, options=None):
        """Reset environment and clear trajectory state."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Clear trajectory state
        with self._lock:
            # Log state before clearing for debugging
            prev_relabeled_count = len(self._relabeled_rewards)
            prev_relabeled_sum = sum(self._relabeled_rewards.values()) if self._relabeled_rewards else 0.0
            prev_env_rewards_sum = sum(self._env_rewards.values()) if self._env_rewards else 0.0
            prev_episode_id = self._episode_id
            
            self._trajectory = []
            self._step_in_episode = 0
            self._pending_batch_indices = 0
            self._prev_reward = 0.0
            self._success_probs_window.clear()
            self._relabeled_rewards.clear()
            self._relabeled_reward_deltas.clear()
            self._env_rewards.clear()
            self._success_probs.clear()
            self._success_probs_by_key.clear()
            self._progress_predictions_by_key.clear()
            self._processed_steps.clear()
            self._info_dicts.clear()
            self._buffer_indices = []  # Clear buffer indices tracking
            self._trajectory_stage_start_idx = 0  # Reset stage boundary
            self._last_obs = obs  # Store initial observation
            self._success_head_detected = False
            self._success_detection_source = "none"
            self._terminal_adjacent_delta = None

            # Note: Don't clear _pending_batches_by_episode/_completed_batches_by_episode
            # They track ALL episodes (including past ones with delayed callbacks)
            # We'll initialize counters for the new episode when first batch is sent
            
            # Log that reset happened and state was cleared
            logger.info(
                f"[Async] RESET: Cleared state from prev_episode_id={prev_episode_id}. "
                f"Cleared {prev_relabeled_count} relabeled rewards (sum={prev_relabeled_sum:.4f}), "
                f"env_rewards_sum={prev_env_rewards_sum:.4f}. "
                f"After clear: relabeled_rewards={len(self._relabeled_rewards)}, env_rewards={len(self._env_rewards)}"
            )

            # Generate episode_id internally (increment counter on each reset)
            # If environment provides episode_id in info, use it; otherwise use internal counter
            if isinstance(info, dict) and "episode_id" in info:
                self._episode_id = info.get("episode_id")
            else:
                self._episode_id = self._episode_counter
                self._episode_counter += 1

            self._language_instruction = self._get_language_instruction(obs, info)

            logger.trace(
                f"[Async] Reset: episode_id={self._episode_id}, "
                f"language_instruction={self._language_instruction}, "
                f"trajectory_length={len(self._trajectory)}"
            )

        return obs, info

    def _wait_for_episode_labels_locked(self) -> bool:
        """Wait for all reward requests for the current episode.

        The caller must hold ``self._lock``. ``Condition.wait`` temporarily
        releases it so the reward callback can publish the terminal score.
        """
        if self.sync_mode or not self.wait_for_completion_on_episode_end:
            return True
        pending = self._pending_batches_by_episode.get(self._episode_id, 0)
        if pending <= 0:
            return True
        for _ in range(60):
            completed = self._completed_batches_by_episode.get(self._episode_id, 0)
            if completed >= pending:
                return True
            self._pending_batches_condition.wait(timeout=1.0)
        completed = self._completed_batches_by_episode.get(self._episode_id, 0)
        logger.warning(
            f"[Async] Timed out waiting for terminal labels: episode_id={self._episode_id}, "
            f"pending={pending}, completed={completed}"
        )
        return completed >= pending

    def _confirm_reward_model_success(self, env_reward: float):
        """Apply the official wrapper's success-check handshake."""
        done_from_success = True
        new_obs = None
        new_info = None
        blocked = False
        if hasattr(self.env.unwrapped, "send_success_check_done"):
            done_from_success, new_obs, new_info, blocked = (
                self.env.unwrapped.send_success_check_done()
            )
            if isinstance(new_info, dict):
                env_reward = new_info.get("new_reward", env_reward)
        return bool(done_from_success), env_reward, new_obs, new_info, bool(blocked)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Step environment and handle async reward relabeling.

        Returns:
            obs: Observation
            reward: Relabeled reward (placeholder if not yet available)
            done: Episode termination (from env or success detection)
            truncated: Episode truncation (from env)
            info: Info dict with success information
        """
        # Detect chunking: check if action is 2D (chunked) or if num_steps is in info
        # We'll check info after stepping, but we can also check action shape
        is_chunked = action.ndim == 2 if isinstance(action, np.ndarray) else False

        # Step underlying environment
        obs, env_reward, done, truncated, info = self.env.step(action)

        # Check for chunking from info dict (more reliable)
        num_steps = None
        if isinstance(info, dict) and "num_steps" in info:
            num_steps = int(info["num_steps"])
            is_chunked = True
        elif is_chunked and isinstance(action, np.ndarray):
            # Fallback: use action shape if num_steps not in info
            num_steps = action.shape[0]

        # Store previous observation (from last step or reset)
        prev_obs = getattr(self, "_last_obs", obs)

        with self._lock:
            # If chunking is enabled, update step_in_episode to reflect the actual step count
            # When chunking: server executes N steps internally, but wrapper step() is called once
            # We want step_in_episode to be the last step of the chunk (e.g., if chunk is 20 steps starting at 0, final step is 19)
            if is_chunked and num_steps is not None:
                # Update step_in_episode to the last step of the chunk
                # If we're at step 0 and execute a chunk of 20 steps, we want step_in_episode to be 19
                self._step_in_episode = self._step_in_episode + num_steps - 1

        # Create pending transition (only one per chunk when chunking is enabled)
        pending = PendingRelabel(
            obs=prev_obs,
            action=action,
            reward=env_reward,
            next_obs=obs,
            done=done,
            truncated=truncated,
            episode_id=self._episode_id,
            step_in_episode=self._step_in_episode,
            timestamp=None,  # Could add timestamp if needed
            language_instruction=self._language_instruction,
        )

        with self._lock:
            # Add to trajectory
            trajectory_idx = len(self._trajectory)  # Index of this transition in trajectory (before appending)
            self._trajectory.append(pending)
            logger.trace(
                f"[Async] Step {self._step_in_episode}, trajectory_length={len(self._trajectory)}, trajectory_idx={trajectory_idx}, env_reward={env_reward:.4f}, done={done}, truncated={truncated}"
            )

            # In DSRL mode (action_exec_len > 1), track which trajectory indices correspond to buffer entries
            # This is only needed when chunking is NOT happening at the wrapper level (is_chunked=False)
            # When is_chunked=True, we already get one transition per chunk, so we use sequential indexing
            if self.action_exec_len is not None and self.action_exec_len > 1 and not is_chunked:
                # Check if this is the last step of a chunk (trajectory index is action_exec_len-1, 2*action_exec_len-1, etc.)
                # trajectory_idx is 0-indexed, so indices action_exec_len-1, 2*action_exec_len-1, ... are buffer entries
                if (trajectory_idx + 1) % self.action_exec_len == 0:
                    # This is the last step of a chunk, add to buffer indices
                    self._buffer_indices.append(trajectory_idx)

            # Check if we should send a batch for relabeling
            # When chunking is happening (is_chunked=True), use sequential indexing
            # When action_exec_len > 1 but not chunked, use buffer indices
            if self.action_exec_len is not None and self.action_exec_len > 1 and not is_chunked:
                # Count buffer indices that haven't been sent yet
                num_pending = sum(
                    1 for idx in self._buffer_indices if idx >= self._pending_batch_indices
                )
            else:
                # Chunked mode or normal mode: count all pending trajectory indices sequentially
                num_pending = len(self._trajectory) - self._pending_batch_indices

            should_send_batch = False

            if done or truncated:
                # Episode terminated, send all remaining transitions
                if num_pending > 0:
                    should_send_batch = True
            elif num_pending >= self.batch_size:
                # Have enough transitions for a batch
                should_send_batch = True

            # Prepare batch if needed (inside lock to access trajectory state)
            batch = None
            if should_send_batch:
                # Get trajectory context (full context for reward model)
                trajectory_context = self._trajectory

                # When chunking is detected (is_chunked=True), use sequential range
                # When action_exec_len > 1 but not chunked, use batch_indices
                if self.action_exec_len is not None and self.action_exec_len > 1 and not is_chunked:
                    # DSRL mode without chunking: collect specific indices that correspond to buffer entries
                    batch_indices_list = [idx for idx in self._buffer_indices if idx >= self._pending_batch_indices]

                    # Limit to batch_size if not done/truncated
                    if not (done or truncated) and len(batch_indices_list) > self.batch_size:
                        batch_indices_list = batch_indices_list[: self.batch_size]

                    batch_size = len(batch_indices_list)

                    # Create batch with specific indices
                    batch = PendingRelabelBatch(
                        transitions=trajectory_context,  # Full trajectory context for reward model
                        batch_indices=batch_indices_list,  # Send specific indices
                        episode_id=self._episode_id,
                        language_instruction=self._language_instruction,
                        callback=self._on_batch_relabeled,
                        trajectory_start_idx=self._trajectory_stage_start_idx,
                    )

                    # Update pending_batch_indices to the max of the sent indices + 1
                    self._pending_batch_indices = (
                        max(batch_indices_list) + 1 if batch_indices_list else self._pending_batch_indices
                    )
                else:   
                    # Chunked mode or normal mode: use sequential range
                    if done or truncated:
                        batch_size = num_pending
                    else:
                        batch_size = min(self.batch_size, num_pending)

                    # Create batch with sequential range
                    batch = PendingRelabelBatch(
                        transitions=trajectory_context,
                        batch_start_idx=self._pending_batch_indices,
                        batch_end_idx=self._pending_batch_indices + batch_size,
                        episode_id=self._episode_id,
                        language_instruction=self._language_instruction,
                        callback=self._on_batch_relabeled,
                        trajectory_start_idx=self._trajectory_stage_start_idx,
                    )

                    # Update indices and stats inside lock
                    self._pending_batch_indices += batch_size

                self.stats["batches_sent"] += 1
                self.stats["transitions_sent"] += batch_size

                # Track pending batch if wait_for_completion mode is enabled (only for async mode)
                # In sync mode, batches complete immediately, so no need to track
                if self.wait_for_completion_on_episode_end and not self.sync_mode:
                    # Initialize episode counters if first batch for this episode
                    if self._episode_id not in self._pending_batches_by_episode:
                        self._pending_batches_by_episode[self._episode_id] = 0
                        self._completed_batches_by_episode[self._episode_id] = 0
                    self._pending_batches_by_episode[self._episode_id] += 1

                # For async mode, send immediately (just queues, doesn't block)
                if not self.sync_mode:
                    self.client.relabel_batch(batch)
                    pending_for_episode = self._pending_batches_by_episode.get(self._episode_id, 0)
                    if batch.batch_indices is not None:
                        logger.trace(
                            f"[Async] Sent batch for async relabeling (DSRL non-chunked mode): "
                            f"batch_size={batch_size}, batch_indices={batch.batch_indices}, episode_id={self._episode_id}, "
                            f"total_batches_sent={self.stats['batches_sent']}, pending_batches={pending_for_episode}"
                        )
                    else:
                        mode_str = "chunked" if is_chunked else "normal"
                        logger.trace(
                            f"[Async] Sent batch for async relabeling ({mode_str} mode): "
                            f"batch_size={batch_size}, batch_start_idx={batch.batch_start_idx}, "
                            f"batch_end_idx={batch.batch_end_idx}, episode_id={self._episode_id}, "
                            f"total_batches_sent={self.stats['batches_sent']}, pending_batches={pending_for_episode}"
                        )

            terminal_labels_complete = True
            if done or truncated:
                terminal_labels_complete = self._wait_for_episode_labels_locked()

            # Check success detection window for termination using voting across image keys
            # The window is updated asynchronously by callbacks, so we check it here
            done_from_success = False
            if len(self._success_probs_window) >= self.success_detection_duration:
                # Voting: check if majority of image keys agree on success
                window_probs = list(self._success_probs_window)
                logger.trace(
                    f"[Async] Success window check: "
                    f"window_size={len(self._success_probs_window)}, "
                    f"window_probs={[f'{p:.3f}' for p in window_probs]}, "
                    f"threshold={self.success_detection_threshold}"
                )

                # Get success probabilities by key for current step
                current_step = self._step_in_episode
                success_votes = 0
                total_votes = 0
                for image_key, success_probs_dict in self._success_probs_by_key.items():
                    if current_step in success_probs_dict:
                        total_votes += 1
                        if success_probs_dict[current_step] >= self.success_detection_threshold:
                            success_votes += 1

                # Also check aggregated window (backwards compatibility)
                high_success_count = sum(
                    1 for sp in self._success_probs_window if sp >= self.success_detection_threshold
                )

                # Success if: (1) majority of image keys vote success, OR (2) aggregated window shows success
                if total_votes > 0:
                    # Use voting across image keys
                    majority_success = success_votes > (total_votes / 2)
                    if majority_success:
                        if hasattr(self.env.unwrapped, "send_success_check_done"):
                            logger.info(f"[Async] Sending success check done to environment")
                            done_from_success, new_obs, new_info, blocked = self.env.unwrapped.send_success_check_done()
                            # update reward if new reward is available
                            if isinstance(new_info, dict):
                                env_reward = new_info.get("new_reward", env_reward)
                                logger.info(f"!!!!!! Setting new env reward of: {env_reward}")
                            logger.info(f"!!!!!! SEND SUCCESS CHECK DONE TO ENVIRONMENT: {done_from_success}")
                            if new_obs is not None and not blocked:
                                # in case there's an extra obs wrapper
                                if hasattr(self.env, "_format_obs"):
                                    new_obs = self.env._format_obs(new_obs)
                                obs = new_obs
                                info = new_info
                                self._language_instruction = self._get_language_instruction(new_obs, new_info)
                                # Advance stage boundary so reward model only sees frames from this subtask onward
                                self._trajectory_stage_start_idx = len(self._trajectory)
                                logger.info(f"[Async] Stage advanced (voting): trajectory_stage_start_idx={self._trajectory_stage_start_idx}")
                        else:
                            done_from_success = True

                        self.stats["success_detections"] += 1
                        if isinstance(info, dict):
                            info["is_success"] = done_from_success
                            info["success_from_reward_model"] = True
                        logger.trace(
                            f"[Async] Success detected from reward model (voting): "
                            f"step={current_step}, success_votes={success_votes}/{total_votes}, "
                            f"total_detections={self.stats['success_detections']}"
                        )
                elif high_success_count > (self.success_detection_duration / 2):
                    # Fallback to aggregated window if no per-key data
                    if hasattr(self.env.unwrapped, "send_success_check_done"):
                        logger.info(f"[Async] Sending success check done to environment")
                        done_from_success, new_obs, new_info, blocked = self.env.unwrapped.send_success_check_done()
                        # update reward if new reward is available
                        if isinstance(new_info, dict):
                            env_reward = new_info.get("new_reward", env_reward)
                            logger.info(f"!!!!!! Setting new env reward upon success check of: {env_reward}")
                        logger.info(f"!!!!!! SEND SUCCESS CHECK DONE TO ENVIRONMENT: {done_from_success}")
                        if new_obs is not None and not blocked:
                            # in case there's an extra obs wrapper
                            if hasattr(self.env, "_format_obs"):
                                new_obs = self.env._format_obs(new_obs)
                            obs = new_obs
                            info = new_info
                            self._language_instruction = self._get_language_instruction(new_obs, new_info)
                            # Advance stage boundary so reward model only sees frames from this subtask onward
                            self._trajectory_stage_start_idx = len(self._trajectory)
                            logger.info(f"[Async] Stage advanced (aggregated): trajectory_stage_start_idx={self._trajectory_stage_start_idx}")
                    else:
                        done_from_success = True
                        
                    self.stats["success_detections"] += 1
                    if isinstance(info, dict):
                        info["is_success"] = done_from_success
                        info["success_from_reward_model"] = True
                    logger.trace(
                        f"[Async] Success detected from reward model (aggregated): "
                        f"step={current_step}, high_success_count={high_success_count}, "
                        f"total_detections={self.stats['success_detections']}"
                    )

            # Update done flag if success detected
            if done_from_success:
                logger.info(f"[Async] Success detected from reward model, marking episode as done")
                self._success_head_detected = True
                self._success_detection_source = "success_prob"
                self.stats["success_head_detections"] += 1
                done = True
                truncated = False
                if isinstance(info, dict):
                    info["success_detection_source"] = "success_prob"
                    info["success_from_reward_model"] = True

            if (
                not done_from_success
                and truncated
                and self.success_detection_mode == "enhanced_terminal_delta"
                and self.terminal_adjacent_delta_detection
                and not self._success_head_detected
                and terminal_labels_complete
            ):
                terminal_delta = terminal_adjacent_progress_delta(self._relabeled_rewards)
                self._terminal_adjacent_delta = terminal_delta
                if isinstance(info, dict):
                    info["terminal_adjacent_progress_delta"] = terminal_delta
                if (
                    terminal_delta is not None
                    and terminal_delta > self.terminal_adjacent_delta_threshold
                ):
                    (
                        done_from_success,
                        env_reward,
                        new_obs,
                        new_info,
                        blocked,
                    ) = self._confirm_reward_model_success(env_reward)
                    if done_from_success:
                        done = True
                        truncated = False
                        self._success_detection_source = "terminal_adjacent_delta"
                        self.stats["success_detections"] += 1
                        self.stats["terminal_delta_detections"] += 1
                        if new_obs is not None and not blocked:
                            if hasattr(self.env, "_format_obs"):
                                new_obs = self.env._format_obs(new_obs)
                            obs = new_obs
                            self._language_instruction = self._get_language_instruction(
                                new_obs, new_info or {}
                            )
                        if isinstance(info, dict):
                            info["is_success"] = True
                            info["success_from_reward_model"] = True
                            info["success_detection_source"] = "terminal_adjacent_delta"
                        logger.info(
                            "[Async] Enhanced terminal success: "
                            f"delta={terminal_delta:.8f} > "
                            f"threshold={self.terminal_adjacent_delta_threshold:.8f}"
                        )

            # Store env_reward for this step (needed for summing with relabeled rewards later)
            self._env_rewards[self._step_in_episode] = float(env_reward)

            # Get reward for this step (env_reward + relabeled if available)
            reward = self._get_reward(self._step_in_episode, env_reward)

            # Add relabeled reward info to info dict if available
            if isinstance(info, dict):
                # Create a copy of info dict to store (so we can update it later)
                info_copy = info.copy()

                # Add step_in_episode so rollout worker can pass it to buffer
                # Ensure it's a Python int (not numpy scalar) for hashability
                step_in_episode_int = int(self._step_in_episode)
                info["step_in_episode"] = step_in_episode_int
                info_copy["step_in_episode"] = step_in_episode_int

                if self._step_in_episode in self._relabeled_rewards:
                    relabeled_reward = float(self._relabeled_rewards[self._step_in_episode])
                    if self.use_relative_rewards:
                        relabeled_reward = float(
                            self._relabeled_reward_deltas.get(
                                self._step_in_episode,
                                relabeled_reward
                                - self._get_prev_progress(self._step_in_episode, self._relabeled_rewards),
                            )
                        )
                    info["relabeled_reward"] = relabeled_reward
                    info_copy["relabeled_reward"] = relabeled_reward
                else:
                    # Not yet relabeled - indicate this in info
                    info["relabeled_reward"] = None
                    info_copy["relabeled_reward"] = None

                # Add success probability if available
                if self._step_in_episode in self._success_probs:
                    success_prob = float(self._success_probs[self._step_in_episode])
                    info["success_prob"] = success_prob
                    info_copy["success_prob"] = success_prob
                else:
                    info["success_prob"] = None
                    info_copy["success_prob"] = None

                # Always include env_reward and total_reward for consistency
                info["env_reward"] = float(env_reward)
                info["total_reward"] = float(reward)
                info_copy["env_reward"] = float(env_reward)
                info_copy["total_reward"] = float(reward)

                # Store info dict for retroactive updates when relabeled rewards arrive
                self._info_dicts[self._step_in_episode] = info_copy

            # Update step counter
            self._step_in_episode += 1
            self._last_obs = obs

            # Clean up on episode end
            if done or truncated:
                self.stats["episodes"] += 1
                logger.trace(
                    f"[Async] Episode ended: step={self._step_in_episode}, "
                    f"done={done}, truncated={truncated}, trajectory_length={len(self._trajectory)}, "
                    f"pending_batches={self._pending_batch_indices}, total_episodes={self.stats['episodes']}"
                )

        # For sync mode, release lock before blocking call to avoid deadlock
        # The callback needs to acquire the lock, so we must release it first
        if should_send_batch and self.sync_mode and batch is not None:
            # Synchronous mode: blocks until relabeling is complete
            # Must be outside the lock since callback needs to acquire it
            self.client.relabel_batch_sync(batch)
            logger.trace(
                f"[Async] Synchronously relabeled batch: "
                f"batch_size={batch_size}, batch_start_idx={batch.batch_start_idx}, "
                f"batch_end_idx={batch.batch_end_idx}, episode_id={self._episode_id}"
            )

        return obs, reward, done, truncated, info

    def _get_prev_progress(self, step_idx: int, progress_by_step: Dict[int, float]) -> float:
        """Find the most recent progress value before step_idx."""
        if step_idx <= 0 or not progress_by_step:
            return 0.0
        prev_step = step_idx - 1
        if prev_step in progress_by_step:
            return progress_by_step[prev_step]
        prev_steps = [s for s in progress_by_step.keys() if s < step_idx]
        return progress_by_step[max(prev_steps)] if prev_steps else 0.0

    def _get_reward(self, step_idx: int, env_reward: float) -> float:
        """
        Get reward for step - return env_reward + relabeled reward (if available).
        If relabeled reward not yet available, return env_reward.

        Note: This method assumes the caller already holds self._lock.
        """
        # Get env_reward for this step (should be stored, but fallback to parameter)
        step_env_reward = self._env_rewards.get(step_idx, env_reward)

        if step_idx in self._relabeled_rewards:
            # Relabeled reward available - sum with env_reward
            relabeled_reward = self._relabeled_rewards[step_idx]
            if self.use_relative_rewards:
                relabeled_delta = self._relabeled_reward_deltas.get(
                    step_idx, relabeled_reward - self._get_prev_progress(step_idx, self._relabeled_rewards)
                )
                reward = step_env_reward + relabeled_delta
                reward_source = "env_plus_relabeled_relative_progress"
            else:
                reward = step_env_reward + relabeled_reward
                reward_source = "env_plus_relabeled"
        else:
            # Not yet relabeled - return env_reward
            reward = step_env_reward
            reward_source = "env"

        logger.trace(
            f"[Async] _get_reward: step_idx={step_idx}, reward={reward:.4f}, source={reward_source}, env_reward={step_env_reward:.4f}"
        )
        return reward

    def _on_batch_relabeled(
        self,
        success_probs: List[float],
        batch: PendingRelabelBatch,
        progress_predictions_by_key: Dict[str, Dict[str, List[float]]] = None,
    ):
        """
        Callback when batch reward relabeling is complete. Updates internal state and buffer retroactively.

        Args:
            success_probs: List of aggregated success probabilities (backwards compatibility)
            batch: Batch metadata
            progress_predictions_by_key: Dict mapping image_key -> {"progress": [...], "success_probs": [...]}
        """
        logger.debug(f"="*50)
        logger.success(f"[Async] _on_batch_relabeled: Received batch callback")
        episode_id = batch.episode_id
        batch_start_idx = batch.batch_start_idx
        buffer_updates = []  # For reward updates: List of (episode_id, step_idx, combined_reward)
        buffer_info_updates = []  # For info updates: List of (episode_id, step_idx, info_dict)

        # Track batch completion for wait_for_completion mode
        batch_completed = False

        if progress_predictions_by_key is None:
            progress_predictions_by_key = {}

        # Determine batch size from progress_predictions_by_key
        batch_size = 0
        if progress_predictions_by_key:
            first_key = next(iter(progress_predictions_by_key))
            batch_size = len(progress_predictions_by_key[first_key]["progress"])
        else:
            batch_size = len(success_probs) if success_probs else 0

        logger.debug(
            f"[Async] _on_batch_relabeled: Received batch callback: "
            f"episode_id={episode_id}, batch_start_idx={batch_start_idx} batch_size={batch_size}"
        )

        with self._lock:
            # Check if this batch is for the current episode or a past episode
            is_current_episode = (
                episode_id is not None and self._episode_id is not None and episode_id == self._episode_id
            )

            if not is_current_episode:
                logger.info(
                    f"[Async] Processing delayed batch from previous episode: "
                    f"callback_episode_id={episode_id}, current_episode_id={self._episode_id}. "
                    f"Will update buffer but skip internal state updates."
                )

            # Validate batch size consistency across image keys
            if progress_predictions_by_key:
                batch_sizes = {
                    len(predictions.get("progress", [])) for predictions in progress_predictions_by_key.values()
                }
                if len(batch_sizes) > 1:
                    logger.warning(
                        f"[Async] Inconsistent batch sizes across image keys: {batch_sizes}, "
                        f"using size={batch_size} from first key"
                    )
                # Also check success_probs lengths match
                success_probs_sizes = {
                    len(predictions.get("success_probs", [])) for predictions in progress_predictions_by_key.values()
                }
                if len(success_probs_sizes) > 1:
                    logger.warning(
                        f"[Async] Inconsistent success_probs sizes across image keys: {success_probs_sizes}"
                    )

            # Initialize dicts for new image keys (only for current episode)
            if is_current_episode:
                for image_key in progress_predictions_by_key.keys():
                    if image_key not in self._success_probs_by_key:
                        self._success_probs_by_key[image_key] = {}
                    if image_key not in self._progress_predictions_by_key:
                        self._progress_predictions_by_key[image_key] = {}

            # Update rewards and success probabilities for the batch
            duplicates_skipped = 0
            processed_steps_in_batch = set()

            # Check if batch_indices is provided (DSRL mode)
            if batch.batch_indices is not None:
                # DSRL mode: use specific indices
                batch_indices_list = batch.batch_indices
            else:
                # Normal mode: use sequential indices
                batch_indices_list = None

            for i in range(batch_size):
                if batch_indices_list is not None:
                    # DSRL mode: use specific trajectory index
                    trajectory_idx = batch_indices_list[i]
                else:
                    # Normal mode: use sequential index
                    trajectory_idx = batch_start_idx + i

                # Get step_in_episode from the batch's transitions (works even for past episodes)
                # The batch contains the full trajectory context, so we can get step_in_episode from there
                if trajectory_idx < len(batch.transitions):
                    pending_transition = batch.transitions[trajectory_idx]
                    actual_step_in_episode = pending_transition.step_in_episode
                    # Also get env_reward from the transition (for buffer updates)
                    transition_env_reward = pending_transition.reward
                else:
                    logger.warning(
                        f"[Async._on_batch_relabeled] trajectory_idx={trajectory_idx} out of range "
                        f"(trajectory_length={len(batch.transitions)}, batch_start_idx={batch_start_idx}, batch_size={batch_size})"
                    )
                    continue

                # Use actual_step_in_episode for all indexing (this matches what's in the buffer)
                step_idx = actual_step_in_episode
                processed_steps_in_batch.add(step_idx)

                # Compute progress from progress_predictions_by_key (average across image keys)
                progress_values = []
                key_success_probs = []
                for image_key, predictions in progress_predictions_by_key.items():
                    if i < len(predictions.get("progress", [])):
                        progress_key = predictions["progress"][i]
                        progress_values.append(float(progress_key))
                        # Only update internal state if this is the current episode
                        if is_current_episode:
                            self._progress_predictions_by_key[image_key][step_idx] = float(progress_key)
                    if i < len(predictions.get("success_probs", [])):
                        success_prob_key = predictions["success_probs"][i]
                        key_success_probs.append(float(success_prob_key))
                        # Only update internal state if this is the current episode
                        if is_current_episode:
                            self._success_probs_by_key[image_key][step_idx] = float(success_prob_key)

                # Average progress across image keys (this is the relabeled reward = progress)
                avg_progress = float(np.mean(progress_values)) if progress_values else 0.0
                
                # Log if progress is outside expected [0, 1] range
                if avg_progress < 0.0 or avg_progress > 1.0:
                    logger.warning(
                        f"[Async] Progress out of range! avg_progress={avg_progress:.4f} "
                        f"(step_idx={step_idx}, episode_id={episode_id}, progress_values={progress_values})"
                    )

                # Compute aggregated success prob from per-key data (average across keys)
                aggregated_success_prob = float(np.mean(key_success_probs)) if key_success_probs else 0.0

                # Compute relative progress (if enabled) using per-episode progress history
                relabeled_delta = avg_progress
                if self.use_relative_rewards and episode_id is not None:
                    episode_progress = self._progress_by_episode.setdefault(episode_id, {})
                    prev_progress = self._get_prev_progress(step_idx, episode_progress)
                    relabeled_delta = avg_progress - prev_progress
                    episode_progress[step_idx] = avg_progress

                # Only update internal state if this is the current episode (skip for stale episodes)
                if is_current_episode:
                    # Check if this step has already been processed (prevent duplicates)
                    if step_idx in self._processed_steps:
                        duplicates_skipped += 1
                        logger.debug(
                            f"[Async] Skipping duplicate processing for step_idx={step_idx} "
                            f"(episode_id={episode_id}, batch_start_idx={batch_start_idx})"
                        )
                        # Still process buffer updates even if we skip internal state
                    else:
                        self._relabeled_rewards[step_idx] = avg_progress
                        if self.use_relative_rewards:
                            self._relabeled_reward_deltas[step_idx] = relabeled_delta
                        self._success_probs[step_idx] = aggregated_success_prob
                        self._processed_steps.add(step_idx)
                        
                        # # Debug log for reward storage
                        # logger.debug(
                        #     f"[Async] Stored relabeled_reward: step_idx={step_idx}, "
                        #     f"progress={avg_progress:.4f}, total_relabeled_count={len(self._relabeled_rewards)}"
                        # )

                        # Update stored info dict retroactively with relabeled reward
                        if step_idx in self._info_dicts:
                            self._info_dicts[step_idx]["relabeled_reward"] = float(
                                relabeled_delta if self.use_relative_rewards else avg_progress
                            )
                            # Also update total_reward if env_reward is available
                            if step_idx in self._env_rewards:
                                step_env_reward = self._env_rewards[step_idx]
                                self._info_dicts[step_idx]["env_reward"] = float(step_env_reward)
                                self._info_dicts[step_idx]["total_reward"] = float(
                                    step_env_reward
                                    + (relabeled_delta if self.use_relative_rewards else avg_progress)
                                )

                        # Add aggregated success prob to window (for backwards compatibility fallback)
                        # The window will maintain the most recent N values
                        self._success_probs_window.append(aggregated_success_prob)

                # Always prepare buffer updates (even for past episodes) - buffer can handle any episode_id
                if self.buffer is not None and episode_id is not None:
                    # Use env_reward from transition if available, otherwise try to get from stored dict
                    # For past episodes, we might not have _env_rewards, so use transition's reward
                    step_env_reward = (
                        transition_env_reward
                        if not is_current_episode
                        else self._env_rewards.get(step_idx, transition_env_reward)
                    )
                    # Combined reward = env_reward + relabeled reward (relative if enabled)
                    combined_reward = step_env_reward + (
                        relabeled_delta if self.use_relative_rewards else avg_progress
                    )
                    
                    # Log warning if combined_reward is unexpectedly high
                    if combined_reward > 1.0:
                        logger.warning(
                            f"[Async] High combined_reward={combined_reward:.4f}! "
                            f"env_reward={step_env_reward:.4f}, progress={avg_progress:.4f}, "
                            f"step_idx={step_idx}, episode_id={episode_id}"
                        )
                    
                    # logger.debug(
                    #     f"[Async._on_batch_relabeled] Preparing buffer update {i}: "
                    #     f"episode_id={episode_id} (type={type(episode_id).__name__}), step_idx={step_idx} (type={type(step_idx).__name__}), "
                    #     f"combined_reward={combined_reward}, is_current_episode={is_current_episode}"
                    # )
                    buffer_updates.append((episode_id, step_idx, combined_reward))

                    # Prepare info dict update with relabeled reward information
                    info_update = {
                        "relabeled_reward": float(
                            relabeled_delta if self.use_relative_rewards else avg_progress
                        ),
                        "env_reward": float(step_env_reward),
                        "total_reward": float(combined_reward),
                        "success_prob": float(aggregated_success_prob),
                    }
                    buffer_info_updates.append((episode_id, step_idx, info_update))

            if duplicates_skipped > 0:
                logger.warning(
                    f"[Async] Skipped {duplicates_skipped} duplicate step(s) in batch "
                    f"(episode_id={episode_id}, batch_start_idx={batch_start_idx}, batch_size={batch_size})"
                )

            # logger.trace(
            #     f"[Async] _on_batch_relabeled: Updated internal state: "
            #     f"relabeled_rewards_count={len(self._relabeled_rewards)}, "
            #     f"success_probs_count={len(self._success_probs)}, "
            #     f"success_probs_by_key={[(k, len(v)) for k, v in self._success_probs_by_key.items()]}, "
            #     f"success_window_size={len(self._success_probs_window)}, "
            #     f"buffer_updates_prepared={len(buffer_updates)}, "
            #     f"duplicates_skipped={duplicates_skipped}"
            # )

            # Mark batch as completed for wait_for_completion mode
            # Track completion for ANY episode (including past ones) to handle delayed callbacks
            if self.wait_for_completion_on_episode_end and episode_id is not None:
                # Initialize counter if first callback for this episode
                if episode_id not in self._completed_batches_by_episode:
                    self._completed_batches_by_episode[episode_id] = 0
                self._completed_batches_by_episode[episode_id] += 1
                batch_completed = True
                
                pending_for_episode = self._pending_batches_by_episode.get(episode_id, 0)
                completed_for_episode = self._completed_batches_by_episode[episode_id]
                
                logger.debug(
                    f"[Async] Batch completed: "
                    f"pending={pending_for_episode}, completed={completed_for_episode} "
                    f"(episode_id={episode_id}, is_current={is_current_episode})"
                )
                # Notify any threads waiting for batch completion (in case they're waiting for this episode)
                self._pending_batches_condition.notify_all()

            # If we processed a terminal step for this episode, clear per-episode progress history
            if self.use_relative_rewards and episode_id is not None and episode_id in self._progress_by_episode:
                terminal_steps = [
                    tr.step_in_episode for tr in batch.transitions if tr.done or tr.truncated
                ]
                if terminal_steps:
                    final_step = max(terminal_steps)
                    if final_step in processed_steps_in_batch:
                        self._progress_by_episode.pop(episode_id, None)

        # Update buffer retroactively (outside lock to avoid deadlock)
        if self.buffer is not None and (buffer_updates or buffer_info_updates):
            logger.debug(
                f"[Async._on_batch_relabeled] Updating buffer with {len(buffer_updates)} rewards "
                f"and {len(buffer_info_updates)} info dicts (episode_id={episode_id}, type={type(episode_id).__name__})"
            )
            # if buffer_updates:
            #     logger.warning(
            #         f"[Async._on_batch_relabeled] Buffer updates details: "
            #         f"episode_id={episode_id} (type={type(episode_id).__name__}), "
            #         f"step_range=[{min(step for _, step, _ in buffer_updates)}, {max(step for _, step, _ in buffer_updates)}], "
            #         f"sample updates: {buffer_updates[:3]}"
            #     )
            # if buffer_info_updates:
            #     logger.warning(
            #         f"[Async._on_batch_relabeled] Buffer info updates details: "
            #         f"episode_id={episode_id} (type={type(episode_id).__name__}), "
            #         f"step_range=[{min(step for _, step, _ in buffer_info_updates)}, {max(step for _, step, _ in buffer_info_updates)}], "
            #         f"sample updates: {[(ep, step, list(info.keys())) for ep, step, info in buffer_info_updates[:3]]}"
            #     )

            # Update rewards
            if buffer_updates:
                # logger.warning(
                #     f"[Async._on_batch_relabeled] Calling buffer.update_rewards_batch with {len(buffer_updates)} updates"
                # )
                updated_count = self.buffer.update_rewards_batch(buffer_updates)
                # logger.warning(
                #     f"[Async._on_batch_relabeled] buffer.update_rewards_batch returned: {updated_count}/{len(buffer_updates)}"
                # )

                with self._lock:
                    self.stats["buffer_updates"] += updated_count
                    self.stats["buffer_update_failures"] += len(buffer_updates) - updated_count

                if updated_count < len(buffer_updates):
                    logger.warning(
                        f"[Async] Only updated {updated_count}/{len(buffer_updates)} "
                        f"rewards in buffer (episode_id={episode_id}, type={type(episode_id).__name__})"
                    )

            # Update info dicts
            if buffer_info_updates:
                # logger.debug(
                #     f"[Async._on_batch_relabeled] Calling buffer.update_info_batch with {len(buffer_info_updates)} updates"
                # )
                info_updated_count = self.buffer.update_info_batch(buffer_info_updates)
                # logger.debug(
                #     f"[Async._on_batch_relabeled] buffer.update_info_batch returned: {info_updated_count}/{len(buffer_info_updates)}"
                # )

                # logger.trace(
                #     f"[Async] _on_batch_relabeled: Buffer info update complete: "
                #     f"updated={info_updated_count}/{len(buffer_info_updates)} info dicts "
                #     f"(episode_id={episode_id})"
                # )

                if info_updated_count < len(buffer_info_updates):
                    logger.warning(
                        f"[Async] Only updated {info_updated_count}/{len(buffer_info_updates)} "
                        f"info dicts in buffer (episode_id={episode_id}, type={type(episode_id).__name__})"
                    )

            # logger.trace(
            #     f"[Async] _on_batch_relabeled: Buffer update complete: "
            #     f"rewards_updated={len(buffer_updates) if buffer_updates else 0}, "
            #     f"info_updated={len(buffer_info_updates) if buffer_info_updates else 0}, "
            #     f"total_buffer_updates={self.stats['buffer_updates']}, "
            #     f"total_failures={self.stats['buffer_update_failures']}"
            # )

    def set_buffer(self, buffer):
        """Set buffer reference for retroactive reward updates. Can be called after initialization."""
        self.buffer = buffer

    def get_stats(self) -> Dict[str, Any]:
        """Get wrapper statistics."""
        client_stats = self.client.get_stats()
        with self._lock:
            stats_copy = self.stats.copy()  # Copy dict to avoid holding lock during client call
            trajectory_length = len(self._trajectory)
            pending_batches = self._pending_batch_indices
            # Compute average predicted progress reward from all relabeled rewards
            progress_rewards = list(self._relabeled_rewards.values())
            avg_progress_reward = float(np.mean(progress_rewards)) if progress_rewards else 0.0
            num_progress_rewards = len(progress_rewards)
        return {
            **stats_copy,
            **client_stats,
            "trajectory_length": trajectory_length,
            "pending_batches": pending_batches,
            "avg_predicted_progress_reward": avg_progress_reward,
            "num_progress_rewards": num_progress_rewards,
        }

    def __getattr__(self, name):
        """Forward unknown attributes to wrapped environment (e.g., language_instruction)."""
        return getattr(self.env, name)
