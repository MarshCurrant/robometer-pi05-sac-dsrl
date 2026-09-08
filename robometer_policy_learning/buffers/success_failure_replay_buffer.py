"""
Success-Failure Replay Buffer

A replay buffer that automatically routes trajectories to separate success and failure buffers
based on episode outcomes. Useful for analyzing performance, curriculum learning, or sampling
strategies that differentiate between successful and unsuccessful episodes.

The buffer temporarily stores transitions until an episode completes, then routes the entire
episode to either the success_buffer or failure_buffer based on the terminal info.
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional
from collections import defaultdict
from loguru import logger

from robometer_policy_learning.buffers.base_replay_buffer import BaseReplayBuffer, Transition
from robometer_policy_learning.buffers.samplers import BaseSampler


class SuccessFailureReplayBuffer(BaseReplayBuffer):
    """
    A replay buffer that separates successful and unsuccessful trajectories.

    Transitions are temporarily stored until an episode completes. Once an episode ends,
    all its transitions are routed to either success_buffer or failure_buffer based on
    the success indicator in the terminal info.

    Args:
        success_buffer: Buffer to store successful episode transitions
        failure_buffer: Buffer to store unsuccessful episode transitions
        sample_ratio: Ratio for sampling from success_buffer vs failure_buffer (default 0.5 for 50/50)
        obs_keys: List of keys to include in the observation
        remove_obs_keys: List of keys to remove from the observation
        rename_obs_keys: Dictionary of keys to rename in the observation
        sampler: Default sampler for this buffer
    """

    def __init__(
        self,
        success_buffer: BaseReplayBuffer,
        failure_buffer: BaseReplayBuffer,
        sample_ratio: float = 0.5,
        obs_keys: List[str] = None,
        remove_obs_keys: List[str] = None,
        rename_obs_keys: Dict[str, str] = None,
        sampler=None,
    ):
        super().__init__(
            obs_keys=obs_keys,
            remove_obs_keys=remove_obs_keys,
            rename_obs_keys=rename_obs_keys,
            sampler=sampler,
        )

        self.success_buffer = success_buffer
        self.failure_buffer = failure_buffer
        self.sample_ratio = sample_ratio

        # Temporary storage for incomplete episodes
        # Maps episode_id -> list of (obs, action, reward, next_obs, done, truncated, kwargs)
        self._pending_episodes = defaultdict(list)

        # Track episode metadata
        # Maps episode_id -> dict with 'success', 'is_success', or other info keys
        self._episode_info = {}

        # Statistics
        self.stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "failed_episodes": 0,
            "pending_episodes": 0,
        }

        # Validate sample ratio
        if not 0.0 <= sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be between 0.0 and 1.0")

    @property
    def observation_space(self):
        """Return observation space from success_buffer (assuming both buffers have same obs space)."""
        if hasattr(self.success_buffer, "observation_space"):
            return self.success_buffer.observation_space
        raise NotImplementedError("Success buffer does not implement observation_space")

    @property
    def action_space(self):
        """Return action space from success_buffer (assuming both buffers have same action space)."""
        if hasattr(self.success_buffer, "action_space"):
            return self.success_buffer.action_space
        raise NotImplementedError("Success buffer does not implement action_space")

    def _add(self, obs, action, reward, next_obs, done, truncated, **kwargs):
        """
        Add a transition to temporary storage until episode completes.

        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode is done
            truncated: Whether episode was truncated
            **kwargs: Additional data (must include episode_id)
        """
        episode_id = kwargs.get("episode_id", None)
        if episode_id is None:
            raise ValueError("episode_id must be provided in kwargs for SuccessFailureReplayBuffer")

        # Store transition in pending episodes
        self._pending_episodes[episode_id].append(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "truncated": truncated,
                "kwargs": kwargs.copy(),
            }
        )

        # If episode is done, check for success info and route to appropriate buffer
        if done or truncated:
            # Extract success information from kwargs or stored info
            success = self._determine_success(episode_id, kwargs)
            self._finalize_episode(episode_id, success)

    def add_episode_info(self, episode_id: Any, info: Dict[str, Any]):
        """
        Provide episode-level info (e.g., from environment's terminal info dict).

        This is useful if success information arrives separately from transitions.
        Call this before or when the episode completes.

        Args:
            episode_id: Episode identifier
            info: Info dict containing success indicators (e.g., 'is_success', 'success')
        """
        if episode_id not in self._episode_info:
            self._episode_info[episode_id] = {}
        self._episode_info[episode_id].update(info)

    def _determine_success(self, episode_id: Any, terminal_kwargs: Dict[str, Any]) -> bool:
        """
        Determine if an episode was successful.

        Checks multiple sources for success information:
        1. 'is_success' or 'success' in terminal transition kwargs
        2. Stored episode info from add_episode_info()

        Args:
            episode_id: Episode identifier
            terminal_kwargs: kwargs from the terminal transition

        Returns:
            True if episode was successful, False otherwise
        """
        # Check terminal kwargs first
        if "is_success" in terminal_kwargs:
            return bool(terminal_kwargs["is_success"])
        if "success" in terminal_kwargs:
            return bool(terminal_kwargs["success"])

        # Check stored episode info
        if episode_id in self._episode_info:
            info = self._episode_info[episode_id]
            if "is_success" in info:
                return bool(info["is_success"])
            if "success" in info:
                return bool(info["success"])

        # Default to failure if no success info found
        logger.debug(f"No success info found for episode {episode_id}, marking as failure")
        return False

    def _finalize_episode(self, episode_id: Any, success: bool):
        """
        Move ALL transitions from a completed episode to the appropriate buffer.

        This method routes ENTIRE EPISODES (all transitions) to either success_buffer
        or failure_buffer based on the episode outcome.

        Args:
            episode_id: Episode identifier
            success: Whether the episode was successful
        """
        if episode_id not in self._pending_episodes:
            logger.warning(f"Tried to finalize unknown episode {episode_id}")
            return

        transitions = self._pending_episodes[episode_id]
        episode_length = len(transitions)
        target_buffer = self.success_buffer if success else self.failure_buffer
        buffer_name = "SUCCESS" if success else "FAILURE"

        # Add ALL transitions from this episode to the target buffer
        for transition_data in transitions:
            target_buffer.add(
                obs=transition_data["obs"],
                action=transition_data["action"],
                reward=transition_data["reward"],
                next_obs=transition_data["next_obs"],
                done=transition_data["done"],
                truncated=transition_data["truncated"],
                **transition_data["kwargs"],
            )

        # Update statistics
        self.stats["total_episodes"] += 1
        if success:
            self.stats["successful_episodes"] += 1
        else:
            self.stats["failed_episodes"] += 1

        # Log episode routing for verification
        logger.debug(
            f"[SuccessFailureBuffer] Episode {episode_id} ({episode_length} transitions) → {buffer_name} buffer"
        )

        # Clean up
        del self._pending_episodes[episode_id]
        if episode_id in self._episode_info:
            del self._episode_info[episode_id]

        self.stats["pending_episodes"] = len(self._pending_episodes)

        # Log periodically
        if self.stats["total_episodes"] % 100 == 0:
            self._log_statistics()

    def _log_statistics(self):
        """Log buffer statistics."""
        total = self.stats["total_episodes"]
        if total == 0:
            return

        success_rate = self.stats["successful_episodes"] / total * 100
        logger.info(
            f"[SuccessFailureReplayBuffer] Episodes: {total} | "
            f"Success: {self.stats['successful_episodes']} ({success_rate:.1f}%) | "
            f"Failure: {self.stats['failed_episodes']} | "
            f"Pending: {self.stats['pending_episodes']} | "
            f"Success buffer: {len(self.success_buffer)} | "
            f"Failure buffer: {len(self.failure_buffer)}"
        )

    def get_all_transitions(self) -> List[Transition]:
        """Get all transitions from both buffers."""
        all_transitions = []
        all_transitions.extend(self.success_buffer.get_all_transitions())
        all_transitions.extend(self.failure_buffer.get_all_transitions())
        return all_transitions

    def get_episode_boundaries(self) -> Dict[Any, tuple]:
        """Return episode boundaries from both buffers, with offset for failure_buffer."""
        boundaries = {}

        # Get boundaries from success_buffer
        boundaries_success = self.success_buffer.get_episode_boundaries()
        for episode_id, (start, end) in boundaries_success.items():
            boundaries[f"success_{episode_id}"] = (start, end)

        # Get boundaries from failure_buffer with offset
        success_buffer_size = len(self.success_buffer)
        boundaries_failure = self.failure_buffer.get_episode_boundaries()
        for episode_id, (start, end) in boundaries_failure.items():
            boundaries[f"failure_{episode_id}"] = (
                start + success_buffer_size,
                end + success_buffer_size,
            )

        return boundaries

    def get_episode_end_transitions(self, count: int) -> List[Transition]:
        """Get episode end transitions from both buffers."""
        episode_ends = []

        # Sample proportionally from both buffers
        success_count = int(count * self.sample_ratio)
        failure_count = count - success_count

        episode_ends.extend(self.success_buffer.get_episode_end_transitions(success_count))
        episode_ends.extend(self.failure_buffer.get_episode_end_transitions(failure_count))

        return episode_ends

    def __len__(self):
        """Return the total size of both buffers (excludes pending transitions)."""
        return len(self.success_buffer) + len(self.failure_buffer)

    def size(self):
        """Return the total size of both buffers (excludes pending transitions)."""
        return len(self.success_buffer) + len(self.failure_buffer)

    def is_empty(self):
        """Check if both buffers are empty."""
        return self.success_buffer.is_empty() and self.failure_buffer.is_empty()

    def clear(self):
        """Clear both buffers and pending episodes."""
        self.success_buffer.clear()
        self.failure_buffer.clear()
        self._pending_episodes.clear()
        self._episode_info.clear()
        self.stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "failed_episodes": 0,
            "pending_episodes": 0,
        }

    def clear_buffer(self, buffer_name: str):
        """
        Clear a specific buffer.

        Args:
            buffer_name: 'success', 'failure', or 'pending'
        """
        if buffer_name == "success":
            self.success_buffer.clear()
        elif buffer_name == "failure":
            self.failure_buffer.clear()
        elif buffer_name == "pending":
            self._pending_episodes.clear()
            self._episode_info.clear()
            self.stats["pending_episodes"] = 0
        else:
            raise ValueError("buffer_name must be 'success', 'failure', or 'pending'")

    def get_buffer_sizes(self) -> Dict[str, int]:
        """Get the sizes of both buffers and pending episodes."""
        return {
            "success_buffer_size": len(self.success_buffer),
            "failure_buffer_size": len(self.failure_buffer),
            "pending_transitions": sum(len(ep) for ep in self._pending_episodes.values()),
            "pending_episodes": len(self._pending_episodes),
            "total_size": len(self),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the buffer."""
        stats = self.stats.copy()
        stats.update(self.get_buffer_sizes())

        if stats["total_episodes"] > 0:
            stats["success_rate"] = stats["successful_episodes"] / stats["total_episodes"]
        else:
            stats["success_rate"] = 0.0

        return stats

    def verify_episode_integrity(self) -> Dict[str, Any]:
        """
        Verify that entire episodes are correctly stored in buffers.

        This method checks that:
        1. Each episode has all its transitions in one buffer
        2. Episodes are not split across buffers
        3. Episode IDs are consistent

        Returns:
            Dictionary with verification results and episode counts per buffer
        """
        verification = {
            "success_episodes": {},  # episode_id -> count of transitions
            "failure_episodes": {},  # episode_id -> count of transitions
            "total_success_transitions": 0,
            "total_failure_transitions": 0,
            "issues": [],
        }

        # Check success buffer
        if hasattr(self.success_buffer, "get_all_transitions"):
            success_transitions = self.success_buffer.get_all_transitions()
            for trans in success_transitions:
                ep_id = trans.episode_id
                verification["success_episodes"][ep_id] = verification["success_episodes"].get(ep_id, 0) + 1
            verification["total_success_transitions"] = len(success_transitions)

        # Check failure buffer
        if hasattr(self.failure_buffer, "get_all_transitions"):
            failure_transitions = self.failure_buffer.get_all_transitions()
            for trans in failure_transitions:
                ep_id = trans.episode_id
                verification["failure_episodes"][ep_id] = verification["failure_episodes"].get(ep_id, 0) + 1
            verification["total_failure_transitions"] = len(failure_transitions)

        # Check for episodes split across buffers (should never happen)
        success_ids = set(verification["success_episodes"].keys())
        failure_ids = set(verification["failure_episodes"].keys())
        split_episodes = success_ids & failure_ids

        if split_episodes:
            verification["issues"].append(
                f"ERROR: {len(split_episodes)} episodes split across buffers: {split_episodes}"
            )

        # Summary
        verification["num_success_episodes"] = len(verification["success_episodes"])
        verification["num_failure_episodes"] = len(verification["failure_episodes"])
        verification["is_valid"] = len(verification["issues"]) == 0

        return verification

    def set_sample_ratio(self, ratio: float):
        """Update the sampling ratio."""
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("sample_ratio must be between 0.0 and 1.0")
        self.sample_ratio = ratio

    def sample(
        self,
        batch_size: int,
        sampler: "BaseSampler" = None,
        device: str = None,
        dtype=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sample from both buffers according to sample_ratio.

        This uses the same logic as MixedReplayBuffer for combining batches.
        """
        if self.is_empty():
            return {}

        # Check if we're using ChunkedSequentialSampler
        active_sampler = sampler or self.sampler
        is_chunked = hasattr(active_sampler, "chunk_size")

        # Calculate samples from each buffer
        samples_success = int(batch_size * self.sample_ratio)
        samples_failure = batch_size - samples_success

        # For chunked sampling, check if buffers have enough data
        if is_chunked:
            chunk_size = active_sampler.chunk_size
            # Check if failure_buffer has enough contiguous sequences
            if not self._can_sample_chunks(self.failure_buffer, chunk_size, samples_failure):
                logger.debug(
                    f"Failure buffer doesn't have enough contiguous sequences (chunk_size={chunk_size}), "
                    f"sampling full batch from success buffer"
                )
                batch = self._sample_from_buffer(
                    self.success_buffer, batch_size, active_sampler, device, dtype, **kwargs
                )
                return self._fix_dtypes(batch) if batch else {}

            # Check if success_buffer has enough contiguous sequences
            if not self._can_sample_chunks(self.success_buffer, chunk_size, samples_success):
                logger.debug(
                    f"Success buffer doesn't have enough contiguous sequences (chunk_size={chunk_size}), "
                    f"sampling full batch from failure buffer"
                )
                batch = self._sample_from_buffer(
                    self.failure_buffer, batch_size, active_sampler, device, dtype, **kwargs
                )
                return self._fix_dtypes(batch) if batch else {}

        # Sample from each buffer
        batch_success = self._sample_from_buffer(
            self.success_buffer, samples_success, active_sampler, device, dtype, **kwargs
        )
        batch_failure = self._sample_from_buffer(
            self.failure_buffer, samples_failure, active_sampler, device, dtype, **kwargs
        )

        # Fallback if one buffer failed
        if not batch_success and not batch_failure:
            return {}
        elif not batch_success:
            if is_chunked:
                logger.debug("Success buffer sampling failed, sampling full batch from failure buffer")
                batch_failure = self._sample_from_buffer(
                    self.failure_buffer, batch_size, active_sampler, device, dtype, **kwargs
                )
                return self._fix_dtypes(batch_failure) if batch_failure else {}
            else:
                # Compensate with more from failure buffer
                remaining = batch_size - len(batch_failure.get("reward", []))
                if remaining > 0:
                    extra = self._sample_from_buffer(
                        self.failure_buffer, remaining, active_sampler, device, dtype, **kwargs
                    )
                    batch_success = extra
        elif not batch_failure:
            if is_chunked:
                logger.debug("Failure buffer sampling failed, sampling full batch from success buffer")
                batch_success = self._sample_from_buffer(
                    self.success_buffer, batch_size, active_sampler, device, dtype, **kwargs
                )
                return self._fix_dtypes(batch_success) if batch_success else {}
            else:
                # Compensate with more from success buffer
                remaining = batch_size - len(batch_success.get("reward", []))
                if remaining > 0:
                    extra = self._sample_from_buffer(
                        self.success_buffer, remaining, active_sampler, device, dtype, **kwargs
                    )
                    batch_failure = extra

        # Safety check for chunked sampling: ensure both batches have compatible action shapes
        if is_chunked and batch_success and batch_failure:
            if "action" in batch_success and "action" in batch_failure:
                action_success = batch_success["action"]
                action_failure = batch_failure["action"]
                if hasattr(action_success, "ndim") and hasattr(action_failure, "ndim"):
                    if action_success.ndim != action_failure.ndim:
                        logger.error(
                            f"Action dimension mismatch detected! "
                            f"Success action shape: {action_success.shape}, "
                            f"Failure action shape: {action_failure.shape}. "
                            f"Sampling full batch from success buffer to avoid mismatch"
                        )
                        batch_success = self._sample_from_buffer(
                            self.success_buffer, batch_size, active_sampler, device, dtype, **kwargs
                        )
                        return self._fix_dtypes(batch_success) if batch_success else {}

        return self._combine_batches(batch_success, batch_failure)

    def _can_sample_chunks(self, buffer, chunk_size: int, num_chunks: int) -> bool:
        """
        Check if a buffer has enough contiguous sequences to sample the requested number of chunks.

        Args:
            buffer: The buffer to check
            chunk_size: Size of each chunk
            num_chunks: Number of chunks needed

        Returns:
            True if the buffer can provide enough chunks, False otherwise
        """
        if num_chunks <= 0 or buffer.is_empty():
            return True  # No chunks needed or buffer is empty (handled elsewhere)

        # Get episode boundaries
        try:
            boundaries = buffer.get_episode_boundaries()
            if not boundaries:
                return False

            # Count how many valid chunk starts we can get
            valid_chunk_count = 0
            for start, end in boundaries.values():
                episode_length = end - start + 1
                if episode_length >= chunk_size:
                    # Each episode can provide (episode_length - chunk_size + 1) chunks
                    valid_chunk_count += episode_length - chunk_size + 1

            return valid_chunk_count >= num_chunks
        except Exception as e:
            logger.warning(f"Could not check chunk availability: {e}")
            return False

    def _sample_from_buffer(self, buffer, batch_size, sampler, device, dtype, **kwargs):
        """Sample from a single buffer with error handling."""
        if batch_size <= 0 or buffer.is_empty():
            return {}
        try:
            return buffer.sample(
                batch_size=batch_size,
                sampler=sampler or buffer.sampler,
                device=device,
                dtype=dtype,
                **kwargs,
            )
        except Exception as e:
            logger.warning(f"Buffer sampling failed: {e}")
            return {}

    def add_post_transform(self, transform):
        """Add post-transform to both buffers and itself."""
        self.post_transforms.append(transform)
        self.success_buffer.add_post_transform(transform)
        self.failure_buffer.add_post_transform(transform)
        return self

    def _combine_batches(self, batch_1: Dict[str, Any], batch_2: Dict[str, Any]) -> Dict[str, Any]:
        """Combine two batched samples into one."""
        if not batch_1:
            return self._fix_dtypes(batch_2)
        if not batch_2:
            return self._fix_dtypes(batch_1)

        combined = {}
        for key in set(batch_1.keys()) | set(batch_2.keys()):
            val_1, val_2 = batch_1.get(key), batch_2.get(key)

            if val_1 is None:
                combined[key] = val_2
            elif val_2 is None:
                combined[key] = val_1
            elif key in ["obs", "next_obs"]:
                combined[key] = self._combine_dicts(val_1, val_2)
            else:
                combined[key] = self._concat(val_1, val_2)

        return self._fix_dtypes(combined)

    def _combine_dicts(self, dict_1: Dict[str, Any], dict_2: Dict[str, Any]) -> Dict[str, Any]:
        """Combine observation dictionaries."""
        combined = {}
        for key in set(dict_1.keys()) | set(dict_2.keys()):
            val_1, val_2 = dict_1.get(key), dict_2.get(key)

            if val_1 is None:
                combined[key] = val_2
            elif val_2 is None:
                combined[key] = val_1
            else:
                combined[key] = self._concat(val_1, val_2)
        return combined

    def _concat(self, arr_1: Any, arr_2: Any) -> Any:
        """Concatenate arrays/tensors, converting bools to float."""
        import torch
        import numpy as np

        # Convert to same type and concatenate
        if isinstance(arr_1, torch.Tensor) and isinstance(arr_2, torch.Tensor):
            if arr_1.dtype == torch.bool:
                arr_1 = arr_1.float()
            if arr_2.dtype == torch.bool:
                arr_2 = arr_2.float()

            # Check for dimension mismatch and raise informative error
            if arr_1.ndim != arr_2.ndim:
                raise ValueError(
                    f"Cannot concatenate tensors with different dimensions. "
                    f"arr_1.shape={arr_1.shape}, arr_2.shape={arr_2.shape}. "
                    f"This likely indicates a bug in buffer sampling logic."
                )

            return torch.cat([arr_1, arr_2], dim=0)
        elif isinstance(arr_1, np.ndarray) and isinstance(arr_2, np.ndarray):
            if arr_1.dtype == np.bool_:
                arr_1 = arr_1.astype(np.float32)
            if arr_2.dtype == np.bool_:
                arr_2 = arr_2.astype(np.float32)

            # Check for dimension mismatch and raise informative error
            if arr_1.ndim != arr_2.ndim:
                raise ValueError(
                    f"Cannot concatenate arrays with different dimensions. "
                    f"arr_1.shape={arr_1.shape}, arr_2.shape={arr_2.shape}. "
                    f"This likely indicates a bug in buffer sampling logic."
                )

            return np.concatenate([arr_1, arr_2], axis=0)
        elif isinstance(arr_1, list) and isinstance(arr_2, list):
            return arr_1 + arr_2
        else:
            # Convert mixed types to numpy
            if isinstance(arr_1, torch.Tensor):
                arr_1 = arr_1.cpu().numpy()
            if isinstance(arr_2, torch.Tensor):
                arr_2 = arr_2.cpu().numpy()
            arr_1, arr_2 = np.asarray(arr_1), np.asarray(arr_2)
            if arr_1.dtype == np.bool_:
                arr_1 = arr_1.astype(np.float32)
            if arr_2.dtype == np.bool_:
                arr_2 = arr_2.astype(np.float32)
            return np.concatenate([arr_1, arr_2], axis=0)

    def _fix_dtypes(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Fix dtypes and apply base buffer logic."""
        if not batch or "done" not in batch or "truncated" not in batch:
            return batch

        import torch
        import numpy as np

        done, truncated = batch["done"], batch["truncated"]

        # Convert bools to float
        if isinstance(done, torch.Tensor) and done.dtype == torch.bool:
            done = done.float()
        elif isinstance(done, np.ndarray) and done.dtype == np.bool_:
            done = done.astype(np.float32)

        if isinstance(truncated, torch.Tensor) and truncated.dtype == torch.bool:
            truncated = truncated.float()
        elif isinstance(truncated, np.ndarray) and truncated.dtype == np.bool_:
            truncated = truncated.astype(np.float32)

        # Apply base buffer logic: done = done * (1 - truncated)
        batch["done"] = done * (1 - truncated)
        batch["truncated"] = truncated

        return batch
