import numpy as np
from typing import List, Dict, Any, Union, Optional, TYPE_CHECKING

from robometer_policy_learning.buffers.base_replay_buffer import BaseReplayBuffer, Transition
from robometer_policy_learning.buffers.samplers import BaseSampler


class MixedReplayBuffer(BaseReplayBuffer):
    """
    A mixed replay buffer that combines two separate buffers and samples from them
    with a specified ratio. Useful for offline-to-online RL where you want to sample
    from both offline demonstrations and online experience.

    Args:
        buffer_1: First buffer to sample from
        buffer_2: Second buffer to sample from
        sample_ratio: Ratio for sampling from buffer_1 vs buffer_2 (default 0.5 for 50/50)
        obs_keys: List of keys to include in the observation
        remove_obs_keys: List of keys to remove from the observation
        rename_obs_keys: Dictionary of keys to rename in the observation
        buffer_to_add_to: Buffer to add new experiences to (1 for buffer_1, 2 for buffer_2)
    """

    def __init__(
        self,
        buffer_1: BaseReplayBuffer,
        buffer_2: BaseReplayBuffer,
        sample_ratio: float = 0.5,
        obs_keys: List[str] = None,
        remove_obs_keys: List[str] = None,
        rename_obs_keys: Dict[str, str] = None,
        buffer_to_add_to: int = 1,
        sampler=None,
    ):
        super().__init__(
            obs_keys=obs_keys,
            remove_obs_keys=remove_obs_keys,
            rename_obs_keys=rename_obs_keys,
            sampler=sampler,
        )

        self.buffer_1 = buffer_1
        self.buffer_2 = buffer_2
        self.sample_ratio = sample_ratio
        self.buffer_to_add_to = buffer_to_add_to

        # Validate sample ratio
        if not 0.0 <= sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be between 0.0 and 1.0")

    @property
    def observation_space(self):
        """Return observation space from buffer_1 (assuming both buffers have same obs space)."""
        if hasattr(self.buffer_1, "observation_space"):
            return self.buffer_1.observation_space
        raise NotImplementedError("Buffer 1 does not implement observation_space")

    @property
    def action_space(self):
        """Return action space from buffer_1 (assuming both buffers have same action space)."""
        if hasattr(self.buffer_1, "action_space"):
            return self.buffer_1.action_space
        raise NotImplementedError("Buffer 1 does not implement action_space")

    def _add(self, obs, action, reward, next_obs, done, truncated, **kwargs):
        """This should not be called directly. Use add_to_buffer instead."""
        raise NotImplementedError("Use add_to_buffer method to specify which buffer to add to")

    def add_to_buffer(self, buffer_id: int, obs, action, reward, next_obs, done, truncated, **kwargs):
        """
        Add a transition to a specific buffer.

        Args:
            buffer_id: 1 for buffer_1, 2 for buffer_2
            obs, action, reward, next_obs, done, truncated: Transition components
        """
        if buffer_id == 1:
            self.buffer_1.add(obs, action, reward, next_obs, done, truncated, **kwargs)
        elif buffer_id == 2:
            self.buffer_2.add(obs, action, reward, next_obs, done, truncated, **kwargs)
        else:
            raise ValueError("buffer_id must be 1 or 2")

    def add(self, obs, action, reward, next_obs, done, truncated, **kwargs):
        """
        Add a transition to a specific buffer (defaults to buffer_1).

        Args:
            obs, action, reward, next_obs, done, truncated: Transition components
            buffer_id: 1 for buffer_1, 2 for buffer_2 (default: 1)
        """
        self.add_to_buffer(
            self.buffer_to_add_to,
            obs,
            action,
            reward,
            next_obs,
            done,
            truncated,
            **kwargs,
        )

    def get_all_transitions(self) -> List[Transition]:
        """Get all transitions from both buffers."""
        all_transitions = []
        all_transitions.extend(self.buffer_1.get_all_transitions())
        all_transitions.extend(self.buffer_2.get_all_transitions())
        return all_transitions

    def get_episode_boundaries(self) -> Dict[Any, tuple]:
        """Return episode boundaries from both buffers, with offset for buffer_2."""
        boundaries = {}

        # Get boundaries from buffer_1
        boundaries_1 = self.buffer_1.get_episode_boundaries()
        boundaries.update(boundaries_1)

        # Get boundaries from buffer_2 with offset
        buffer_1_size = len(self.buffer_1)
        boundaries_2 = self.buffer_2.get_episode_boundaries()
        for episode_id, (start, end) in boundaries_2.items():
            # Add offset and ensure unique episode_id
            adjusted_episode_id = f"buffer_2_{episode_id}"
            boundaries[adjusted_episode_id] = (
                start + buffer_1_size,
                end + buffer_1_size,
            )

        return boundaries

    def get_episode_end_transitions(self, count: int) -> List[Transition]:
        """Get episode end transitions from both buffers."""
        episode_ends = []
        episode_ends.extend(self.buffer_1.get_episode_end_transitions(count // 2))
        remaining = count - len(episode_ends)
        if remaining > 0:
            episode_ends.extend(self.buffer_2.get_episode_end_transitions(remaining))
        return episode_ends

    def __len__(self):
        """Return the total size of both buffers."""
        return len(self.buffer_1) + len(self.buffer_2)

    def size(self):
        """Return the total size of both buffers."""
        return len(self.buffer_1) + len(self.buffer_2)

    def is_empty(self):
        """Check if both buffers are empty."""
        return self.buffer_1.is_empty() and self.buffer_2.is_empty()

    def clear(self):
        """Clear both buffers."""
        self.buffer_1.clear()
        self.buffer_2.clear()

    def clear_buffer(self, buffer_id: int):
        """Clear a specific buffer."""
        if buffer_id == 1:
            self.buffer_1.clear()
        elif buffer_id == 2:
            self.buffer_2.clear()
        else:
            raise ValueError("buffer_id must be 1 or 2")

    def get_buffer_sizes(self) -> Dict[str, int]:
        """Get the sizes of both buffers."""
        return {
            "buffer_1_size": len(self.buffer_1),
            "buffer_2_size": len(self.buffer_2),
            "total_size": len(self),
        }

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
        """Sample from both buffers according to sample_ratio."""
        if self.is_empty():
            return {}

        # Check if we're using ChunkedSequentialSampler
        active_sampler = sampler or self.sampler
        is_chunked = hasattr(active_sampler, "chunk_size")

        # Calculate samples from each buffer
        samples_1 = int(batch_size * self.sample_ratio)
        samples_2 = batch_size - samples_1

        # For chunked sampling, check if buffers have enough data
        if is_chunked:
            chunk_size = active_sampler.chunk_size
            # Check if buffer_2 has enough contiguous sequences
            if not self._can_sample_chunks(self.buffer_2, chunk_size, samples_2):
                # Buffer 2 can't form chunks, sample everything from buffer 1
                print(
                    f"Warning: Buffer 2 doesn't have enough contiguous sequences (chunk_size={chunk_size}), sampling full batch from buffer 1"
                )
                batch_1 = self._sample_from_buffer(self.buffer_1, batch_size, active_sampler, device, dtype, **kwargs)
                return self._fix_dtypes(batch_1) if batch_1 else {}
            # Check if buffer_1 has enough contiguous sequences
            if not self._can_sample_chunks(self.buffer_1, chunk_size, samples_1):
                # Buffer 1 can't form chunks, sample everything from buffer 2
                print(
                    f"Warning: Buffer 1 doesn't have enough contiguous sequences (chunk_size={chunk_size}), sampling full batch from buffer 2"
                )
                batch_2 = self._sample_from_buffer(self.buffer_2, batch_size, active_sampler, device, dtype, **kwargs)
                return self._fix_dtypes(batch_2) if batch_2 else {}

        # Sample from each buffer
        batch_1 = self._sample_from_buffer(self.buffer_1, samples_1, active_sampler, device, dtype, **kwargs)
        batch_2 = self._sample_from_buffer(self.buffer_2, samples_2, active_sampler, device, dtype, **kwargs)

        # Fallback if one buffer failed
        if not batch_1 and not batch_2:
            return {}
        elif not batch_1:
            # If buffer_1 failed and we're using chunked sampling, sample full batch from buffer_2
            if is_chunked:
                print(f"Warning: Buffer 1 sampling failed, sampling full batch from buffer 2")
                batch_2 = self._sample_from_buffer(self.buffer_2, batch_size, active_sampler, device, dtype, **kwargs)
                return self._fix_dtypes(batch_2) if batch_2 else {}
            else:
                batch_1 = self._sample_from_buffer(
                    self.buffer_1,
                    batch_size - len(batch_2.get("reward", [])),
                    active_sampler,
                    device,
                    dtype,
                    **kwargs,
                )
        elif not batch_2:
            # If buffer_2 failed and we're using chunked sampling, sample full batch from buffer_1
            if is_chunked:
                print(f"Warning: Buffer 2 sampling failed, sampling full batch from buffer 1")
                batch_1 = self._sample_from_buffer(self.buffer_1, batch_size, active_sampler, device, dtype, **kwargs)
                return self._fix_dtypes(batch_1) if batch_1 else {}
            else:
                batch_2 = self._sample_from_buffer(
                    self.buffer_2,
                    batch_size - len(batch_1.get("reward", [])),
                    active_sampler,
                    device,
                    dtype,
                    **kwargs,
                )

        # Safety check for chunked sampling: ensure both batches have compatible action shapes
        if is_chunked and batch_1 and batch_2:
            if "action" in batch_1 and "action" in batch_2:
                action_1 = batch_1["action"]
                action_2 = batch_2["action"]
                if hasattr(action_1, "ndim") and hasattr(action_2, "ndim"):
                    if action_1.ndim != action_2.ndim:
                        print(f"ERROR: Action dimension mismatch detected!")
                        print(f"  Buffer 1 action shape: {action_1.shape}")
                        print(f"  Buffer 2 action shape: {action_2.shape}")
                        print(f"  Sampling full batch from buffer 1 to avoid mismatch")
                        batch_1 = self._sample_from_buffer(
                            self.buffer_1, batch_size, active_sampler, device, dtype, **kwargs
                        )
                        return self._fix_dtypes(batch_1) if batch_1 else {}

        return self._combine_batches(batch_1, batch_2)

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
            print(f"Warning: Could not check chunk availability: {e}")
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
            print(f"Warning: Buffer sampling failed: {e}")
            return {}

    def add_post_transform(self, transform):
        # Add to both buffers and itself
        self.post_transforms.append(transform)
        self.buffer_1.add_post_transform(transform)
        self.buffer_2.add_post_transform(transform)
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
