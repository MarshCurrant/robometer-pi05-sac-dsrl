import abc
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Tuple, TYPE_CHECKING
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback
from robometer_policy_learning.utils.gpu_utils import move_to_device  # NOTE: Use of this function in BaseReplayBuffer might be problematic
from robometer_policy_learning.utils.transition_utilities import (
    batch_transitions,
    unbatch_transitions,
    apply_transforms_batched,
)
from loguru import logger 

if TYPE_CHECKING:
    from robometer_policy_learning.buffers.samplers import BaseSampler
    # from reward_models import BaseRewardModel


@dataclass
class Transition:
    """Represents a single transition in a replay buffer."""

    obs: Dict[str, Any]
    action: Any
    reward: float
    next_obs: Dict[str, Any]
    done: bool
    truncated: bool = False
    episode_id: Any = None
    step_in_episode: int = 0
    max_steps_in_episode: int = 0
    timestamp: Optional[float] = None
    language_instruction: Optional[str] = None  # Raw language instruction
    info: Optional[Dict[str, Any]] = None  # Info dict for retroactive updates (e.g., relabeled rewards)

    def replace(self, **kwargs):
        """Create a copy of this transition with some fields replaced."""
        # Get all current field values
        current_values = {
            "obs": self.obs,
            "action": self.action,
            "reward": self.reward,
            "next_obs": self.next_obs,
            "done": self.done,
            "truncated": self.truncated,
            "episode_id": self.episode_id,
            "step_in_episode": self.step_in_episode,
            "max_steps_in_episode": self.max_steps_in_episode,
            "timestamp": self.timestamp,
            "language_instruction": self.language_instruction,
            "info": self.info,
        }
        # Update with provided kwargs
        current_values.update(kwargs)
        return Transition(**current_values)


class BackgroundSampler:
    """
    Background sampler that pre-caches batches from a replay buffer in separate threads.
    Handles queue full conditions gracefully during online training transitions.
    """

    def __init__(
        self,
        buffer,
        sampler,
        batch_size: int,
        cache_size: int = 10,
        num_workers: int = 2,
    ):
        self.buffer = buffer
        self.sampler = sampler
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.num_workers = num_workers

        # Thread-safe queue for caching batches
        self.sample_queue = queue.Queue(maxsize=cache_size)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.running = False
        self._lock = threading.Lock()

        # Statistics for monitoring
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "queue_full_events": 0,  # Track queue full events
            "background_errors": 0,
        }

    def start(self):
        """Start background sampling."""
        with self._lock:
            if self.running:
                return
            self.running = True

        # Pre-fill the cache
        for _ in range(min(self.num_workers, self.cache_size)):
            self.executor.submit(self._background_sample)

    def stop(self):
        """Stop background sampling and cleanup."""
        with self._lock:
            if not self.running:
                return
            self.running = False

        self.executor.shutdown(wait=True)

        # Clear the queue
        while not self.sample_queue.empty():
            try:
                self.sample_queue.get_nowait()
            except queue.Empty:
                break

    def _background_sample(self):
        """Background sampling worker function with improved queue handling."""
        try:
            # Always delegate to the configured sampler to respect its semantics
            sampled_transitions = self.sampler.sample(self.buffer, self.batch_size)
            if not sampled_transitions:
                # Nothing to process; schedule another attempt
                if self.running:
                    self.executor.submit(self._background_sample)
                return

            # Apply transforms (prefer batched transforms for efficiency)
            if self.buffer.post_transforms:
                _n_transitions = len(sampled_transitions)
                _t0 = time.perf_counter()
                _batched = batch_transitions(sampled_transitions)
                _batched_out = apply_transforms_batched(_batched, self.buffer.post_transforms)
                sampled_transitions = unbatch_transitions(_batched_out)
                _dt = time.perf_counter() - _t0

            # Batch the transitions
            batched = self.buffer._batch_transitions(sampled_transitions)
            if not batched:
                # Skip enqueuing empty batches
                if self.running:
                    self.executor.submit(self._background_sample)
                return

            # Try to put in queue with graceful handling of full queue
            if self.running:
                try:
                    self.sample_queue.put(batched, timeout=0.1)  # Shorter timeout

                    # Submit another sampling task to maintain cache
                    if self.running:
                        self.executor.submit(self._background_sample)

                except queue.Full:
                    # Queue is full - this is normal during online training transitions
                    # Don't print errors, just track statistics and retry later
                    self.stats["queue_full_events"] += 1

                    # Brief pause before retrying to avoid busy waiting
                    if self.running:
                        # Schedule retry with a small delay
                        def delayed_retry():
                            time.sleep(0.5)  # Wait 500ms before retrying
                            if self.running:
                                self.executor.submit(self._background_sample)

                        self.executor.submit(delayed_retry)

        except Exception as e:
            # Only print errors for actual failures (not queue full)
            self.stats["background_errors"] += 1
            print(f"Background sampling error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Continue sampling even if one batch fails
            if self.running:
                self.executor.submit(self._background_sample)

    def get_batch(self, device: str = None, dtype=None) -> Dict[str, Any]:
        """
        Get a pre-cached batch. Falls back to synchronous sampling if cache is empty.
        """
        try:
            # Try to get from cache first
            batch = self.sample_queue.get_nowait()
            self.stats["cache_hits"] += 1

            # Move to device/dtype if needed
            if device is not None or dtype is not None:
                # If already tensor batch, just move
                try:
                    import torch

                    def _move(v):
                        if isinstance(v, torch.Tensor):
                            out = v
                            if dtype is not None:
                                out = out.to(dtype)
                            if device is not None:
                                out = out.to(device, non_blocking=True)
                            return out
                        return v

                    if isinstance(batch.get("obs"), dict) and any(
                        isinstance(v, torch.Tensor) for v in batch["obs"].values()
                    ):
                        batch["obs"] = {k: _move(v) for k, v in batch["obs"].items()}
                        batch["next_obs"] = {k: _move(v) for k, v in batch["next_obs"].items()}
                        batch["action"] = _move(batch["action"]) if "action" in batch else batch.get("action")
                        for k in ("reward", "done", "truncated"):
                            if k in batch:
                                batch[k] = _move(batch[k])
                    else:
                        # Fallback to standard conversion
                        batch = self.buffer._convert_batch_to_tensors(batch, device, dtype)
                except Exception:
                    batch = self.buffer._convert_batch_to_tensors(batch, device, dtype)

            return batch

        except queue.Empty:
            # Cache miss - fall back to synchronous sampling
            self.stats["cache_misses"] += 1
            cache_misses = self.stats["cache_misses"]
            total_requests = self.stats["cache_hits"] + cache_misses
            hit_rate = (self.stats["cache_hits"] / total_requests) * 100 if total_requests > 0 else 0

            # print(
            #     f"⚠️  Background cache miss #{cache_misses} (hits: {self.stats['cache_hits']}, hit_rate: {hit_rate:.2f}%)"
            # )

            # Fall back to synchronous sampling
            return self.buffer.sample(self.batch_size, device=device, dtype=dtype)

    def get_stats(self) -> Dict[str, Any]:
        """Get sampling statistics."""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = (self.stats["cache_hits"] / total_requests) if total_requests > 0 else 0

        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,  # Keep original key name for compatibility
            "queue_size": self.sample_queue.qsize(),  # Current queue size
            "running": self.running,  # Whether background sampling is active
        }


import os

class BaseReplayBuffer(abc.ABC):
    """
    Base class for replay buffers.

    Args:
        obs_keys: List of keys to include in the observation.
        remove_obs_keys: List of keys to remove from the observation.
        rename_obs_keys: Dictionary of keys to rename in the observation.
        pre_transforms: List of Callables to transform the observation before adding to the buffer.
        post_transforms: List of Callables to transform the observation when sampling from the buffer.
        sampler: Sampling strategy to use for this buffer.
    """

    def __init__(
        self,
        obs_keys: List[str] = None,
        remove_obs_keys: List[str] = None,
        rename_obs_keys: Dict[str, str] = None,
        pre_transforms: List[Callable] = None,
        post_transforms: List[Callable] = None,  # Can handle both batch and transition transforms
        sampler: "BaseSampler" = None,
        reward_model=None,
    ):
        self.obs_keys = obs_keys
        self.remove_obs_keys = remove_obs_keys or []
        self.rename_obs_keys = rename_obs_keys or {}
        self.pre_transforms = pre_transforms or []
        self.post_transforms = post_transforms or []
        # Import here to avoid circular imports
        from robometer_policy_learning.buffers.samplers import RandomSampler

        self.sampler = sampler or RandomSampler()

        if reward_model is not None:
            if reward_model.when_to_apply == "pre":
                self.pre_transforms.append(reward_model)
            elif reward_model.when_to_apply == "post":
                self.post_transforms.append(reward_model)
            else:
                raise ValueError(f"Invalid apply_to_buffer value: {reward_model.apply_to_buffer}")

    @property
    def observation_space(self):
        raise NotImplementedError("Observation space not implemented. You must implement this in the subclass.")

    @property
    def action_space(self):
        raise NotImplementedError("Action space not implemented. You must implement this in the subclass.")

    @abc.abstractmethod
    def _add(self, obs, action, reward, next_obs, done, truncated, **kwargs):
        raise NotImplementedError

    def _sample(self, batch_size: int, **kwargs) -> List[Transition]:
        """Delegate to the sampling strategy"""
        return self.sampler.sample(self, batch_size, **kwargs)

    @abc.abstractmethod
    def get_all_transitions(self) -> List[Transition]:
        """
        Return all transitions in the buffer.

        Note: Implementations should return direct references (not copies)
        for efficiency, since samplers only need read access.
        """
        pass

    def set_sampler(self, sampler: "BaseSampler"):
        """Allow runtime sampler switching"""
        self.sampler = sampler

    def add_post_transform(self, transform):
        self.post_transforms.append(transform)
        return self

    def get_episode_boundaries(self) -> Dict[Any, Tuple[int, int]]:
        """Return episode_id -> (start_idx, end_idx) mapping"""
        all_transitions = self.get_all_transitions()
        boundaries = {}
        current_episode = None
        start_idx = 0

        for i, transition in enumerate(all_transitions):
            if transition.episode_id != current_episode:
                if current_episode is not None:
                    boundaries[current_episode] = (start_idx, i - 1)
                current_episode = transition.episode_id
                start_idx = i

        # Handle last episode
        if current_episode is not None:
            boundaries[current_episode] = (start_idx, len(all_transitions) - 1)

        return boundaries

    def get_episode_end_transitions(self, count: int) -> List[Transition]:
        """Return transitions that end episodes"""
        all_transitions = self.get_all_transitions()
        episode_ends = [t for t in all_transitions if t.done or t.truncated]
        return random.sample(episode_ends, min(count, len(episode_ends))) if episode_ends else []

    def get_transitions_by_episode(self, episode_id: Any) -> List[Transition]:
        """Return all transitions from a specific episode"""
        all_transitions = self.get_all_transitions()
        return [t for t in all_transitions if t.episode_id == episode_id]

    def get_relabeled_transitions(self) -> List[Transition]:
        """
        Return only transitions that have been relabeled (i.e., have relabeled_reward in info).
        Useful for async reward relabeling to ensure we only sample transitions with valid rewards.
        
        Returns:
            List of transitions that have been relabeled
        """
        all_transitions = self.get_all_transitions()
        relabeled_transitions = []
        
        for transition in all_transitions:
            if transition.info is not None and "relabeled_reward" in transition.info:
                relabeled_transitions.append(transition)
        
        return relabeled_transitions
    
    def get_relabeled_count(self) -> int:
        """
        Return the count of transitions that have been relabeled.
        More efficient than len(get_relabeled_transitions()) as it doesn't build the list.
        
        Returns:
            Number of transitions with relabeled rewards
        """
        all_transitions = self.get_all_transitions()
        count = 0
        
        for transition in all_transitions:
            if transition.info is not None and "relabeled_reward" in transition.info:
                count += 1
        
        return count

    def update_reward(self, episode_id: Any, step_in_episode: int, new_reward: float) -> bool:
        """
        Update the reward for a specific transition by episode_id and step_in_episode.
        Used for retroactive reward updates (e.g., async reward relabeling).

        Returns True if transition was found and updated, False otherwise.
        """
        # logger.debug(
        #     f"[BaseReplayBuffer.update_reward] Called with episode_id={episode_id} (type={type(episode_id).__name__}), "
        #     f"step_in_episode={step_in_episode} (type={type(step_in_episode).__name__}), new_reward={new_reward}"
        # )

        # # Use index if available (for subclasses like ReplayBuffer that maintain _transition_index)
        # if hasattr(self, "_transition_index"):
        #     key = (episode_id, step_in_episode)
        #     logger.debug(
        #         f"[BaseReplayBuffer.update_reward] Using index, key={key} (type={type(key[0]).__name__}, {type(key[1]).__name__})"
        #     )
        #     logger.debug(f"[BaseReplayBuffer.update_reward] Index size: {len(self._transition_index)}")
        #     if key in self._transition_index:
        #         idx = self._transition_index[key]
        #         all_transitions = self.get_all_transitions()
        #         logger.debug(
        #             f"[BaseReplayBuffer.update_reward] Found key in index, idx={idx}, buffer_size={len(all_transitions)}"
        #         )
        #         if 0 <= idx < len(all_transitions):
        #             t = all_transitions[idx]
        #             logger.debug(
        #                 f"[BaseReplayBuffer.update_reward] Transition at idx={idx}: episode_id={t.episode_id} (type={type(t.episode_id).__name__}), "
        #                 f"step_in_episode={t.step_in_episode} (type={type(t.step_in_episode).__name__})"
        #             )
        #             if t.episode_id == episode_id and t.step_in_episode == step_in_episode:
        #                 t.reward = new_reward
        #                 logger.debug(f"[BaseReplayBuffer.update_reward] Successfully updated reward at idx={idx}")
        #                 return True
        #             else:
        #                 logger.warning(
        #                     f"[BaseReplayBuffer.update_reward] Mismatch at idx={idx}: "
        #                     f"expected episode_id={episode_id} (type={type(episode_id).__name__}), got {t.episode_id} (type={type(t.episode_id).__name__}); "
        #                     f"expected step_in_episode={step_in_episode} (type={type(step_in_episode).__name__}), got {t.step_in_episode} (type={type(t.step_in_episode).__name__})"
        #                 )
        #             # Index stale, remove it
        #             self._transition_index.pop(key, None)
        #         else:
        #             logger.warning(
        #                 f"[BaseReplayBuffer.update_reward] Index idx={idx} out of range [0, {len(all_transitions)})"
        #             )
        #     else:
        #         logger.debug(
        #             f"[BaseReplayBuffer.update_reward] Key {key} not in index. Sample keys: {list(self._transition_index.keys())[:5]}"
        #         )

        # Fallback to linear search
        all_transitions = self.get_all_transitions()
        # logger.debug(
        #     f"[BaseReplayBuffer.update_reward] Falling back to linear search over {len(all_transitions)} transitions"
        # )
        matches_found = 0
        for i, transition in enumerate(all_transitions):
            if transition.episode_id == episode_id and transition.step_in_episode == step_in_episode:
                transition.reward = new_reward
                # logger.debug(
                #     f"[BaseReplayBuffer.update_reward] Successfully updated reward at idx={i} (via linear search)"
                # )
                matches_found += 1
                return True
        # logger.warning(
        #     f"[BaseReplayBuffer.update_reward] No match found after linear search. "
        #     f"Looking for episode_id={episode_id} (type={type(episode_id).__name__}), step_in_episode={step_in_episode} (type={type(step_in_episode).__name__}). "
        #     f"Sample transitions: {[(t.episode_id, t.step_in_episode, type(t.episode_id).__name__, type(t.step_in_episode).__name__) for t in all_transitions[:5]]}"
        # )
        return False

    def update_rewards_batch(self, updates: List[Tuple[Any, int, float]]) -> int:
        """Batch update rewards. updates: List of (episode_id, step_in_episode, new_reward) tuples."""
        # logger.debug(f"[BaseReplayBuffer.update_rewards_batch] Called with {len(updates)} updates")
        if updates:
            sample_update = updates[0]
            # logger.debug(
            #     f"[BaseReplayBuffer.update_rewards_batch] Sample update: episode_id={sample_update[0]} (type={type(sample_update[0]).__name__}), "
            #     f"step_in_episode={sample_update[1]} (type={type(sample_update[1]).__name__}), new_reward={sample_update[2]}"
            # )
            all_unique_episodes = set(ep_id for ep_id, _, _ in updates)
            # logger.debug(f"[BaseReplayBuffer.update_rewards_batch] Unique episode_ids in batch: {all_unique_episodes}")
            step_ranges = {
                ep_id: (min(steps), max(steps))
                for ep_id, steps, _ in [
                    (ep_id, [s for e, s, _ in updates if e == ep_id], None) for ep_id in all_unique_episodes
                ]
            }
            # logger.debug(f"[BaseReplayBuffer.update_rewards_batch] Step ranges per episode: {step_ranges}")

        count = 0
        for i, (episode_id, step_in_episode, new_reward) in enumerate(updates):
            # logger.debug(
            #     f"[BaseReplayBuffer.update_rewards_batch] Processing update {i + 1}/{len(updates)}: episode_id={episode_id}, step_in_episode={step_in_episode}"
            # )
            if self.update_reward(episode_id, step_in_episode, new_reward):
                count += 1
        logger.debug(f"[BaseReplayBuffer.update_rewards_batch] Completed: {count}/{len(updates)} updates successful")
        return count

    def update_info(self, episode_id: Any, step_in_episode: int, info: Dict[str, Any]) -> bool:
        """
        Update the info dict for a specific transition by episode_id and step_in_episode.
        Used for retroactive info updates (e.g., async reward relabeling).

        The info is stored in the Transition's info field.

        Returns True if transition was found and updated, False otherwise.
        """
        # logger.debug(
        #     f"[BaseReplayBuffer.update_info] Called with episode_id={episode_id} (type={type(episode_id).__name__}), "
        #     f"step_in_episode={step_in_episode} (type={type(step_in_episode).__name__}), info_keys={list(info.keys())}"
        # )

        # # Use index if available (for subclasses like ReplayBuffer that maintain _transition_index)
        # if hasattr(self, "_transition_index"):
        #     key = (episode_id, step_in_episode)
        #     logger.debug(f"[BaseReplayBuffer.update_info] Using index, key={key}")
        #     if key in self._transition_index:
        #         idx = self._transition_index[key]
        #         all_transitions = self.get_all_transitions()
        #         logger.debug(
        #             f"[BaseReplayBuffer.update_info] Found key in index, idx={idx}, buffer_size={len(all_transitions)}"
        #         )
        #         if 0 <= idx < len(all_transitions):
        #             t = all_transitions[idx]
        #             logger.debug(
        #                 f"[BaseReplayBuffer.update_info] Transition at idx={idx}: episode_id={t.episode_id} (type={type(t.episode_id).__name__}), "
        #                 f"step_in_episode={t.step_in_episode} (type={type(t.step_in_episode).__name__})"
        #             )
        #             if t.episode_id == episode_id and t.step_in_episode == step_in_episode:
        #                 # Store info in transition's info field
        #                 if t.info is None:
        #                     t.info = {}
        #                 t.info.update(info)
        #                 logger.debug(f"[BaseReplayBuffer.update_info] Successfully updated info at idx={idx}")
        #                 return True
        #             else:
        #                 logger.warning(
        #                     f"[BaseReplayBuffer.update_info] Mismatch at idx={idx}: "
        #                     f"expected episode_id={episode_id} (type={type(episode_id).__name__}), got {t.episode_id} (type={type(t.episode_id).__name__}); "
        #                     f"expected step_in_episode={step_in_episode} (type={type(step_in_episode).__name__}), got {t.step_in_episode} (type={type(t.step_in_episode).__name__})"
        #                 )
        #             # Index stale, remove it
        #             self._transition_index.pop(key, None)
        #         else:
        #             logger.warning(
        #                 f"[BaseReplayBuffer.update_info] Index idx={idx} out of range [0, {len(all_transitions)})"
        #             )
        #     else:
        #         logger.debug(f"[BaseReplayBuffer.update_info] Key {key} not in index")

        # Fallback to linear search
        all_transitions = self.get_all_transitions()
        # logger.debug(
        #     f"[BaseReplayBuffer.update_info] Falling back to linear search over {len(all_transitions)} transitions"
        # )
        for i, transition in enumerate(all_transitions):
            if transition.episode_id == episode_id and transition.step_in_episode == step_in_episode:
                # logger.debug(f"[BaseReplayBuffer.update_info] Found match at linear search idx={i}")
                # Store info in transition's info field
                if transition.info is None:
                    transition.info = {}
                transition.info.update(info)
                return True
        # logger.warning(
        #     f"[BaseReplayBuffer.update_info] No match found after linear search. "
        #     f"Looking for episode_id={episode_id} (type={type(episode_id).__name__}), step_in_episode={step_in_episode} (type={type(step_in_episode).__name__}). "
        #     f"Sample transitions: {[(t.episode_id, t.step_in_episode, type(t.episode_id).__name__, type(t.step_in_episode).__name__) for t in all_transitions[:5]]}"
        # )
        return False

    def update_info_batch(self, updates: List[Tuple[Any, int, Dict[str, Any]]]) -> int:
        """Batch update info dicts. updates: List of (episode_id, step_in_episode, info_dict) tuples."""
        # logger.debug(f"[BaseReplayBuffer.update_info_batch] Called with {len(updates)} updates")
        if updates:
            sample_update = updates[0]
            # logger.debug(
            #     f"[BaseReplayBuffer.update_info_batch] Sample update: episode_id={sample_update[0]} (type={type(sample_update[0]).__name__}), "
            #     f"step_in_episode={sample_update[1]} (type={type(sample_update[1]).__name__}), info_keys={list(sample_update[2].keys())}"
            # )
            all_unique_episodes = set(ep_id for ep_id, _, _ in updates)
            # logger.debug(f"[BaseReplayBuffer.update_info_batch] Unique episode_ids in batch: {all_unique_episodes}")
            # Calculate step ranges per episode
            episode_steps = {}
            for ep_id, step, _ in updates:
                if ep_id not in episode_steps:
                    episode_steps[ep_id] = []
                episode_steps[ep_id].append(step)
            step_ranges = {ep_id: (min(steps), max(steps)) for ep_id, steps in episode_steps.items()}
            # logger.debug(f"[BaseReplayBuffer.update_info_batch] Step ranges per episode: {step_ranges}")

        count = 0
        for i, (episode_id, step_in_episode, info) in enumerate(updates):
            # logger.debug(
            #     f"[BaseReplayBuffer.update_info_batch] Processing update {i + 1}/{len(updates)}: episode_id={episode_id}, step_in_episode={step_in_episode}"
            # )
            if self.update_info(episode_id, step_in_episode, info):
                count += 1
        logger.debug(f"[BaseReplayBuffer.update_info_batch] Completed: {count}/{len(updates)} updates successful")
        return count

    def _update_chunk_cache_info(self, chunk_size: int | None, episode_id: Any = None, new_index: int | None = None):
        """Incrementally update cached episode boundaries and valid chunk starts.

        If chunk_size is None, only episode boundaries are updated. If chunk_size
        differs from the last cached value, valid chunk starts are rebuilt.
        """
        # Ensure boundaries cache exists; on first call, build from existing data
        if not hasattr(self, "_cached_boundaries") or self._cached_boundaries is None:
            try:
                self._cached_boundaries = self.get_episode_boundaries()
            except Exception:
                self._cached_boundaries = {}

        # Determine the index and episode_id of the newly added transition (if provided)
        if new_index is None or episode_id is None:
            # Fallback path: derive from current data (slower)
            all_transitions = self.get_all_transitions()
            if not all_transitions:
                return
            new_index = len(all_transitions) - 1
            last_transition = all_transitions[new_index]
            new_episode_id = getattr(last_transition, "episode_id", None)
        else:
            new_episode_id = episode_id

        # Update/extend boundaries for the episode of the last transition
        if new_episode_id in self._cached_boundaries:
            start_idx, end_idx = self._cached_boundaries[new_episode_id]
            if new_index > end_idx:
                self._cached_boundaries[new_episode_id] = (start_idx, new_index)
        else:
            self._cached_boundaries[new_episode_id] = (new_index, new_index)

        # If no chunk_size provided, stop after boundaries update
        if chunk_size is None:
            return

        # Initialize or rebuild valid_chunk_starts when chunk size changes or not present
        if (
            not hasattr(self, "_valid_chunk_starts")
            or self._valid_chunk_starts is None
            or getattr(self, "_chunk_size_cache", None) != chunk_size
        ):
            self._valid_chunk_starts = []
            self._chunk_size_cache = chunk_size
            # Rebuild from all boundaries
            for _episode_id, (start, end) in self._cached_boundaries.items():
                episode_length = end - start + 1
                if episode_length >= chunk_size:
                    self._valid_chunk_starts.extend(range(start, end - chunk_size + 2))
            return

        # Incrementally append the new valid start for the episode that just grew
        start_idx, end_idx = self._cached_boundaries[new_episode_id]
        episode_length = end_idx - start_idx + 1
        if episode_length >= chunk_size:
            new_start = end_idx - chunk_size + 1
            if len(self._valid_chunk_starts) == 0 or self._valid_chunk_starts[-1] != new_start:
                self._valid_chunk_starts.append(new_start)

    def get_contiguous_chunks(self, chunk_size: int, max_chunks: int) -> List[List[Transition]]:
        """
        Return random contiguous chunks of transitions from episodes.
        Optimized version that pre-computes valid chunk positions.

        Args:
            chunk_size: Size of each chunk
            max_chunks: Maximum number of chunks to return

        Returns:
            List of chunks, where each chunk is a list of transitions
        """
        # Ensure caches are up to date for the requested chunk_size
        try:
            self._update_chunk_cache_info(chunk_size)
        except Exception:
            # On any error, fall back to a conservative rebuild path
            self._cached_boundaries = self.get_episode_boundaries()
            self._valid_chunk_starts = []
            self._chunk_size_cache = chunk_size
            for _episode_id, (start, end) in self._cached_boundaries.items():
                episode_length = end - start + 1
                if episode_length >= chunk_size:
                    self._valid_chunk_starts.extend(range(start, end - chunk_size + 2))

        valid_starts = getattr(self, "_valid_chunk_starts", [])
        if not valid_starts:
            return []

        # Randomly sample starting positions
        if max_chunks > len(valid_starts):
            # Sample with replacement if we request more chunks than available
            sampled_starts = random.choices(valid_starts, k=max_chunks)
        else:
            # Sample without replacement otherwise
            sampled_starts = random.sample(valid_starts, max_chunks)

        # Get transitions once (avoid repeated calls)
        all_transitions = self.get_all_transitions()

        # Create chunks using slicing (much faster than list comprehension)
        chunks = []
        for start_idx in sampled_starts:
            end_idx = start_idx + chunk_size
            if end_idx <= len(all_transitions):
                # Use slicing instead of creating new lists
                chunk = all_transitions[start_idx:end_idx]
                chunks.append(chunk)

        return chunks

    def _batch_transitions(self, sampled: List[Transition]) -> Dict[str, Any]:
        # Check if obs is dict or array
        if not sampled:
            return {}

        first_obs = sampled[0].obs
        is_dict_obs = isinstance(first_obs, dict)

        if is_dict_obs:
            batched = {
                "obs": {k: [] for k in first_obs.keys()},
                "action": [],
                "reward": [],
                "next_obs": {k: [] for k in sampled[0].next_obs.keys()},
                "done": [],
                "truncated": [],
                "info": [],
            }
            for tr in sampled:
                for k, v in tr.obs.items():
                    batched["obs"][k].append(v)
                for k, v in tr.next_obs.items():
                    batched["next_obs"][k].append(v)
                batched["action"].append(tr.action)
                batched["reward"].append(tr.reward)
                batched["done"].append(tr.done)
                batched["truncated"].append(tr.truncated)
                batched["info"].append(tr.info if tr.info is not None else {})

            # Handle observations - avoid VisibleDeprecationWarning for ragged sequences
            for k in batched["obs"]:
                batched["obs"][k] = self._convert_obs_to_array(batched["obs"][k], f"obs.{k}")

            for k in batched["next_obs"]:
                batched["next_obs"][k] = self._convert_obs_to_array(batched["next_obs"][k], f"next_obs.{k}")
        else:
            batched = {
                "obs": np.array([tr.obs for tr in sampled]),
                "action": np.array([tr.action for tr in sampled]),
                "reward": np.array([tr.reward for tr in sampled]),
                "next_obs": np.array([tr.next_obs for tr in sampled]),
                "done": np.array([tr.done for tr in sampled]),
                "truncated": np.array([tr.truncated for tr in sampled]),
                "info": [tr.info if tr.info is not None else {} for tr in sampled],
            }

        # Post-process to match ReplayBuffer API - handle actions specially
        try:
            batched["action"] = np.array(batched["action"])
        except (ValueError, TypeError):
            # Actions might be tensors from ChunkedSequentialSampler
            action_list = []
            for a in batched["action"]:
                if hasattr(a, "detach"):  # It's a tensor
                    action_list.append(a.detach().cpu().numpy())
                else:
                    action_list.append(a)
            try:
                batched["action"] = np.array(action_list)
            except (ValueError, TypeError):
                # Keep as object array if we can't create regular array
                batched["action"] = np.array(action_list, dtype=object)

        batched["reward"] = np.array(batched["reward"])
        batched["done"] = np.array(batched["done"])
        batched["truncated"] = np.array(batched["truncated"])
        batched["done"] = batched["done"] * (1 - batched["truncated"])

        return batched

    def _is_ragged_sequence(self, seq):
        """Check if a sequence contains arrays/tensors of different shapes."""
        if not seq or len(seq) <= 1:
            return False

        # Get shape of first element
        first_item = seq[0]
        if hasattr(first_item, "shape"):
            first_shape = first_item.shape
        elif isinstance(first_item, (list, tuple)):
            first_shape = np.array(first_item).shape
        else:
            return False

        # Check if all other elements have the same shape
        for item in seq[1:]:
            if hasattr(item, "shape"):
                item_shape = item.shape
            elif isinstance(item, (list, tuple)):
                item_shape = np.array(item).shape
            else:
                return True  # Mixed types = ragged

            if item_shape != first_shape:
                return True

        return False

    def _convert_obs_to_array(self, obs_list, key_name):
        """Convert observation list to numpy array, handling ragged sequences."""
        # Handle tensor conversion first
        converted_list = []
        for v in obs_list:
            if hasattr(v, "detach"):  # It's a tensor
                converted_list.append(v.detach().cpu().numpy())
            else:
                converted_list.append(v)

        # Check if sequence is ragged before attempting numpy array creation
        if self._is_ragged_sequence(converted_list):
            return np.array(converted_list, dtype=object)
        else:
            try:
                return np.array(converted_list)
            except (ValueError, TypeError):
                # Fallback to object array
                return np.array(converted_list, dtype=object)

    def sample(
        self,
        batch_size: int,
        device: str = None,
        dtype=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sample with optional strategy override. Respects sampler semantics
        (e.g., chunked sequence sampling) and returns a batched tensor dict.

        Args:
            batch_size: Number of transitions to sample
            convert_to_tensors: Whether to convert numpy arrays to PyTorch tensors
            device: Device to move tensors to (e.g., 'cuda', 'cpu')
            dtype: Data type for tensors (default: torch.float32)
            **kwargs: Additional sampling parameters
        """
        active_sampler = self.sampler
        sampled = active_sampler.sample(self, batch_size, **kwargs)

        # Apply post-transforms (prefer batched transforms for efficiency)
        if self.post_transforms:
            _n_transitions = len(sampled)
            _t0 = time.perf_counter()
            _batched = batch_transitions(sampled)
            _batched_out = apply_transforms_batched(_batched, self.post_transforms)
            sampled = unbatch_transitions(_batched_out)
            _dt = time.perf_counter() - _t0

        transition = self._batch_transitions(sampled)

        # Convert to tensors if requested
        if transition:
            transition = self._convert_batch_to_tensors(transition, device, dtype)

        return transition

    def _convert_batch_to_tensors(self, batch: Dict[str, Any], device: str = None, dtype=None) -> Dict[str, Any]:
        """Convert batch from buffer to tensors on the specified device."""
        import torch

        # Set default dtype
        if dtype is None:
            dtype = torch.float32

        # Use pinned memory on CPU for faster HtoD copies
        pin = device is not None and isinstance(device, str) and device.startswith("cuda")

        # Handle different observation formats
        if isinstance(batch["obs"], dict):
            # Dictionary observations (CNN)
            obs = {}
            next_obs = {}
            for key in batch["obs"].keys():
                # Handle object arrays from ChunkedSequentialSampler
                if batch["obs"][key].dtype == object:
                    # Convert object array to tensor
                    obs_list = []
                    for item in batch["obs"][key]:
                        if isinstance(item, np.ndarray):
                            obs_list.append(torch.from_numpy(item))
                        elif hasattr(item, "detach"):  # Already a tensor
                            obs_list.append(item)
                        else:
                            obs_list.append(torch.tensor(item))

                    # Check if all tensors have the same shape
                    if len(obs_list) > 0:
                        shapes = [t.shape for t in obs_list]
                        if all(s == shapes[0] for s in shapes):
                            # All same shape, can stack normally
                            obs[key] = torch.stack(obs_list).to(dtype=dtype)
                        else:
                            # Different shapes means the batch is corrupted for learning.
                            # Silently repeating the first element poisons training and can look like "instability".
                            raise ValueError(
                                f"Variable shapes detected for obs key '{key}': {shapes[:10]} "
                                f"(showing up to 10). Fix upstream observation generation or add explicit padding."
                            )
                    else:
                        obs[key] = torch.empty(0).to(dtype=dtype)
                else:
                    t = torch.from_numpy(batch["obs"][key]).to(dtype=dtype)
                    if pin:
                        t = t.pin_memory()
                    obs[key] = t

                # Same for next_obs
                if batch["next_obs"][key].dtype == object:
                    next_obs_list = []
                    for item in batch["next_obs"][key]:
                        if isinstance(item, np.ndarray):
                            next_obs_list.append(torch.from_numpy(item))
                        elif hasattr(item, "detach"):  # Already a tensor
                            next_obs_list.append(item)
                        else:
                            next_obs_list.append(torch.tensor(item))

                    # Check if all tensors have the same shape
                    if len(next_obs_list) > 0:
                        shapes = [t.shape for t in next_obs_list]
                        if all(s == shapes[0] for s in shapes):
                            # All same shape, can stack normally
                            next_obs[key] = torch.stack(next_obs_list).to(dtype=dtype)
                        else:
                            raise ValueError(
                                f"Variable shapes detected for next_obs key '{key}': {shapes[:10]} "
                                f"(showing up to 10). Fix upstream observation generation or add explicit padding."
                            )
                    else:
                        next_obs[key] = torch.empty(0).to(dtype=dtype)
                else:
                    t = torch.from_numpy(batch["next_obs"][key]).to(dtype=dtype)
                    if pin:
                        t = t.pin_memory()
                    next_obs[key] = t

                if device is not None:
                    obs[key] = obs[key].to(device)
                    next_obs[key] = next_obs[key].to(device)
        else:
            # Simple observations (MLP)
            obs = torch.from_numpy(batch["obs"]).to(dtype=dtype)
            next_obs = torch.from_numpy(batch["next_obs"]).to(dtype=dtype)
            if pin:
                obs = obs.pin_memory()
                next_obs = next_obs.pin_memory()
            if device is not None:
                obs = obs.to(device)
                next_obs = next_obs.to(device)

        # Handle actions - might be object array from ChunkedSequentialSampler
        if batch["action"].dtype == object:
            action_list = []
            for item in batch["action"]:
                if isinstance(item, np.ndarray):
                    action_list.append(torch.from_numpy(item))
                elif hasattr(item, "detach"):  # Already a tensor
                    action_list.append(item)
                else:
                    action_list.append(torch.tensor(item))

            # Check if all tensors have the same shape
            if len(action_list) > 0:
                shapes = [t.shape for t in action_list]
                if all(s == shapes[0] for s in shapes):
                    # All same shape, can stack normally
                    actions = torch.stack(action_list).to(dtype=dtype)
                else:
                    # Different shapes - this shouldn't happen for actions but handle it
                    print(f"Warning: Variable action shapes detected: {shapes[:5]}...")
                    actions = torch.stack(action_list).to(dtype=dtype)  # This will fail, but that's expected
            else:
                actions = torch.empty(0).to(dtype=dtype)
        else:
            actions = torch.from_numpy(batch["action"]).to(dtype=dtype)
            if pin:
                actions = actions.pin_memory()

        rewards = torch.from_numpy(batch["reward"]).to(dtype=dtype)
        dones = torch.from_numpy(batch["done"]).to(dtype=dtype)
        truncateds = torch.from_numpy(batch["truncated"]).to(dtype=dtype)
        if pin:
            rewards = rewards.pin_memory()
            dones = dones.pin_memory()
            truncateds = truncateds.pin_memory()

        if device is not None:
            non_block = True
            actions = actions.to(device, non_blocking=non_block)
            rewards = rewards.to(device, non_blocking=non_block)
            dones = dones.to(device, non_blocking=non_block)
            truncateds = truncateds.to(device, non_blocking=non_block)

        return {
            "obs": obs,
            "action": actions,
            "reward": rewards,
            "next_obs": next_obs,
            "done": dones,
            "truncated": truncateds,
            "info": batch.get("info", [{}] * len(batch["reward"])),
        }

    def add(self, obs, action, reward, next_obs, done, truncated, **kwargs):
        # Put everything on the cpu
        obs = move_to_device(obs, "cpu")
        action = move_to_device(action, "cpu")
        reward = move_to_device(reward, "cpu")
        next_obs = move_to_device(next_obs, "cpu")
        done = move_to_device(done, "cpu")
        truncated = move_to_device(truncated, "cpu")

        obs, action, reward, next_obs, done, truncated = self._preprocess_input(
            obs, action, reward, next_obs, done, truncated
        )
        self._add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, truncated=truncated, **kwargs)

        # Incrementally update caches using the known episode_id and new index without scanning the buffer
        try:
            episode_id = kwargs.get("episode_id", None)
            new_index = len(self) - 1
            # Use fast-path update; if chunk size cache is active, we maintain starts too
            fast_chunk_size = getattr(self, "_chunk_size_cache", None)
            self._update_chunk_cache_info(fast_chunk_size, episode_id=episode_id, new_index=new_index)
        except Exception:
            # On any unexpected issue, defer to rebuild on next request
            self._cached_boundaries = None
            self._valid_chunk_starts = None
            # keep _chunk_size_cache so next rebuild knows desired chunk size

    def _preprocess_input(self, obs, action, reward, next_obs, done, truncated) -> Tuple[Any, Any, Any, Any, Any, Any]:
        # Call the pre-transforms
        for transform in self.pre_transforms:
            input_dict = {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "truncated": truncated,
            }
            obs, action, reward, next_obs, done, truncated = transform(**input_dict)

        return obs, action, reward, next_obs, done, truncated

    def update_reward_model(self, reward_model):
        """
        Update the reward model.
        """
        self.reward_model = reward_model

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def size(self):
        pass

    @abc.abstractmethod
    def is_empty(self):
        pass

    @abc.abstractmethod
    def clear(self):
        pass

    def save_to_npz(self, save_path: str, save_images: bool = False, image_keys: List[str] = None):
        """
        Save the replay buffer to an npz file, organized by episodes.
        
        The saved file structure is:
        - metadata: dict with num_episodes, num_transitions, obs_keys, etc.
        - episode_starts: array of starting indices for each episode
        - episode_lengths: array of length for each episode
        - episode_ids: array of episode IDs
        - For each data key (obs/*, next_obs/*, actions, rewards, etc.):
          The data is stored as a flat array, but can be reconstructed per-episode
          using episode_starts and episode_lengths.
        
        Args:
            save_path: Path to save the npz file (should end with .npz)
            save_images: If True, save raw image frames (can be large). If False, only save embeddings.
            image_keys: List of image keys to save (e.g., ['image', 'observation/image']). 
                       If None and save_images=True, saves all image-like keys found.
        """
        import os
        
        if save_images:
            logger.info(f"[ReplayBuffer.save_to_npz] save_images=True: Will save raw image frames (file may be large)")
        
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Get all transitions
        transitions = self.get_all_transitions()
        
        if len(transitions) == 0:
            logger.warning(f"[ReplayBuffer.save_to_npz] Buffer is empty, nothing to save")
            return
        
        logger.info(f"[ReplayBuffer.save_to_npz] Saving {len(transitions)} transitions to {save_path}")
        
        # Group transitions by episode
        episodes = {}  # episode_id -> list of (original_idx, transition)
        for i, transition in enumerate(transitions):
            ep_id = transition.episode_id if transition.episode_id is not None else -1
            if ep_id not in episodes:
                episodes[ep_id] = []
            episodes[ep_id].append((i, transition))
        
        # Sort episodes by their first transition's index to maintain order
        sorted_episode_ids = sorted(episodes.keys(), key=lambda ep: episodes[ep][0][0])
        
        logger.info(f"[ReplayBuffer.save_to_npz] Found {len(sorted_episode_ids)} episodes")
        
        # Build episode metadata
        episode_starts = []
        episode_lengths = []
        episode_id_list = []
        episode_returns = []
        episode_successes = []
        
        # Organize data by keys (flat arrays, but in episode order)
        data_dict = {}
        current_idx = 0
        
        for ep_id in sorted_episode_ids:
            ep_transitions = episodes[ep_id]
            # Sort by step_in_episode within each episode
            ep_transitions.sort(key=lambda x: x[1].step_in_episode if x[1].step_in_episode is not None else x[0])
            
            episode_starts.append(current_idx)
            episode_lengths.append(len(ep_transitions))
            episode_id_list.append(ep_id)
            
            # Calculate episode return and success
            ep_return = sum(t.reward for _, t in ep_transitions)
            ep_success = any(t.done and not t.truncated for _, t in ep_transitions)
            # Also check info for success
            for _, t in ep_transitions:
                if t.info is not None and t.info.get("is_success", False):
                    ep_success = True
                    break
            
            episode_returns.append(ep_return)
            episode_successes.append(ep_success)
            
            # Add transitions for this episode
            for _, transition in ep_transitions:
                # Handle observations (dict format)
                if isinstance(transition.obs, dict):
                    for key, value in transition.obs.items():
                        # Skip image keys if save_images=False
                        if not save_images:
                            is_image_key = "image" in key.lower()
                            if is_image_key:
                                if isinstance(value, np.ndarray):
                                    if value.ndim >= 3 or (value.ndim == 2 and value.shape[0] > 100):
                                        continue
                        else:
                            if image_keys is not None and "image" in key.lower():
                                if not any(img_key in key for img_key in image_keys):
                                    continue
                        
                        obs_key = f"obs/{key}"
                        if obs_key not in data_dict:
                            data_dict[obs_key] = []
                        data_dict[obs_key].append(np.array(value))
                
                # Handle next_obs (dict format)
                if isinstance(transition.next_obs, dict):
                    for key, value in transition.next_obs.items():
                        if not save_images:
                            is_image_key = "image" in key.lower()
                            if is_image_key:
                                if isinstance(value, np.ndarray):
                                    if value.ndim >= 3 or (value.ndim == 2 and value.shape[0] > 100):
                                        continue
                        else:
                            if image_keys is not None and "image" in key.lower():
                                if not any(img_key in key for img_key in image_keys):
                                    continue
                        
                        next_obs_key = f"next_obs/{key}"
                        if next_obs_key not in data_dict:
                            data_dict[next_obs_key] = []
                        data_dict[next_obs_key].append(np.array(value))
                
                # Handle actions
                if "actions" not in data_dict:
                    data_dict["actions"] = []
                data_dict["actions"].append(np.array(transition.action))
                
                # Handle rewards
                if "rewards" not in data_dict:
                    data_dict["rewards"] = []
                data_dict["rewards"].append(float(transition.reward))
                
                # Handle dones
                if "dones" not in data_dict:
                    data_dict["dones"] = []
                data_dict["dones"].append(bool(transition.done))
                
                # Handle truncated
                if "truncated" not in data_dict:
                    data_dict["truncated"] = []
                data_dict["truncated"].append(bool(transition.truncated))
                
                # Handle step_in_episode
                if "step_in_episode" not in data_dict:
                    data_dict["step_in_episode"] = []
                data_dict["step_in_episode"].append(transition.step_in_episode if transition.step_in_episode is not None else -1)
                
                # Handle info dict if present
                if transition.info is not None and isinstance(transition.info, dict):
                    for key, value in transition.info.items():
                        info_key = f"info/{key}"
                        if info_key not in data_dict:
                            data_dict[info_key] = []
                        try:
                            data_dict[info_key].append(np.array(value))
                        except:
                            pass
                
                current_idx += 1
        
        # Convert lists to numpy arrays
        save_dict = {}
        
        # Add episode-level metadata
        save_dict["episode_starts"] = np.array(episode_starts, dtype=np.int64)
        save_dict["episode_lengths"] = np.array(episode_lengths, dtype=np.int64)
        save_dict["episode_ids"] = np.array(episode_id_list)
        save_dict["episode_returns"] = np.array(episode_returns, dtype=np.float32)
        save_dict["episode_successes"] = np.array(episode_successes, dtype=bool)
        
        # Add transition data
        for key, value_list in data_dict.items():
            try:
                save_dict[key] = np.array(value_list)
                logger.debug(f"[ReplayBuffer.save_to_npz] Saved {key} with shape {save_dict[key].shape}")
            except Exception as e:
                logger.warning(f"[ReplayBuffer.save_to_npz] Failed to convert {key} to numpy array: {e}")
        
        # Save to npz
        np.savez_compressed(save_path, **save_dict)
        
        # Log summary
        logger.info(f"[ReplayBuffer.save_to_npz] Successfully saved buffer to {save_path}")
        logger.info(f"  - Total transitions: {len(transitions)}")
        logger.info(f"  - Total episodes: {len(sorted_episode_ids)}")
        logger.info(f"  - Avg episode length: {np.mean(episode_lengths):.1f}")
        logger.info(f"  - Success rate: {np.mean(episode_successes) * 100:.1f}%")
        logger.info(f"  - Avg return: {np.mean(episode_returns):.4f}")
        
        # Log file size
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        logger.info(f"  - File size: {file_size_mb:.2f} MB")
