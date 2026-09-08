import random
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Callable

from robometer_policy_learning.buffers.base_replay_buffer import BaseReplayBuffer, Transition


class ReplayBuffer(BaseReplayBuffer):
    """
    A basic replay buffer for storing and sampling experience transitions.
    Supports both simple and dictionary observations.
    """

    def __init__(
        self,
        capacity: int = 100000,
        obs_keys: List[str] = None,
        remove_obs_keys: List[str] = None,
        rename_obs_keys: Dict[str, str] = None,
        pre_transforms: List[Callable] = None,
        post_transforms: List[Callable] = None,
        sampler=None,
    ):
        super().__init__(
            obs_keys=obs_keys,
            remove_obs_keys=remove_obs_keys,
            rename_obs_keys=rename_obs_keys,
            pre_transforms=pre_transforms,
            post_transforms=post_transforms,
            sampler=sampler,
        )
        self.capacity = capacity
        self.buffer = [None] * capacity
        self._size = 0
        self._write_pos = 0
        self._transition_index: Dict[Tuple[Any, int], int] = {}
        self._episode_steps: Dict[Any, Dict[int, int]] = {}

    def _add(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        truncated,
        episode_id=None,
        step_in_episode=None,
        timestamp=None,
        **kwargs,
    ):
        obs = {k: v for k, v in obs.items() if k not in self.remove_obs_keys}
        next_obs = {k: v for k, v in next_obs.items() if k not in self.remove_obs_keys}
        """Add a single transition to the buffer."""
        transition = Transition(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            truncated=truncated,
            episode_id=episode_id,
            step_in_episode=step_in_episode,
            timestamp=timestamp,
        )

        # Evict the entry being overwritten (if buffer is full)
        if self._size == self.capacity:
            old = self.buffer[self._write_pos]
            if old is not None and old.episode_id is not None and old.step_in_episode is not None:
                self._transition_index.pop((old.episode_id, old.step_in_episode), None)
                ep_dict = self._episode_steps.get(old.episode_id)
                if ep_dict is not None:
                    ep_dict.pop(old.step_in_episode, None)
                    if not ep_dict:
                        del self._episode_steps[old.episode_id]

        self.buffer[self._write_pos] = transition
        if episode_id is not None and step_in_episode is not None:
            self._transition_index[(episode_id, step_in_episode)] = self._write_pos
            self._episode_steps.setdefault(episode_id, {})[step_in_episode] = self._write_pos

        self._write_pos = (self._write_pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample_indices(self, batch_size: int, sampler=None) -> List[int]:
        """Return random buffer-position indices for O(batch_size) sampling."""
        if self._size < batch_size:
            return random.choices(range(self._size), k=batch_size)
        return random.sample(range(self._size), batch_size)

    def transitions_from_indices(self, indices) -> List[Transition]:
        """Fetch transitions by their buffer-position indices."""
        return [self.buffer[i] for i in indices]

    def get_all_transitions(self) -> List[Transition]:
        """Get all transitions in the buffer in insertion order."""
        if self._size < self.capacity:
            return self.buffer[: self._size]
        return self.buffer[self._write_pos :] + self.buffer[: self._write_pos]

    def __len__(self):
        """Return the current size of the buffer."""
        return self._size

    def size(self):
        """Return the current size of the buffer."""
        return self._size

    def is_empty(self):
        """Check if the buffer is empty."""
        return self._size == 0

    def clear(self):
        """Clear all transitions from the buffer."""
        self.buffer = [None] * self.capacity
        self._size = 0
        self._write_pos = 0
        self._transition_index.clear()
        self._episode_steps.clear()

    def get_contiguous_chunks(self, chunk_size: int, max_chunks: int):
        """Return contiguous chunks that correctly handle interleaved episodes.

        Uses the incrementally maintained ``_episode_steps`` index
        (ep_id → {step → buf_pos}) so this is O(max_chunks) per call rather
        than O(buffer_size).
        """
        if self._size == 0 or chunk_size <= 0 or max_chunks <= 0:
            return []

        eligible = [
            (ep_id, steps)
            for ep_id, steps in self._episode_steps.items()
            if len(steps) >= chunk_size
        ]
        if not eligible:
            return []

        weights = [len(steps) - chunk_size + 1 for _, steps in eligible]
        ep_indices = random.choices(range(len(eligible)), weights=weights, k=max_chunks * 2)

        sorted_cache: dict = {}
        chunks: list = []
        for ep_idx in ep_indices:
            if len(chunks) >= max_chunks:
                break
            if ep_idx not in sorted_cache:
                sorted_cache[ep_idx] = sorted(eligible[ep_idx][1].items())
            sorted_steps = sorted_cache[ep_idx]
            n = len(sorted_steps)
            start = random.randint(0, n - chunk_size)
            ok = True
            for j in range(chunk_size - 1):
                if sorted_steps[start + j + 1][0] != sorted_steps[start + j][0] + 1:
                    ok = False
                    break
            if ok:
                chunks.append([self.buffer[pos] for _, pos in sorted_steps[start : start + chunk_size]])

        return chunks

    def update_reward(self, episode_id: Any, step_in_episode: int, new_reward: float) -> bool:
        """Optimized reward update using index for O(1) lookup."""

        key = (episode_id, step_in_episode)
        if key in self._transition_index:
            idx = self._transition_index[key]
            if 0 <= idx < self._size:
                t = self.buffer[idx]
                if t.episode_id == episode_id and t.step_in_episode == step_in_episode:
                    t.reward = new_reward
                    return True
                self._transition_index.pop(key, None)
        return super().update_reward(episode_id, step_in_episode, new_reward)

    def update_info(self, episode_id: Any, step_in_episode: int, info: Dict[str, Any]) -> bool:
        """Optimized info update using index for O(1) lookup."""

        key = (episode_id, step_in_episode)
        if key in self._transition_index:
            idx = self._transition_index[key]
            if 0 <= idx < self._size:
                t = self.buffer[idx]
                if t.episode_id == episode_id and t.step_in_episode == step_in_episode:
                    if t.info is None:
                        t.info = {}
                    t.info.update(info)
                    return True
                self._transition_index.pop(key, None)
        return super().update_info(episode_id, step_in_episode, info)


if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=100000)
    buffer.add(
        obs=np.array([1, 2, 3]),
        action=np.array([4, 5, 6]),
        reward=np.array([7, 8, 9]),
        next_obs=np.array([10, 11, 12]),
        done=np.array([0, 0, 0]),
        truncated=np.array([0, 0, 0]),
    )
    print(buffer.sample(batch_size=10))
