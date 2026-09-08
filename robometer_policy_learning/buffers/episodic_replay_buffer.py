import random
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import numpy as np

from robometer_policy_learning.buffers.base_replay_buffer import BaseReplayBuffer, Transition


class EpisodicReplayBuffer(BaseReplayBuffer):
    """
    Episodic replay buffer for multi-env/parallel envs.
    Transitions are only added to the main buffer once a full episode is completed (done or truncated).
    Each episode is tracked by a unique episode_id (e.g., env_id).
    Only supports sampling transitions (not episodes).
    """

    def __init__(
        self,
        capacity: int = 100000,
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
        self.capacity = capacity
        self.buffer: List[Transition] = []  # Main buffer of transitions
        self.episode_buffers: Dict[Any, List[Transition]] = defaultdict(list)  # Temporary per-episode buffers
        self.episode_id_printed = False

    def _add(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        truncated,
        episode_id=-1,
        step_in_episode=None,
        timestamp=None,
    ):
        """
        Add a transition to the episode buffer for episode_id.
        When done/truncated, flush the episode to the main buffer.
        """

        if episode_id == -1:
            # print only one time ever that the episode is not tracked and this is okay if
            # only one environment is used
            if not self.episode_id_printed:
                print("EpisodicReplayBuffer: Episode not tracked. This is okay if only one environment is used.")
                self.episode_id_printed = True

        # Set step_in_episode if not provided
        if step_in_episode is None:
            step_in_episode = len(self.episode_buffers[episode_id])

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
        self.episode_buffers[episode_id].append(transition)
        if done or truncated:
            self._finalize_episode(episode_id)

    def _finalize_episode(self, episode_id):
        """
        Move all transitions from the episode buffer to the main buffer.
        """
        episode = self.episode_buffers.pop(episode_id, [])
        if episode:
            # Enforce capacity (FIFO)
            total = len(self.buffer) + len(episode)
            if total > self.capacity:
                excess = total - self.capacity
                self.buffer = self.buffer[excess:]
            self.buffer.extend(episode)

    def get_all_transitions(self) -> List[Transition]:
        """Get all transitions in the buffer."""
        return self.buffer.copy()

    def get_episode_boundaries(self) -> Dict[Any, Tuple[int, int]]:
        """Return episode_id -> (start_idx, end_idx) mapping - optimized version."""
        boundaries = {}
        current_idx = 0

        # Group transitions by episode_id while maintaining order
        episode_groups = {}
        for i, transition in enumerate(self.buffer):
            if transition.episode_id not in episode_groups:
                episode_groups[transition.episode_id] = []
            episode_groups[transition.episode_id].append(i)

        # Create boundaries for contiguous episodes
        for episode_id, indices in episode_groups.items():
            if indices:
                boundaries[episode_id] = (indices[0], indices[-1])

        return boundaries

    def get_episode_end_transitions(self, count: int) -> List[Transition]:
        """Return transitions that end episodes - optimized version."""
        episode_ends = [t for t in self.buffer if t.done or t.truncated]
        return random.sample(episode_ends, min(count, len(episode_ends))) if episode_ends else []

    def get_transitions_by_episode(self, episode_id: Any) -> List[Transition]:
        """Return all transitions from a specific episode - optimized version."""
        return [t for t in self.buffer if t.episode_id == episode_id]

    def __len__(self):
        return len(self.buffer)

    def size(self):
        return len(self.buffer)

    def is_empty(self):
        return len(self.buffer) == 0

    def clear(self):
        self.buffer.clear()
        self.episode_buffers.clear()
