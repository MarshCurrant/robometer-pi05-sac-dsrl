import abc
import random
from typing import List, Any, TYPE_CHECKING
import torch
import numpy as np
import dataclasses

if TYPE_CHECKING:
    from robometer_policy_learning.buffers.base_replay_buffer import BaseReplayBuffer, Transition


def _stack_values(values):
    """Batch-convert a list of same-typed arrays/tensors into a single stacked tensor.

    Checks the first element's type once and branches accordingly, avoiding
    per-element isinstance/is_tensor checks.
    """
    first = values[0]
    if isinstance(first, np.ndarray):
        return torch.from_numpy(np.stack(values))
    if torch.is_tensor(first):
        return torch.stack(values)
    return torch.stack([torch.as_tensor(v) for v in values])


class BaseSampler(abc.ABC):
    """Base class for sampling strategies."""

    @abc.abstractmethod
    def sample(self, buffer: "BaseReplayBuffer", batch_size: int, **kwargs) -> List["Transition"]:
        """Sample transitions from buffer according to strategy."""
        pass

    def can_sample(self, buffer: "BaseReplayBuffer", batch_size: int) -> bool:
        """Check if sampling is possible with current buffer state."""
        return not buffer.is_empty() and len(buffer) > 0


class RandomSampler(BaseSampler):
    """Random sampling - default behavior."""

    def sample(self, buffer: "BaseReplayBuffer", batch_size: int, **kwargs) -> List["Transition"]:
        if buffer.is_empty():
            return []
        if hasattr(buffer, "sample_indices") and hasattr(buffer, "transitions_from_indices"):
            idxs = buffer.sample_indices(batch_size, sampler=self)
            return buffer.transitions_from_indices(np.array(idxs, dtype=np.int64))
        all_transitions = buffer.get_all_transitions()
        if not all_transitions:
            return []

        if len(all_transitions) < batch_size:
            return random.choices(all_transitions, k=batch_size)

        return random.sample(all_transitions, batch_size)


class RelabeledOnlySampler(BaseSampler):
    """
    Random sampling that only samples from transitions with relabeled rewards.
    Useful for async reward relabeling to ensure we only train on properly relabeled data.

    Returns empty list if not enough relabeled transitions are available.
    """

    def __init__(self, min_relabeled_ratio: float = 0.1):
        """
        Args:
            min_relabeled_ratio: Minimum ratio of relabeled transitions required before sampling.
                                If ratio is below this, returns empty list.
        """
        self.min_relabeled_ratio = min_relabeled_ratio

    def sample(self, buffer: "BaseReplayBuffer", batch_size: int, **kwargs) -> List["Transition"]:
        total_count = len(buffer)
        if total_count == 0:
            return []

        relabeled_transitions = buffer.get_relabeled_transitions()
        if not relabeled_transitions:
            return []

        if len(relabeled_transitions) / total_count < self.min_relabeled_ratio:
            return []

        if len(relabeled_transitions) < batch_size:
            return random.choices(relabeled_transitions, k=batch_size)
        return random.sample(relabeled_transitions, batch_size)

    def can_sample(self, buffer: "BaseReplayBuffer", batch_size: int) -> bool:
        """Check if we can sample with current relabeling state."""
        if buffer.is_empty():
            return False
        total_count = len(buffer)
        if total_count == 0:
            return False
        relabeled_count = buffer.get_relabeled_count()
        return (relabeled_count / total_count) >= self.min_relabeled_ratio


class ChunkedSequentialSampler(BaseSampler):
    """
    Sample chunks and return them as sequences for RNN training.
    Returns fewer transitions but each contains sequence data.
    """

    def __init__(self, chunk_size, gamma, obs_as_sequence: bool = False):
        """
        Args:
            chunk_size: Size of chunks to sample
            gamma: Discount factor for rewards for the chunk.
            obs_as_sequence: If True, stack observations as sequences (for RNN).
                           If False, use single observation (for transformer chunking).
        """
        self.chunk_size = chunk_size
        self.obs_as_sequence = obs_as_sequence
        self.gamma = float(gamma)
        # Pre-compute in float64 for precision, then store as float32
        self._discount_factors = torch.pow(
            torch.tensor(gamma, dtype=torch.float64),
            torch.arange(chunk_size, dtype=torch.float64),
        ).float()

    def sample(self, buffer: "BaseReplayBuffer", batch_size: int, **kwargs) -> List["Transition"]:
        if hasattr(buffer, "get_contiguous_chunks_optimized"):
            chunks = buffer.get_contiguous_chunks_optimized(
                self.chunk_size, batch_size, obs_as_sequence=self.obs_as_sequence
            )
        else:
            chunks = buffer.get_contiguous_chunks(self.chunk_size, batch_size)

        sequence_transitions = []
        for chunk in chunks:
            if len(chunk) >= self.chunk_size:
                seq_transition = self._chunk_to_sequence(chunk[: self.chunk_size])
                sequence_transitions.append(seq_transition)

        return sequence_transitions[:batch_size]

    def sample_indices(self, buffer: "BaseReplayBuffer", batch_size: int):
        return None

    def _chunk_to_sequence(self, chunk: List["Transition"]) -> "Transition":
        """Convert chunk to sequence format for RNN training."""
        first = chunk[0]

        # --- Observations ---
        if self.obs_as_sequence:
            if isinstance(first.obs, dict):
                obs_seq = {key: _stack_values([t.obs[key] for t in chunk]) for key in first.obs}
            else:
                obs_seq = _stack_values([t.obs for t in chunk])

            if isinstance(first.next_obs, dict):
                next_obs_seq = {key: _stack_values([t.next_obs[key] for t in chunk]) for key in first.next_obs}
            else:
                next_obs_seq = _stack_values([t.next_obs for t in chunk])
        else:
            obs_seq = first.obs
            next_obs_seq = chunk[-1].next_obs

        # --- Actions (always sequenced) ---
        action_seq = _stack_values([t.action for t in chunk])

        # --- Discounted reward sum ---
        # Single torch.tensor call instead of N individual conversions + stack
        reward_tensor = torch.tensor([t.reward for t in chunk])
        disc = self._discount_factors[: len(chunk)]
        if reward_tensor.dtype != disc.dtype:
            disc = disc.to(dtype=reward_tensor.dtype)
        reward_seq = torch.dot(reward_tensor, disc)

        # --- Done / truncated flags ---
        # Python any() short-circuits; avoids N tensor creations + stack + torch.any
        done_seq = torch.tensor(any(t.done for t in chunk))
        truncated_seq = torch.tensor(any(t.truncated for t in chunk))

        return dataclasses.replace(
            chunk[-1],
            obs=obs_seq,
            action=action_seq,
            reward=reward_seq,
            next_obs=next_obs_seq,
            done=done_seq,
            truncated=truncated_seq,
        )


class EpisodeEndSampler(BaseSampler):
    """Sample the last transition from each episode."""

    def sample(self, buffer: "BaseReplayBuffer", batch_size: int, **kwargs) -> List["Transition"]:
        return buffer.get_episode_end_transitions(batch_size)


class EpisodeStartSampler(BaseSampler):
    """Sample the first transition from each episode."""

    def sample(self, buffer: "BaseReplayBuffer", batch_size: int, **kwargs) -> List["Transition"]:
        boundaries = buffer.get_episode_boundaries()
        all_transitions = buffer.get_all_transitions()

        episode_starts = []
        for start, _end in boundaries.values():
            if start < len(all_transitions):
                episode_starts.append(all_transitions[start])

        return random.sample(episode_starts, min(batch_size, len(episode_starts))) if episode_starts else []


class TemporalSampler(BaseSampler):
    """Sample based on recency/temporal patterns."""

    def __init__(self, recency_weight: float = 2.0):
        """
        Args:
            recency_weight: Weight for recent transitions (higher = more recent bias)
        """
        self.recency_weight = recency_weight

    def sample(self, buffer: "BaseReplayBuffer", batch_size: int, **kwargs) -> List["Transition"]:
        all_transitions = buffer.get_all_transitions()
        if not all_transitions:
            return []

        n = len(all_transitions)
        k = min(batch_size, n)

        if all_transitions[0].timestamp is not None:
            weights = np.empty(n, dtype=np.float64)
            for i, t in enumerate(all_transitions):
                weights[i] = t.timestamp if t.timestamp is not None else float(i + 1)
        else:
            weights = np.arange(1, n + 1, dtype=np.float64)

        np.power(weights, self.recency_weight, out=weights)

        total = weights.sum()
        if total == 0:
            indices = np.random.randint(0, n, size=k)
        else:
            weights /= total
            indices = np.random.choice(n, size=k, replace=True, p=weights)

        return [all_transitions[i] for i in indices]


class EpisodeBalancedSampler(BaseSampler):
    """Sample transitions while ensuring balanced representation across episodes."""

    def sample(self, buffer: "BaseReplayBuffer", batch_size: int, **kwargs) -> List["Transition"]:
        boundaries = buffer.get_episode_boundaries()
        if not boundaries:
            return RandomSampler().sample(buffer, batch_size, **kwargs)

        all_transitions = buffer.get_all_transitions()
        n_episodes = len(boundaries)
        transitions_per_episode = batch_size // n_episodes
        remainder = batch_size % n_episodes

        sampled_transitions = []
        episode_ids = list(boundaries.keys())
        random.shuffle(episode_ids)

        for i, episode_id in enumerate(episode_ids):
            start, end = boundaries[episode_id]
            episode_transitions = all_transitions[start : end + 1]

            num_samples = transitions_per_episode + (1 if i < remainder else 0)
            num_samples = min(num_samples, len(episode_transitions))

            if num_samples > 0:
                sampled_transitions.extend(random.sample(episode_transitions, num_samples))

        return sampled_transitions
