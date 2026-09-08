"""
Replay buffer wrapper that uses async reward relabeling service in "pre" mode.
Rewards are relabeled before transitions are added to the underlying buffer.
"""

import threading
from typing import Dict, Any, List, Optional, Callable
import numpy as np
from loguru import logger

from robometer_policy_learning.buffers.base_replay_buffer import BaseReplayBuffer, Transition
from robometer_policy_learning.distributed.clients.reward_relabel_client import RewardRelabelClient, PendingRelabel, PendingRelabelBatch


class AsyncRewardRelabelBuffer(BaseReplayBuffer):
    """
    Replay buffer wrapper that relabels rewards asynchronously before adding to buffer.

    In "pre" mode, transitions are queued for reward relabeling, and only added
    to the underlying buffer once relabeling is complete.

    Args:
        underlying_buffer: The actual replay buffer to add relabeled transitions to
        reward_relabel_client: RewardRelabelClient instance (will be started automatically)
        use_relative_rewards: Whether to use relative rewards (delta from previous)
    """

    def __init__(
        self,
        underlying_buffer: BaseReplayBuffer,
        reward_relabel_client: RewardRelabelClient,
        use_relative_rewards: bool = False,
        batch_size: int = 32,  # Number of transitions to batch before sending
        obs_keys: List[str] = None,
        remove_obs_keys: List[str] = None,
        rename_obs_keys: Dict[str, str] = None,
        pre_transforms: List[Callable] = None,
        post_transforms: List[Callable] = None,
        sampler=None,
    ):
        # Initialize base class
        super().__init__(
            obs_keys=obs_keys,
            remove_obs_keys=remove_obs_keys,
            rename_obs_keys=rename_obs_keys,
            pre_transforms=pre_transforms or [],
            post_transforms=post_transforms or [],
            sampler=sampler,
        )

        self.underlying_buffer = underlying_buffer
        self.client = reward_relabel_client
        self.use_relative_rewards = use_relative_rewards
        self.batch_size = batch_size

        # Track previous rewards for relative reward computation
        self.prev_rewards: Dict[Any, float] = {}  # episode_id -> previous reward

        # Full trajectory history per episode (keep until episode terminates)
        self._episode_transitions: Dict[Any, List[PendingRelabel]] = {}  # episode_id -> full trajectory
        self._episode_language_instructions: Dict[Any, str] = {}  # episode_id -> language instruction

        # Pending transitions waiting to be sent for relabeling
        self._pending_batch_indices: Dict[Any, int] = {}  # episode_id -> next index to send
        self._lock = threading.Lock()

        # Start client if not already started
        if not self.client.running:
            self.client.start()

        # Statistics
        self.stats = {
            "batches_queued": 0,
            "transitions_queued": 0,
            "transitions_added": 0,
        }

    def _on_batch_relabeled(
        self, relabeled_rewards: List[float], success_probs: List[float], batch: PendingRelabelBatch
    ):
        """Callback when batch reward relabeling is complete."""
        # batch.transitions contains only the batch transitions (not full trajectory context)
        if len(relabeled_rewards) != len(batch.transitions):
            logger.error(
                f"[AsyncRewardRelabelBuffer] Mismatch: {len(relabeled_rewards)} rewards "
                f"for {len(batch.transitions)} transitions"
            )
            # Use original rewards
            relabeled_rewards = [p.reward for p in batch.transitions]
            success_probs = [0.0] * len(relabeled_rewards)

        # Add all transitions in batch to underlying buffer with relabeled rewards
        episode_id = batch.episode_id
        prev_reward = 0.0

        with self._lock:
            # Get previous reward for relative reward computation
            if self.use_relative_rewards and episode_id in self.prev_rewards:
                prev_reward = self.prev_rewards[episode_id]

        for i, pending in enumerate(batch.transitions):
            reward = relabeled_rewards[i]

            # Apply relative rewards if enabled
            if self.use_relative_rewards:
                current_reward = reward
                reward = reward - prev_reward
                prev_reward = current_reward

            # Add to underlying buffer with relabeled reward
            self.underlying_buffer.add(
                obs=pending.obs,
                action=pending.action,
                reward=reward,
                next_obs=pending.next_obs,
                done=pending.done,
                truncated=pending.truncated,
                episode_id=pending.episode_id,
                step_in_episode=pending.step_in_episode,
                timestamp=pending.timestamp,
            )

            self.stats["transitions_added"] += 1

        # Update previous reward and pending batch index
        batch_size = len(batch.transitions)
        with self._lock:
            if self.use_relative_rewards:
                self.prev_rewards[episode_id] = prev_reward

            # Update the next index to send (these transitions have been processed)
            if episode_id in self._pending_batch_indices:
                self._pending_batch_indices[episode_id] += batch_size

            # Clean up episode tracking if episode is done and all transitions processed
            if episode_id in self._episode_transitions:
                full_trajectory = self._episode_transitions[episode_id]
                next_index = self._pending_batch_indices.get(episode_id, 0)
                # Check if episode is done and all transitions have been processed
                if len(full_trajectory) > 0:
                    last_transition = full_trajectory[-1]
                    if (last_transition.done or last_transition.truncated) and next_index >= len(full_trajectory):
                        # All transitions processed and episode is done, clean up
                        self._episode_transitions.pop(episode_id, None)
                        self._episode_language_instructions.pop(episode_id, None)
                        self._pending_batch_indices.pop(episode_id, None)
                        if self.use_relative_rewards:
                            self.prev_rewards.pop(episode_id, None)

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
        language_instruction=None,
        **kwargs,
    ):
        """Add transition - batch and send every N transitions or when episode terminates."""
        # Create pending transition (data is already in obs/next_obs)
        pending = PendingRelabel(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            truncated=truncated,
            episode_id=episode_id,
            step_in_episode=step_in_episode,
            timestamp=timestamp,
            language_instruction=language_instruction,
        )

        with self._lock:
            # Initialize episode tracking if needed
            if episode_id not in self._episode_transitions:
                self._episode_transitions[episode_id] = []
                self._pending_batch_indices[episode_id] = 0
                if language_instruction:
                    self._episode_language_instructions[episode_id] = language_instruction

            # Add transition to full trajectory history
            self._episode_transitions[episode_id].append(pending)

            # Update episode-level data if provided
            if language_instruction:
                self._episode_language_instructions[episode_id] = language_instruction

            # Check if we should send a batch for relabeling
            full_trajectory = self._episode_transitions[episode_id]
            next_index = self._pending_batch_indices[episode_id]
            num_pending = len(full_trajectory) - next_index

            should_send_batch = False
            if done or truncated:
                # Episode terminated, send remaining transitions
                if num_pending > 0:
                    should_send_batch = True
            elif num_pending >= self.batch_size:
                # Have enough transitions for a batch
                should_send_batch = True

            if should_send_batch:
                # Determine how many transitions to send in this batch
                if done or truncated:
                    # Send all remaining transitions
                    batch_size = num_pending
                else:
                    # Send exactly batch_size transitions
                    batch_size = self.batch_size

                # Get the batch of transitions to send
                batch_transitions = full_trajectory[next_index : next_index + batch_size]

                # For reward model context, we need to send the full trajectory up to the last transition in the batch
                # The server will create subsequences [0:1], [0:2], ..., [0:t] for transitions at indices 0, 1, ..., t
                # So we send all transitions from 0 to (next_index + batch_size - 1)
                trajectory_context = full_trajectory[: next_index + batch_size]

                lang_instruction = self._episode_language_instructions.get(episode_id)

                # Create batch with full trajectory context
                batch = PendingRelabelBatch(
                    transitions=trajectory_context,  # Full context for reward model
                    batch_start_idx=next_index,  # Start index of batch within trajectory
                    batch_end_idx=next_index + batch_size,  # End index of batch
                    episode_id=episode_id,
                    language_instruction=lang_instruction,
                    callback=self._on_batch_relabeled,
                )

                # Send batch for relabeling
                self.client.relabel_batch(batch)
                self.stats["batches_queued"] += 1
                self.stats["transitions_queued"] += batch_size

    # Delegate all other methods to underlying buffer
    def get_all_transitions(self) -> List[Transition]:
        return self.underlying_buffer.get_all_transitions()

    def __len__(self):
        return len(self.underlying_buffer)

    def size(self):
        return self.underlying_buffer.size()

    def is_empty(self):
        return self.underlying_buffer.is_empty()

    def clear(self):
        self.underlying_buffer.clear()
        with self._lock:
            self._episode_transitions.clear()
            self._episode_language_instructions.clear()
            self._pending_batch_indices.clear()
            self.prev_rewards.clear()

    def stop(self):
        """Stop the reward relabeling client."""
        self.client.stop()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about remote relabeling."""
        client_stats = self.client.get_stats()
        return {
            **self.stats,
            **client_stats,
        }
