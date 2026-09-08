"""
Transition-level transforms for modifying individual transitions.

These transforms operate on single Transition objects and can be applied
during sampling to modify rewards, observations, or other transition data.
"""

import numpy as np
from typing import Any
from robometer_policy_learning.buffers.base_replay_buffer import Transition


class TransitionTransform:
    """
    Base class for transforms that operate on individual transitions.
    This is more flexible than batch-level transforms.
    """

    def __call__(self, transition: Transition) -> Transition:
        """Apply transform to a single transition."""
        raise NotImplementedError

    # By default, transforms do not support batched input
    supports_batch = False


class MonotonicRewardTransform(TransitionTransform):
    """
    Transform that relabels rewards to be monotonically increasing from 0 to 1
    based on progress through the episode.

    This is useful for sparse reward environments where you want to provide
    dense reward signal based on temporal progress.
    """

    def __init__(
        self,
        mode: str = "linear",  # "linear", "quadratic", "exponential"
        success_bonus: float = 0.0,  # Additional bonus for successful episodes
        use_success_only: bool = False,
    ):  # Only apply to successful episodes
        """
        Args:
            mode: Type of monotonic increase ("linear", "quadratic", "exponential")
            success_bonus: Additional reward bonus for the final step of successful episodes
            use_success_only: If True, only apply monotonic rewards to successful episodes
        """
        self.mode = mode
        self.success_bonus = success_bonus
        self.use_success_only = use_success_only

    def __call__(self, transition: Transition) -> Transition:
        """
        Apply monotonic reward relabeling to a single transition.

        Args:
            transition: Input transition

        Returns:
            Modified transition with relabeled reward
        """
        if transition.max_steps_in_episode is None or transition.step_in_episode is None:
            # Can't apply monotonic reward without episode length info
            return transition

        # Check if this is a successful episode (if filtering is enabled)
        if self.use_success_only:
            # Assume success if the episode ends with done=True and reward > 0
            # You might want to customize this logic based on your environment
            is_successful = transition.done and transition.reward > 0
            if not is_successful:
                return transition

        # Calculate progress through episode (0 to 1)
        progress = transition.step_in_episode / (transition.max_steps_in_episode - 1)
        # if transition.done:
        #     print(transition.step_in_episode, transition.max_steps_in_episode)
        #     breakpoint()
        # if transition.max_steps_in_episode == (transition.step_in_episode - 1):
        #     breakpoint()
        progress = np.clip(progress, 0.0, 1.0)

        # Apply monotonic transformation
        if self.mode == "linear":
            new_reward = progress
        elif self.mode == "quadratic":
            new_reward = progress**2
        elif self.mode == "exponential":
            # Exponential growth: e^(progress) - 1, normalized to [0, 1]
            new_reward = (np.exp(progress) - 1) / (np.e - 1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Add success bonus for final step if applicable
        if transition.done and self.success_bonus > 0:
            new_reward += self.success_bonus

        # Create new transition with modified reward using replace helper
        return transition.replace(reward=new_reward)

    # This transform can be applied in batch by falling back to per-transition
    supports_batch = False


class SuccessBonusTransform(TransitionTransform):
    """
    Transform that adds a bonus reward to successful episodes.

    This is equivalent to the success_bonus function but implemented
    as a transition-level transform for consistency.
    """

    def __init__(self, bonus_value: float = 10.0, debug: bool = False):
        """
        Args:
            bonus_value: Bonus reward to add to successful episodes
            debug: Whether to print debug information about reward transformations
        """
        self.bonus_value = bonus_value
        self.debug = debug

    def __call__(self, transition: Transition) -> Transition:
        """
        Add success bonus to successful episodes.

        Args:
            transition: Input transition

        Returns:
            Modified transition with success bonus
        """
        # Add bonus to successful episodes (done=True)
        if transition.done:
            new_reward = transition.reward + self.bonus_value
            if self.debug:
                print(
                    f"SuccessBonusTransform: original_reward={transition.reward}, bonus={self.bonus_value}, new_reward={new_reward}"
                )
        else:
            new_reward = transition.reward

        # Create new transition with modified reward using replace helper
        return transition.replace(reward=new_reward)

    # This transform can be applied in batch by falling back to per-transition
    supports_batch = False

    def __repr__(self):
        return f"SuccessBonusTransform(bonus_value={self.bonus_value}, debug={self.debug})"


# Legacy batch-level transform functions for backward compatibility
def success_bonus(bonus_value):
    """
    Legacy success bonus transform that now works with Transition objects.

    This is kept for backward compatibility with existing code.
    For new code, consider using SuccessBonusTransform directly.
    """

    def _success_bonus_transform(transition: Transition) -> Transition:
        # Add bonus to successful episodes (done=True)
        if transition.done:
            new_reward = transition.reward + bonus_value
        else:
            new_reward = transition.reward

        # Create new transition with modified reward using replace helper
        return transition.replace(reward=new_reward)

    return _success_bonus_transform
