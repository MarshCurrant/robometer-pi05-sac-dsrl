import torch
import abc
import gymnasium as gym
from typing import List
from torch import nn
from dataclasses import dataclass
from typing import Callable


@dataclass
class BaseCriticConfig(abc.ABC):
    """
    A base class for all critic configurations.
    """

    observation_space: gym.Space = None
    action_space: gym.Space = None
    feature_extractor: nn.Module = None
    preprocess_obs_transform: Callable = None
    remove_obs_keys: List[str] = None

    #  Whether this critic uses actions (Q-function vs V-function)
    use_action: bool = True  # Whether to concatenate obs and action as input

    @property
    def critic_class(self):
        # Import BaseActor here to avoid circular import
        from robometer_policy_learning.modules.base import BaseCritic

        return BaseCritic

    def create(self):
        # Import BaseActor here to avoid circular import
        from robometer_policy_learning.modules.base import BaseCritic

        critic_class = self.critic_class
        if critic_class is None:
            raise ValueError(f"critic_class not defined for {self.__class__.__name__}")
        return critic_class(self)
