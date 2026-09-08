import torch
import abc
import gymnasium as gym
from torch import nn
from typing import Callable, Union, List
from dataclasses import dataclass


@dataclass
class BaseActorConfig(abc.ABC):
    """
    A base class for all actor configurations.
    """

    observation_space: gym.Space = None
    action_space: gym.Space = None
    feature_extractor: nn.Module = None
    preprocess_obs_transform: Callable = None
    min_action: Union[float, torch.Tensor] = None
    max_action: Union[float, torch.Tensor] = None
    remove_obs_keys: List[str] = None

    @property
    def actor_class(self):
        # Import BaseActor here to avoid circular import
        from robometer_policy_learning.modules.base import BaseActor

        return BaseActor

    def create(self):
        # Import BaseActor here to avoid circular import
        from robometer_policy_learning.modules.base import BaseActor

        actor_class = self.actor_class
        if actor_class is None:
            raise ValueError(f"actor_class not defined for {self.__class__.__name__}")
        return actor_class(self)
