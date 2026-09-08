import abc
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
from torch import nn

from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.loggers.logger import Logger


@dataclass
class BaseAlgorithmConfig(abc.ABC):
    """
    A configuration algorithm is an algorithm that takes in a buffer and a reward model and returns a configuration.
    """

    # Runtime fields (set at runtime, not from config) - use Any to avoid Hydra validation
    env: Optional[Any] = None
    actor: Optional[Any] = None
    critic: Optional[Any] = None
    buffer: Optional[Any] = None
    logger: Optional[Any] = None

    @property
    def algorithm_class(self):
        # Import BaseActor here to avoid circular import
        from robometer_policy_learning.algorithms import BaseAlgorithm

        return BaseAlgorithm

    def create(self):
        # Import BaseActor here to avoid circular import
        from robometer_policy_learning.algorithms import BaseAlgorithm

        algorithm_class = self.algorithm_class
        if algorithm_class is None:
            raise ValueError(f"algorithm_class not defined for {self.__class__.__name__}")
        return algorithm_class(self)
