from typing import Union, Any, Optional

import torch
from dataclasses import dataclass

from robometer_policy_learning.algorithms.configuration_algorithm import BaseAlgorithmConfig

import gymnasium as gym
from torch import nn

from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer


@dataclass
class IQLConfig(BaseAlgorithmConfig):
    """
    A configuration for the IQL (Implicit Q-Learning) algorithm.
    """

    # Runtime fields (inherited from BaseAlgorithmConfig, but need to add v_net)
    v_net: Optional[Any] = None  # Value function network
    action_space: Optional[Any] = None

    # Set up IQL parameters
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    compute_chunked_gamma: bool = True  # if true, will use action chunk size to automatically discount gamma properly
    target_update_interval: int = 1
    pooled_critic_features: bool = True

    # IQL-specific parameters
    advantage_temp: float = 2.5  # Temperature for advantage weighting
    expectile: float = 0.7  # Expectile for value function regression
    clip_score: float = 100.0  # Clipping term on the advantage temp
    policy_extraction: str = "awr"  # ["awr", "ddpg"] policy extraction algorithm
    ddpg_bc_weight: float = 0.1  # DDPG's behavior cloning weight (when policy_extraction="ddpg")

    # Training configuration
    num_critics: int = 2  # Number of critics in the ensemble
    n_critics_to_sample: int = 2  # Number of critics to sample from
    num_updates_per_train_step: int = 1
    offline_critic_update_ratio: int = 1  # Ratio for offline critic updates

    # Optimizer configurations
    actor_optimizer_lr: float = 1e-5
    actor_optimizer_eps: float = 1e-8
    actor_optimizer_weight_decay: float = 0.0

    critic_optimizer_lr: float = 4e-4
    critic_optimizer_betas: tuple = (0.9, 0.999)
    critic_optimizer_eps: float = 1e-8
    critic_optimizer_weight_decay: float = 0.0

    v_net_optimizer_lr: float = 3e-4
    v_net_optimizer_betas: tuple = (0.9, 0.999)
    v_net_optimizer_eps: float = 1e-8
    v_net_optimizer_weight_decay: float = 0.0

    @property
    def algorithm_class(self):
        from robometer_policy_learning.algorithms.iql import IQL

        return IQL
