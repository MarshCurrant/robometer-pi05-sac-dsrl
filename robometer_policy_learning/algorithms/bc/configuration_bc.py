from typing import Union, Any, Optional

import torch
from dataclasses import dataclass

from robometer_policy_learning.algorithms.configuration_algorithm import BaseAlgorithmConfig

import gymnasium as gym
from torch import nn

from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer


@dataclass
class BCConfig(BaseAlgorithmConfig):
    """
    A configuration for the BC (Behavior Cloning) algorithm.
    """

    # Runtime fields (inherited from BaseAlgorithmConfig)
    action_space: Optional[Any] = None

    # Set up BC parameters
    learning_starts: int = 100
    batch_size: int = 256

    # Training configuration
    num_updates_per_train_step: int = 1

    # Optimizer configurations
    actor_optimizer_lr: float = 3e-5
    actor_optimizer_eps: float = 1e-8
    actor_optimizer_weight_decay: float = 0.0

    # BC-specific parameters
    loss_type: str = "nll"  # ["mse", "nll", "huber", "smooth_l1"] - MSE for deterministic, NLL for stochastic
    l2_regularization: float = 0.0  # L2 regularization weight
    use_weighted_bc: bool = False  # Use weighted BC

    # Anti-overfitting regularization parameters
    obs_noise_std: float = 0.0  # Standard deviation of noise added to observations
    action_noise_std: float = 0.0  # Standard deviation of noise added to expert actions
    gradient_penalty_weight: float = 0.0  # Weight for gradient penalty regularization
    consistency_weight: float = 0.0  # Weight for consistency regularization
    clip_grad_norm: float = 10.0  # Clip gradient norm

    @property
    def algorithm_class(self):
        from robometer_policy_learning.algorithms.bc import BC

        return BC
