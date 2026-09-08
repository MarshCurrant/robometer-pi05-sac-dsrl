from typing import Union, Any, Optional

import torch
from dataclasses import dataclass

from robometer_policy_learning.algorithms.configuration_algorithm import BaseAlgorithmConfig

import gymnasium as gym
from torch import nn

from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer


@dataclass
class SACConfig(BaseAlgorithmConfig):
    """
    A configuration for the SAC algorithm.
    """

    # Set up environment, actor, critic, and buffer (runtime fields - use Any to avoid Hydra validation)
    action_space: Optional[Any] = None  # Action space for target entropy calculation

    # Set up SAC parameters
    learning_starts: int = 1000
    action_noise: float = 0.0
    ent_coef: Union[str, float] = "auto"
    target_entropy: Union[str, float] = "auto"
    target_update_interval: int = 1
    gamma: float = 0.99
    compute_chunked_gamma: bool = True  # if true, will use action chunk size to automatically discount gamma properly
    tau: float = 0.005
    pooled_critic_features: bool = True

    # Extra
    batch_size: int = 128
    n_critics_to_sample: int = 2
    num_critics: int = 2  # Number of critics in the ensemble
    critic_reduction: str = "min"  # "mean" or "min"
    num_critic_updates_per_actor_update: int = 4
    num_updates_per_train_step: int = 1
    train_critic_with_entropy: bool = False
    train_actor_with_entropy: bool = True

    # Training presets
    actor_optimizer_lr: float = 3e-4
    actor_optimizer_eps: float = 1e-8
    actor_optimizer_weight_decay: float = 1e-6
    actor_scheduler_name: str = "cosine"
    actor_scheduler_warmup_steps: int = 500

    critic_optimizer_lr: float = 3e-4
    critic_optimizer_betas: tuple = (0.95, 0.999)
    critic_optimizer_eps: float = 1e-8
    critic_optimizer_weight_decay: float = 1e-6
    critic_scheduler_name: str = "cosine"
    critic_scheduler_warmup_steps: int = 500

    ent_coef_lr: float = 3e-4
    ent_coef_update_interval: int = 1

    train_actor_with_kl_divergence: bool = True

    # Optional gradient clipping for stability
    clip_grad_norm: Optional[float] = None  # Set to e.g. 1.0 to enable gradient clipping
    actor_clip_grad_norm: Optional[float] = None
    critic_clip_grad_norm: Optional[float] = None

    @property
    def algorithm_class(self):
        from robometer_policy_learning.algorithms.sac import SAC

        return SAC
