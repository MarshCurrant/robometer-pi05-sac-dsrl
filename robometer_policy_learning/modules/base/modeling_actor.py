import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import abc
from typing import Union, Tuple, Any, Dict

from robometer_policy_learning.modules.base import BaseActorConfig
import inspect


class BaseActor(nn.Module):
    """
    A base class for all actors with action normalization support.
    Uses Template Method pattern - subclasses implement _forward(), BaseActor manages act().
    """

    def __init__(self, config: BaseActorConfig):
        super().__init__()
        self.config = config
        self.action_space = config.action_space
        self.remove_obs_keys = config.remove_obs_keys

        self.preprocess_obs_transform = None

        # Determine action space type
        self.is_continuous = isinstance(self.action_space, gym.spaces.Box)
        self.is_discrete = isinstance(self.action_space, gym.spaces.Discrete)

        if not (self.is_continuous or self.is_discrete):
            raise ValueError(f"Unsupported action space type: {type(self.action_space)}")

        # Setup action normalization parameters
        if self.action_space is not None:
            if isinstance(self.action_space, gym.spaces.Box):
                # if the action space is inf/-inf, we do not normalize for now
                # TODO: allow users to specify a normalization range
                if np.isinf(self.action_space.low).any() or np.isinf(self.action_space.high).any():
                    print("action space is inf/-inf, not normalizing")
                    self.normalize_actions = False
                else:
                    self.action_low = torch.tensor(self.action_space.low, dtype=torch.float32)
                    self.action_high = torch.tensor(self.action_space.high, dtype=torch.float32)
                    self.normalize_actions = True
            elif isinstance(self.action_space, gym.spaces.Discrete):
                self.normalize_actions = False
            else:
                self.normalize_actions = False
        else:
            self.normalize_actions = False

        # But if config.action_low and config.action_high are provided, we use them and override the action space
        if self.config.min_action is not None and self.config.max_action is not None:
            self.action_low = torch.tensor(self.config.min_action, dtype=torch.float32)
            self.action_high = torch.tensor(self.config.max_action, dtype=torch.float32)
            self.normalize_actions = True

        # action_dist will be set by subclasses as instances, not classes
        self.action_dist = None

    def _move_action_tensors_to_device(self, device):
        """Move action normalization tensors to the specified device."""
        if self.normalize_actions:
            self.action_low = self.action_low.to(device)
            self.action_high = self.action_high.to(device)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize action from [action_low, action_high] to [-, 1]."""
        if not self.normalize_actions:
            return action

        device = action.device
        self._move_action_tensors_to_device(device)

        normalized = (action - self.action_low) / (self.action_high - self.action_low)
        normalized = normalized * 2.0 - 1.0
        return torch.clamp(normalized, -1.0, 1.0)

    def unnormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """Unnormalize action from [-1, 1] to [action_low, action_high]."""
        if not self.normalize_actions:
            return normalized_action

        device = normalized_action.device
        self._move_action_tensors_to_device(device)

        # is currently between -1 and 1. scale to action_low to action_high
        return (normalized_action + 1) / 2 * (self.action_high - self.action_low) + self.action_low

    def forward(self, obs, actor_state: Any = None):
        """Standard forward pass - delegates to _forward"""
        # Inspect forward, and if it does not have actor_state as an argument,
        # then we ignore actor_state
        if "actor_state" not in inspect.signature(self._forward).parameters:
            return self._forward(obs)
        else:
            return self._forward(obs, actor_state=actor_state)

    def _forward(self, obs: torch.Tensor):
        """
        Subclasses should implement this method for their specific forward pass.
        Should return actions in a consistent range (e.g., [-1, 1] if using tanh).
        """
        raise NotImplementedError("Subclasses must implement _forward method")

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False, actor_state: Any = None) -> torch.Tensor:
        """
        Generate actions ready for environment interaction (in original action space).

        Note: Wrapped in @torch.no_grad() since rollout actions don't need gradients.
        """
        was_training = self.training
        if was_training:
            self.eval()
        obs = self._remove_obs_keys(obs)

        # Get raw action from subclass implementation (in [-1, 1] range)
        if "actor_state" not in inspect.signature(self._act).parameters:
            action = self._act(obs, deterministic=deterministic)
            actor_state = None
        else:
            action, actor_state = self._act(obs, deterministic=deterministic, actor_state=actor_state)

        # Convert to environment action space if normalization is enabled
        if self.normalize_actions:
            action = self.unnormalize_action(action)

        if was_training:
            self.train()

        return action, actor_state

    def get_action_dist_params(self, obs):
        """Get the parameters of the action distribution."""
        raise NotImplementedError("Subclasses must implement get_action_dist_params")

    def action_log_prob(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions and compute log probabilities using the action distribution.
        Returns (action, log_prob) tuple for SAC training.
        Actions are returned in [-1, 1] range for internal SAC training consistency.
        """
        if self.action_dist is None:
            raise ValueError("action_dist not initialized by subclass")

        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Return action and associated log prob
        # If we use discrete action space, log_std is None
        if log_std is not None:
            return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
        else:
            return self.action_dist.log_prob_from_params(mean_actions, **kwargs)

    def get_initial_state(self):
        """Get the initial state of the actor.
        This is used to get the initial actor state to pass into the rollout worker
        for cases such as passing the hidden state of an RNN actor.
        By default, we return None.
        """
        return None

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def _remove_obs_keys(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove the observation keys specified in the config."""
        if self.remove_obs_keys is None:
            return obs
        for key in self.remove_obs_keys:
            obs.pop(key, None)
        return obs
