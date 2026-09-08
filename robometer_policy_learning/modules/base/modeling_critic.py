import torch
import torch.nn as nn
from typing import Dict
from robometer_policy_learning.modules.base import BaseCriticConfig


class BaseCritic(nn.Module):
    """
    A base class for all critics.
    """

    def __init__(self, config: BaseCriticConfig):
        super().__init__()
        self.config = config
        self.remove_obs_keys = config.remove_obs_keys
        self.preprocess_obs_transform = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def _remove_obs_keys(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove the observation keys specified in the config."""
        if self.remove_obs_keys is None:
            return obs
        for key in self.remove_obs_keys:
            obs.pop(key, None)
        return obs
