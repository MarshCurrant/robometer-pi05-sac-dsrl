import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, Any
import gymnasium as gym

from robometer_policy_learning.modules.base import BaseActor
from robometer_policy_learning.modules.mlp import MLPActorConfig
from robometer_policy_learning.modules.base.distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
)
from robometer_policy_learning.utils.featurizers import ObservationFeaturizer
from robometer_policy_learning.utils.featurizers import _build_mlp_layers


class MLPActor(BaseActor):
    """
    MLP-based actor
    Supports dict obs flattening and per-key featurizers.
    """

    def __init__(self, config: MLPActorConfig):
        super().__init__(config)
        self.config = config
        self.preprocess_obs_transform = config.preprocess_obs_transform

        # Initialize observation featurizer only if featurizer config is provided
        if config.featurizer is not None or config.image_encoder_type == "impala":
            self.obs_featurizer = ObservationFeaturizer(
                observation_space=config.observation_space,
                featurizer_cfg=config.featurizer,
                activation=config.activation,
                use_layer_norm=config.use_layer_norm,
                dropout_rate=config.dropout_rate,
                image_encoder_type=config.image_encoder_type,
                impala_nn_scale=config.impala_nn_scale,
                impala_num_blocks_per_stack=config.impala_num_blocks_per_stack,
                impala_use_smaller=config.impala_use_smaller,
                impala_output_dim=config.impala_output_dim,
            )
            # Input dimension - compute by passing example observation through featurizer
            obs_dim = self.obs_featurizer.output_dim
            if obs_dim <= 0:
                raise ValueError("ObservationFeaturizer produced invalid output dimension for actor.")
            self._flatten_keys = None
        else:
            # No featurizer: calculate obs_dim directly from observation_space
            self.obs_featurizer = None
            if isinstance(config.observation_space, gym.spaces.Dict):
                self._flatten_keys = [
                    k
                    for k, space in config.observation_space.spaces.items()
                    if getattr(space, "shape", None) is not None
                ]
                obs_dim = sum(int(np.prod(config.observation_space.spaces[k].shape)) for k in self._flatten_keys)
            else:
                self._flatten_keys = None
                obs_dim = int(np.prod(config.observation_space.shape))
            if obs_dim <= 0:
                raise ValueError(f"Invalid observation dimension calculated from observation_space: {obs_dim}")

        if self.is_continuous:
            action_dim = int(np.prod(config.action_space.shape))
        else:  # discrete
            action_dim = config.action_space.n

        # Build MLP layers
        if obs_dim is not None:
            self.mlp = self._build_mlp(obs_dim)
        else:
            self.mlp = None  # Will build dynamically in forward if needed

        final_hidden_dim = int(config.hidden_dims[-1])
        self.hidden_dim = final_hidden_dim

        if self.is_continuous:
            # Gaussian policy: mean and log_std
            self.mean_layer = nn.Linear(final_hidden_dim, action_dim)
            self.log_std_layer = nn.Linear(final_hidden_dim, action_dim)

            # Initialize log std
            self.log_std_layer.weight.data.fill_(0.0)
            self.log_std_layer.bias.data.fill_(config.log_std_init)
            if config.use_tanh_output:
                self.action_dist = SquashedDiagGaussianDistribution(action_dim=action_dim)
            else:
                self.action_dist = DiagGaussianDistribution(action_dim=action_dim)
        else:
            # Discrete policy: logits
            self.logits_layer = nn.Linear(final_hidden_dim, action_dim)
            self.action_dist = CategoricalDistribution(action_dim=action_dim)

    def _build_mlp(self, input_size: int) -> nn.Module:
        layers = _build_mlp_layers(
            input_size,
            self.config.hidden_dims,
            self.config.activation,
            self.config.use_layer_norm,
            self.config.dropout_rate,
        )
        return nn.Sequential(*layers)

    def _flatten_obs(self, obs: Union[dict, torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        if self.obs_featurizer is not None:
            return self.obs_featurizer.flatten_obs(obs, device=device)
        else:
            # Manual flattening when no featurizer
            if isinstance(obs, dict):
                # Infer batch size from any clearly-batched tensor (dim >= 2)
                batch_size = None
                for k in self._flatten_keys:
                    v = obs.get(k)
                    if torch.is_tensor(v) and v.dim() >= 2:
                        batch_size = int(v.size(0))
                        break
                if batch_size is None:
                    batch_size = 1
                feats = []
                for k in self._flatten_keys:
                    v = obs.get(k)
                    if v is not None:
                        # Handle common scalar-per-batch tensors: shape (B,) should become (B, 1),
                        # not (1, B) (which silently swaps batch/features).
                        if torch.is_tensor(v):
                            if v.dim() == 0:
                                v = v.view(1, 1)
                            elif v.dim() == 1:
                                v = v.unsqueeze(-1) if int(v.size(0)) == batch_size else v.unsqueeze(0)
                        v_flat = v.reshape(v.size(0), -1) if torch.is_tensor(v) and v.dim() > 1 else v
                        feats.append(v_flat.to(device))
                return torch.cat(feats, dim=-1) if feats else torch.empty(0, device=device)
            else:
                # Tensor observation
                if obs.dim() > 2:
                    return obs.reshape(obs.size(0), -1).to(device)
                elif obs.dim() == 1:
                    return obs.unsqueeze(0).to(device)
                else:
                    return obs.to(device)

    def _act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if self.preprocess_obs_transform is not None:
            for transform in self.preprocess_obs_transform:
                obs = transform(obs)
        obs_flat = self._flatten_obs(obs)
        if self.mlp is not None:
            hidden = self.mlp(obs_flat.float())
        else:
            # If using featurizer, build MLP dynamically
            hidden = obs_flat
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, hidden=hidden)
        if log_std is not None:
            action = self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic)
        else:
            action = self.action_dist.actions_from_params(mean_actions, deterministic=deterministic)
        return action

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action given observation."""
        if self.config.deterministic:
            raise ValueError("Cannot compute log_prob for deterministic policy")
        obs_flat = self._flatten_obs(obs)
        if self.mlp is not None:
            hidden = self.mlp(obs_flat)
        else:
            hidden = obs_flat
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, hidden=hidden)
        if self.is_continuous:
            self.action_dist.proba_distribution(mean_actions, log_std)
            return self.action_dist.log_prob(action)
        else:
            self.action_dist.proba_distribution(mean_actions)
            return self.action_dist.log_prob(action)

    def _forward(
        self, obs: torch.Tensor, actor_state: Any = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        obs_flat = self._flatten_obs(obs)
        if self.mlp is not None:
            hidden = self.mlp(obs_flat)
        else:
            hidden = obs_flat
        if self.config.deterministic:
            output = self.output_layer(hidden)
            if self.is_continuous and self.config.use_tanh_output:
                output = torch.tanh(output)
            return output
        else:
            if self.is_continuous:
                mean = self.mean_layer(hidden)
                log_std = self.log_std_layer(hidden)
                log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)
                return mean, log_std
            else:
                logits = self.logits_layer(hidden)
                return logits

    def get_hidden_state(self, obs: torch.Tensor) -> torch.Tensor:
        # Used in cases where the actor's structure is used as a feature extractor
        obs_flat = self._flatten_obs(obs)
        if self.mlp is not None:
            hidden = self.mlp(obs_flat)
        else:
            hidden = obs_flat
        return hidden

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def get_action_dist_params(self, obs, hidden=None):
        # If hidden is provided, use it; otherwise, flatten obs and pass through MLP/featurizer
        if hidden is None:
            obs_flat = self._flatten_obs(obs)
            if self.mlp is not None:
                hidden = self.mlp(obs_flat)
            else:
                hidden = obs_flat
        if self.is_continuous:
            mean = self.mean_layer(hidden)
            log_std = self.log_std_layer(hidden)
            log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)
            kwargs = {}
            return mean, log_std, kwargs
        else:
            logits = self.logits_layer(hidden)
            return logits, None, {}
