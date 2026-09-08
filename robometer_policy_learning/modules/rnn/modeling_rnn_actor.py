import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, Any, Dict, List
import gymnasium as gym

from robometer_policy_learning.modules.base import BaseActor
from robometer_policy_learning.modules.rnn import RNNActorConfig
from robometer_policy_learning.modules.base.distributions import (
    CategoricalDistribution,
    SquashedDiagGaussianDistribution,
)


def _build_mlp_layers(input_size, hidden_dims, activation, use_layer_norm=False, dropout_rate=0.0):
    """Build MLP layers (same as in MLP actor)."""
    layers = []
    prev_size = int(input_size)
    for hidden_size in hidden_dims:
        hidden_size = int(hidden_size)
        layers.append(nn.Linear(prev_size, hidden_size))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        if activation.lower() == "relu":
            layers.append(nn.ReLU())
        elif activation.lower() == "tanh":
            layers.append(nn.Tanh())
        elif activation.lower() == "elu":
            layers.append(nn.ELU())
        elif activation.lower() == "leaky_relu":
            layers.append(nn.LeakyReLU())
        else:
            raise ValueError(f"Unknown activation: {activation}")
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        prev_size = hidden_size
    return nn.Sequential(*layers)


class RNNActor(BaseActor):
    """
    RNN-based actor for sequential decision making.

    Architecture:
    obs -> [feature_mlp] -> RNN -> [output_mlp] -> action_head

    Supports:
    - Training with chunks (sequences of actions)
    - Inference with single observations + actor_state
    - LSTM, GRU, and vanilla RNN
    - Dict observations with featurizers
    """

    def __init__(self, config: RNNActorConfig):
        super().__init__(config)
        self.config = config
        self.featurizer_cfg = config.featurizer if config.featurizer is not None else {}
        self.preprocess_obs_transform = config.preprocess_obs_transform

        # Build featurizers for dict observations (same as MLP actor)
        # If image_encoder_type is "impala", build IMPALA featurizers for image keys
        if config.image_encoder_type == "impala":
            from robometer_policy_learning.utils.featurizers import build_mlp_impala_featurizers, identify_image_keys

            if not isinstance(config.observation_space, gym.spaces.Dict):
                raise ValueError("IMPALA encoder requires Dict observation space")

            # Build IMPALA featurizers for image keys
            impala_featurizers = build_mlp_impala_featurizers(
                observation_space=config.observation_space,
                keys=None,  # Auto-detect image keys
                nn_scale=config.impala_nn_scale,
                num_blocks_per_stack=config.impala_num_blocks_per_stack,
                use_smaller=config.impala_use_smaller,
                requires_grad=True,
                impala_output_dim=config.impala_output_dim,
            )

            # Merge with existing featurizer_cfg (featurizer_cfg takes precedence)
            image_keys = identify_image_keys(list(config.observation_space.spaces.keys()))
            for key in image_keys:
                if key not in self.featurizer_cfg:
                    self.featurizer_cfg[key] = impala_featurizers[key]
            # For non-image keys, use default MLP if not specified
            for key in config.observation_space.spaces:
                if key not in self.featurizer_cfg:
                    self.featurizer_cfg[key] = [256]  # Default MLP featurizer

        self.featurizers = nn.ModuleDict() if self.featurizer_cfg else None

        if self.featurizer_cfg:
            for key, value in self.featurizer_cfg.items():
                if isinstance(value, (list, tuple)):
                    obs_dim = int(np.prod(config.observation_space.spaces[key].shape))
                    self.featurizers[key] = _build_mlp_layers(
                        obs_dim,
                        value,
                        config.activation,
                        config.use_layer_norm,
                        config.dropout_rate,
                    )
                elif isinstance(value, nn.Module):
                    self.featurizers[key] = value
                else:
                    raise ValueError(f"Featurizer for key {key} must be list/tuple or nn.Module")

        # Determine input dimension to RNN
        if self.featurizer_cfg:
            self._flatten_keys = None
            # sum over featurizer cfg values
            obs_dim = sum(sum(value) for value in self.featurizer_cfg.values())
        elif isinstance(config.observation_space, gym.spaces.Dict):
            self._flatten_keys = [
                k for k, space in config.observation_space.spaces.items() if getattr(space, "shape", None) is not None
            ]
            obs_dim = sum(int(np.prod(config.observation_space.spaces[k].shape)) for k in self._flatten_keys)
        else:
            self._flatten_keys = None
            obs_dim = int(np.prod(config.observation_space.shape))

        # Build feature extraction MLP (before RNN)
        if config.feature_hidden_dims:
            self.feature_mlp = _build_mlp_layers(
                obs_dim,
                config.feature_hidden_dims,
                config.activation,
                config.use_layer_norm,
                config.dropout_rate,
            )
            rnn_input_size = config.feature_hidden_dims[-1]
        else:
            self.feature_mlp = None
            rnn_input_size = obs_dim  # Will be set dynamically if None

        # Build RNN
        self.rnn_input_size = rnn_input_size
        if rnn_input_size is not None:
            self.rnn = self._build_rnn(rnn_input_size)
        else:
            self.rnn = None  # Will build dynamically

        # RNN output size
        rnn_output_size = config.rnn_hidden_size * 2 if config.rnn_bidirectional else config.rnn_hidden_size

        # Build output MLP (after RNN)
        if config.output_hidden_dims:
            self.output_mlp = _build_mlp_layers(
                rnn_output_size,
                config.output_hidden_dims,
                config.activation,
                config.use_layer_norm,
                config.dropout_rate,
            )
            final_hidden_dim = config.output_hidden_dims[-1]
        else:
            self.output_mlp = None
            final_hidden_dim = rnn_output_size

        # Action heads
        if self.is_continuous:
            action_dim = int(np.prod(config.action_space.shape))
            self.mean_layer = nn.Linear(final_hidden_dim, action_dim)
            self.log_std_layer = nn.Linear(final_hidden_dim, action_dim)

            # Initialize log std
            self.log_std_layer.weight.data.fill_(0.0)
            self.log_std_layer.bias.data.fill_(config.log_std_init)
            self.action_dist = SquashedDiagGaussianDistribution(action_dim=action_dim)
        else:
            action_dim = config.action_space.n
            self.logits_layer = nn.Linear(final_hidden_dim, action_dim)
            self.action_dist = CategoricalDistribution(action_dim=action_dim)

    def _build_rnn(self, input_size: int) -> nn.Module:
        """Build the RNN module."""
        if self.config.rnn_type == "LSTM":
            return nn.LSTM(
                input_size=input_size,
                hidden_size=self.config.rnn_hidden_size,
                num_layers=self.config.rnn_num_layers,
                dropout=self.config.rnn_dropout if self.config.rnn_num_layers > 1 else 0,
                bidirectional=self.config.rnn_bidirectional,
                batch_first=True,
            )
        elif self.config.rnn_type == "GRU":
            return nn.GRU(
                input_size=input_size,
                hidden_size=self.config.rnn_hidden_size,
                num_layers=self.config.rnn_num_layers,
                dropout=self.config.rnn_dropout if self.config.rnn_num_layers > 1 else 0,
                bidirectional=self.config.rnn_bidirectional,
                batch_first=True,
            )
        elif self.config.rnn_type == "RNN":
            return nn.RNN(
                input_size=input_size,
                hidden_size=self.config.rnn_hidden_size,
                num_layers=self.config.rnn_num_layers,
                dropout=self.config.rnn_dropout if self.config.rnn_num_layers > 1 else 0,
                bidirectional=self.config.rnn_bidirectional,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown RNN type: {self.config.rnn_type}")

    def _flatten_obs(self, obs: Union[dict, torch.Tensor]) -> torch.Tensor:
        """Flatten observations (same logic as MLP actor)."""
        if isinstance(obs, dict):
            if self.featurizer_cfg:
                feats = []
                for k, v in obs.items():
                    if k in self.featurizers:
                        v_flat = v.view(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0)
                        feats.append(self.featurizers[k](v_flat))
                    else:
                        v_flat = v.view(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0)
                        feats.append(v_flat)
                if feats:
                    return torch.cat(feats, dim=-1)
                else:
                    raise ValueError(
                        f"No valid features found in observation dict. "
                        f"Keys present: {list(obs.keys())}, expected: {self._flatten_keys}"
                    )
            elif self._flatten_keys is not None and len(self._flatten_keys) > 0:
                feats = []
                batch_size = None
                for k in self._flatten_keys:
                    v = obs.get(k, None)
                    shape = self.config.observation_space.spaces[k].shape
                    flat_dim = int(np.prod(shape))
                    if v is None:
                        if batch_size is None:
                            for vv in obs.values():
                                if vv is not None:
                                    batch_size = vv.size(0) if vv.dim() > 1 else 1
                                    break
                            if batch_size is None:
                                batch_size = 1
                        feats.append(
                            torch.zeros(
                                batch_size,
                                flat_dim,
                                device=next(self.parameters()).device,
                            )
                        )
                    else:
                        v_flat = v.view(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0)
                        feats.append(v_flat)

                return torch.cat(feats, dim=-1) if feats else torch.empty(0)
            else:
                feats = [
                    v.view(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0) for v in obs.values() if v is not None
                ]
                return torch.cat(feats, dim=-1) if feats else torch.empty(0)
        else:
            if obs.dim() > 2:
                return obs.view(obs.size(0), -1)
            elif obs.dim() == 1:
                return obs.unsqueeze(0)
            return obs

    def _process_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """Process observations through feature extraction."""
        if self.preprocess_obs_transform is not None:
            for transform in self.preprocess_obs_transform:
                obs = transform(obs)

        obs_flat = self._flatten_obs(obs)

        # Build RNN dynamically if needed
        if self.rnn is None:
            self.rnn_input_size = obs_flat.size(-1)
            self.rnn = self._build_rnn(self.rnn_input_size)

        # Apply feature MLP if present
        if self.feature_mlp is not None:
            obs_flat = self.feature_mlp(obs_flat)

        return obs_flat

    def _forward(
        self, obs: torch.Tensor, actor_state: Any = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for training with chunks.

        Args:
            obs: Observations of shape [batch_size, obs_dim]
            actor_state: Not used in training mode

        Returns:
            Action distribution parameters
        """
        # Handle both single obs and sequences
        if isinstance(obs, dict):
            # Add sequence dimension to dict obs
            obs = {k: v.unsqueeze(1) if v.dim() == 2 else v for k, v in obs.items()}
        else:
            obs = obs.unsqueeze(1)  # [batch, obs_dim] -> [batch, 1, obs_dim]

        # Process observations
        obs_processed = self._process_observations(obs)

        # Pass through RNN by producing self.chunk_size hidden states
        rnn_outputs = []
        for i in range(self.config.chunk_size):
            rnn_output, _ = self.rnn(obs_processed)
            rnn_outputs.append(rnn_output)
        rnn_outputs = torch.stack(rnn_outputs, dim=1)  # [batch, chunk_size, hidden_size]

        # Apply output MLP if present
        if self.output_mlp is not None:
            rnn_outputs = self.output_mlp(rnn_outputs.reshape(-1, rnn_outputs.size(-1)))

        batch_size = obs_processed.size(0)

        # Generate action distribution parameters
        if self.is_continuous:
            mean = self.mean_layer(rnn_outputs)
            log_std = self.log_std_layer(rnn_outputs)
            log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)
            mean, log_std = (
                mean.view(obs.size(0), self.config.chunk_size, -1),
                log_std.view(obs.size(0), self.config.chunk_size, -1),
            )
            return mean, log_std
        else:
            logits = self.logits_layer(rnn_outputs)
            logits = logits.view(obs.size(0), self.config.chunk_size, -1)
            return logits

    def _act(self, obs: torch.Tensor, deterministic: bool = False, actor_state: Any = None) -> Tuple[torch.Tensor, Any]:
        """
        Generate actions for inference with single observations.

        Args:
            obs: Single observation [batch_size, obs_dim]
            deterministic: Whether to use deterministic actions
            actor_state: Dict containing "hidden_latent" for RNN state

        Returns:
            (action, new_actor_state)
        """
        # Process single observation
        obs_processed = self._process_observations(obs)  # [batch, feature_dim]
        obs_processed = obs_processed.unsqueeze(1)  # [batch, 1, feature_dim]

        # Get RNN hidden state from actor_state
        if actor_state is not None and "hidden_latent" in actor_state:
            hidden_state = actor_state["hidden_latent"]
        else:
            hidden_state = self._get_initial_hidden_state(obs_processed.size(0))

        # Pass through RNN for self.chunk_size timesteps
        rnn_outputs = []
        for i in range(self.config.chunk_size):
            rnn_output, _ = self.rnn(obs_processed, hidden_state)
            rnn_outputs.append(rnn_output)
        rnn_outputs = torch.stack(rnn_outputs, dim=1)  # [batch, chunk_size, hidden_size]

        # Apply output MLP if present
        if self.output_mlp is not None:
            rnn_outputs = self.output_mlp(rnn_outputs.reshape(-1, rnn_outputs.size(-1)))

        batch_size = obs_processed.size(0)
        if self.is_continuous:
            # get a mean/log_std for each timestep
            mean = self.mean_layer(rnn_outputs)
            log_std = self.log_std_layer(rnn_outputs)
            log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)
        else:
            mean = self.logits_layer(rnn_outputs)

        if log_std is not None:
            action = self.action_dist.actions_from_params(mean, log_std, deterministic=deterministic)
        else:
            action = self.action_dist.actions_from_params(mean, deterministic=deterministic)

        action = action.view(batch_size, self.config.chunk_size, -1)

        # action = action.squeeze(1)

        # Create new actor state
        new_actor_state = {"hidden_latent": hidden_state}

        return action, new_actor_state

    def get_action_dist_params(self, obs, hidden=None):
        """Get action distribution parameters."""
        obs_processed = self._process_observations(obs)
        if hidden is None:
            # Handle dict observations
            rnn_output, new_hidden_state = self.rnn(obs_processed)
            hidden = rnn_output[:, -1]
            if self.output_mlp is not None:
                hidden = self.output_mlp(hidden)

        else:
            obs_processed = self._process_observations(obs)
            obs_processed = obs_processed.unsqueeze(1)
            rnn_output, new_hidden_state = self.rnn(obs_processed, hidden)

        if self.is_continuous:
            # get a mean/log_std for each timestep
            mean = self.mean_layer(rnn_output)
            log_std = self.log_std_layer(rnn_output)
            log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)
            kwargs = {}
            return mean, log_std, kwargs
        else:
            logits = self.logits_layer(hidden)
            return logits, None, {}

    def get_initial_state(self) -> Dict[str, Any]:
        """Get initial actor state for rollout worker."""
        return {"hidden_latent": None}  # Will be initialized when first used

    def _get_initial_hidden_state(self, batch_size: int):
        """Get initial hidden state for RNN."""
        device = next(self.parameters()).device
        num_directions = 2 if self.config.rnn_bidirectional else 1

        if self.config.rnn_type == "LSTM":
            h_0 = torch.zeros(
                self.config.rnn_num_layers * num_directions,
                batch_size,
                self.config.rnn_hidden_size,
                device=device,
            )
            c_0 = torch.zeros(
                self.config.rnn_num_layers * num_directions,
                batch_size,
                self.config.rnn_hidden_size,
                device=device,
            )
            return (h_0, c_0)
        else:  # GRU or RNN
            h_0 = torch.zeros(
                self.config.rnn_num_layers * num_directions,
                batch_size,
                self.config.rnn_hidden_size,
                device=device,
            )
            return h_0

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))
