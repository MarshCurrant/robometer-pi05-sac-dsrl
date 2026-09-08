import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, Any, Dict, List
import gymnasium as gym

from robometer_policy_learning.modules.base import BaseCritic
from robometer_policy_learning.modules.rnn import RNNCriticConfig


def _build_mlp_layers(input_size, hidden_dims, activation, use_layer_norm=False, dropout_rate=0.0):
    """Build MLP layers (same as in other modules)."""
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


class RNNCritic(BaseCritic):
    """
    RNN-based critic for sequential value estimation.

    Architecture:
    [obs, action] -> [feature_mlp] -> RNN -> [output_mlp] -> value

    Supports:
    - Q-functions (obs + action -> Q-value)
    - V-functions (obs -> V-value)
    - Training with sequences
    - Inference with single observations
    - LSTM, GRU, and vanilla RNN
    - Dict observations with featurizers
    """

    def __init__(self, config: RNNCriticConfig):
        super().__init__(config)
        self.config = config
        self.featurizer_cfg = config.featurizer
        self.preprocess_obs_transform = config.preprocess_obs_transform
        self.featurizers = nn.ModuleDict() if self.featurizer_cfg else None
        self.use_action = config.use_action

        # Build featurizers for dict observations
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
            # sum over featurizer cfg values
            obs_dim = sum(sum(value) for value in self.featurizer_cfg.values())
            self._flatten_keys = None
        elif isinstance(config.observation_space, gym.spaces.Dict):
            self._flatten_keys = [
                k for k, space in config.observation_space.spaces.items() if getattr(space, "shape", None) is not None
            ]
            obs_dim = sum(int(np.prod(config.observation_space.spaces[k].shape)) for k in self._flatten_keys)
        else:
            self._flatten_keys = None
            obs_dim = int(np.prod(config.observation_space.shape))

        # Total input dimension
        total_input_dim = obs_dim or 0

        # Build feature extraction MLP (before RNN)
        if total_input_dim > 0 and config.feature_hidden_dims:
            self.feature_mlp = _build_mlp_layers(
                total_input_dim,
                config.feature_hidden_dims,
                config.activation,
                config.use_layer_norm,
                config.dropout_rate,
            )
            rnn_input_size = config.feature_hidden_dims[-1]
        else:
            self.feature_mlp = None
            rnn_input_size = total_input_dim

        # Add action dimension if this is a Q-function
        if self.use_action:
            if isinstance(config.action_space, gym.spaces.Box):
                action_dim = int(np.prod(config.action_space.shape))
            elif isinstance(config.action_space, gym.spaces.Discrete):
                action_dim = config.action_space.n
            else:
                raise ValueError(f"Unsupported action space: {config.action_space}")
            self.action_mlp = _build_mlp_layers(
                action_dim,
                [rnn_input_size],
                config.activation,
                config.use_layer_norm,
                config.dropout_rate,
            )
        else:
            action_dim = 0

        # Build RNN
        self.rnn_input_size = rnn_input_size

        if self.rnn_input_size is not None and self.rnn_input_size > 0:
            self.rnn = self._build_rnn(self.rnn_input_size)
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

        # Value head (always outputs single value)
        self.value_head = nn.Linear(final_hidden_dim, 1)

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
        """Flatten observations (same logic as RNN actor)."""
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
        """Process observations through feature extraction (same as RNN actor)."""
        if self.preprocess_obs_transform is not None:
            for transform in self.preprocess_obs_transform:
                obs = transform(obs)

        obs_flat = self._flatten_obs(obs)

        # Build RNN dynamically if needed (but we need to account for action size)
        if self.rnn is None:
            # We'll build it later when we know the full input size (obs + action)
            pass

        # Apply feature MLP if present
        if self.feature_mlp is not None:
            obs_flat = self.feature_mlp(obs_flat)

        return obs_flat

    def forward(self, obs: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for value estimation.

        Args:
            obs: Observations [batch, obs_dim] or [batch, seq_len, obs_dim]
            action: Actions [batch, action_dim] or [batch, seq_len, action_dim] (if Q-function)

        Returns:
            Values [batch, 1] or [batch, seq_len, 1]
        """
        pooled = self._compute_pooled_features(obs, action)
        values = self.value_head(pooled)  # [batch, 1]
        return values

    def compute_pooled(self, obs: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """
        Compute pooled hidden representation before the value head.

        Mirrors the semantics of MLP/Transformer critics: returns a tensor of
        shape [batch, hidden_dim] corresponding to the last timestep's RNN
        output (after optional output_mlp), without applying the final value
        head. Useful for feature sharing or auxiliary losses.
        """
        return self._compute_pooled_features(obs, action)

    def _compute_pooled_features(self, obs: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """
        Shared core that handles preprocessing, sequence handling, RNN, and
        optional output MLP. Returns the pooled representation used by the
        value head.
        """
        # Handle both single obs and sequences (same as RNN actor)
        if isinstance(obs, dict):
            # For dict obs, check the first value to determine if sequence
            first_key = next(iter(obs.keys()))
            is_sequence = obs[first_key].dim() == 3
        else:
            is_sequence = obs.dim() == 3

        if not is_sequence:
            if isinstance(obs, dict):
                # Add sequence dimension to dict obs
                obs = {k: v.unsqueeze(1) if v.dim() == 2 else v for k, v in obs.items()}
            else:
                obs = obs.unsqueeze(1)  # [batch, obs_dim] -> [batch, 1, obs_dim]

            if action is not None:
                if action.dim() == 2:
                    action = action.unsqueeze(1)  # [batch, action_dim] -> [batch, 1, action_dim]
                elif action.dim() == 1:
                    action = action.unsqueeze(0).unsqueeze(1)  # [action_dim] -> [1, 1, action_dim]

        # Get dimensions
        if isinstance(obs, dict):
            first_key = next(iter(obs.keys()))
            batch_size, seq_len = obs[first_key].shape[:2]
        else:
            batch_size, seq_len = obs.shape[:2]

        # Process observations (same as RNN actor)
        if isinstance(obs, dict):
            # Handle dict observations
            obs_processed_list = []
            for i in range(seq_len):
                obs_t = {k: v[:, i] for k, v in obs.items()}
                obs_processed_t = self._process_observations(obs_t)
                obs_processed_list.append(obs_processed_t)
            obs_processed = torch.stack(obs_processed_list, dim=1)  # [batch, seq, feature_dim]
        else:
            obs_processed_list = []
            for i in range(seq_len):
                obs_t = obs[:, i]  # [batch, obs_dim]
                obs_processed_t = self._process_observations(obs_t)
                obs_processed_list.append(obs_processed_t)
            obs_processed = torch.stack(obs_processed_list, dim=1)  # [batch, seq, feature_dim]

        # Add action to the processed observations if this is a Q-function
        if self.use_action and action is not None:
            device = next(self.parameters()).device
            action = action.to(device)

            # Handle discrete actions
            if isinstance(self.config.action_space, gym.spaces.Discrete):
                if action.dim() == 3:  # [batch, seq, 1]
                    action = action.float()
                else:
                    action = action.unsqueeze(-1).float()

            # Ensure action has right shape [batch, seq, action_dim]
            if action.dim() == 2:  # [batch, action_dim]
                action = action.unsqueeze(1).expand(-1, seq_len, -1)

            action_chunk_size = action.size(1)
            action = self.action_mlp(action.reshape(batch_size * action_chunk_size, -1))
            action = action.reshape(batch_size, action_chunk_size, -1)

            # Concatenate obs and action
            rnn_input = torch.cat([obs_processed, action], dim=1)
        else:
            rnn_input = obs_processed

        # Build RNN dynamically if needed
        if self.rnn is None:
            input_dim = rnn_input.size(-1)
            self.rnn_input_size = input_dim
            self.rnn = self._build_rnn(input_dim)

        # Pass through RNN
        rnn_output, _ = self.rnn(rnn_input)  # [batch, seq, hidden_size]

        # Use last timestep output for value prediction (same as RNN actor)
        # last_hidden = rnn_output[:, -1]  # [batch, hidden_size]
        last_hidden = rnn_output[:, -1]
        # Apply output MLP if present
        if self.output_mlp is not None:
            last_hidden = self.output_mlp(last_hidden)

        return last_hidden

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))
