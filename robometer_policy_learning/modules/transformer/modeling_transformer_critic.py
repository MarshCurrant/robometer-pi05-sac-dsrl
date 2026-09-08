import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, Any, Dict, List
import gymnasium as gym

from robometer_policy_learning.modules.base import BaseCritic
from robometer_policy_learning.modules.transformer import TransformerCriticConfig
from robometer_policy_learning.modules.transformer.transformer_utils import (
    _build_mlp_layers,
    PositionalEncoding,
    TransformerFeatureExtractor,
)


class TransformerCritic(BaseCritic):
    """
    Transformer-based critic for value estimation with action sequences.

    Architecture for Q-function:
    [obs, action_seq] -> [feature_mlp] -> positional_encoding -> transformer_encoder -> [output_mlp] -> value

    Architecture for V-function:
    obs -> [feature_mlp] -> transformer_encoder -> [output_mlp] -> value

    Supports Q-functions (obs + action -> Q-value) and V-functions (obs -> V-value).
    Actions must be 3D: [batch_size, chunk_size, action_dim] for action chunking support.
    """

    def __init__(self, config: TransformerCriticConfig):
        super().__init__(config)
        self.config = config
        self.use_action = config.use_action

        # Feature extraction for observations
        self.obs_feature_extractor = TransformerFeatureExtractor(
            observation_space=config.observation_space,
            featurizer_cfg=config.featurizer,
            feature_hidden_dims=config.feature_hidden_dims,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
            dropout_rate=config.dropout_rate,
            preprocess_obs_transform=config.preprocess_obs_transform,
            # Image encoder parameters
            image_encoder_type=config.image_encoder_type,
            resnet_backbone=config.resnet_backbone,
            resnet_pretrained=config.resnet_pretrained,
            image_feature_dim=config.image_feature_dim,
            spatial_softmax_num_kp=config.spatial_softmax_num_kp,
            # DINOv2 encoder parameters (used when image_encoder_type == "dinov2")
            dinov2_model=config.dinov2_model,
            dinov2_processor=config.dinov2_processor,
            # IMPALA encoder parameters (used when image_encoder_type == "impala")
            impala_nn_scale=config.impala_nn_scale,
            impala_num_blocks_per_stack=config.impala_num_blocks_per_stack,
            impala_use_smaller=config.impala_use_smaller,
            impala_output_dim=config.impala_output_dim,
            # Language embedding parameters
            use_language_embeddings=config.use_language_embeddings,
            lang_embedding_dim=config.lang_embedding_dim,
            lang_embedding_device=config.lang_embedding_device,
            use_modality_projections=False,
        )

        # Action processing for Q-functions
        if self.use_action:
            if isinstance(config.action_space, gym.spaces.Box):
                action_dim = int(np.prod(config.action_space.shape))
            elif isinstance(config.action_space, gym.spaces.Discrete):
                action_dim = config.action_space.n
            else:
                raise ValueError(f"Unsupported action space: {config.action_space}")

            # Action embedding and projection to transformer dimension
            self.action_embedding = nn.Linear(action_dim, config.action_embedding_dim)
            self.action_projection = nn.Linear(config.action_embedding_dim, config.d_model)

        # Project observation features to transformer dimension
        if (
            self.obs_feature_extractor.output_dim is not None
            and self.obs_feature_extractor.output_dim != config.d_model
        ):
            self.obs_projection = nn.Linear(self.obs_feature_extractor.output_dim, config.d_model)
        else:
            self.obs_projection = None

        # LayerNorm on input before positional encoding
        self.input_norm = nn.LayerNorm(config.d_model)

        # Positional encoding
        self.position_embedding = PositionalEncoding(
            d_model=config.d_model,
            max_len=self.config.chunk_size + 1,  # +1 for the observation token
            dropout=config.positional_dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dropout=config.transformer_dropout,
            activation=config.transformer_activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # Output MLP
        transformer_output_dim = config.d_model
        if config.output_hidden_dims:
            self.output_mlp = _build_mlp_layers(
                transformer_output_dim,
                config.output_hidden_dims,
                config.activation,
                config.use_layer_norm,
                config.dropout_rate,
            )
            final_hidden_dim = config.output_hidden_dims[-1]
        else:
            self.output_mlp = None
            final_hidden_dim = transformer_output_dim

        # Pooling strategy
        self.pooling_strategy = config.pooling_strategy

        # Attention pooling (if used)
        if self.pooling_strategy == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(config.d_model, config.d_model), nn.Tanh(), nn.Linear(config.d_model, 1, bias=False)
            )

        # Value head with LayerNorm
        self.value_head = nn.Linear(final_hidden_dim, 1)
        # nn.Sequential(
        #    #nn.LayerNorm(final_hidden_dim),
        #    nn.Linear(final_hidden_dim, 1)
        # )

    def _process_actions(self, obs_features: torch.Tensor, action: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Process observations and actions into a sequence for the transformer.

        Args:
            obs_features: Observation features [batch_size, obs_feature_dim]
            action: Actions [batch_size, chunk_size, action_dim] (must be 3D)
            batch_size: Batch size

        Returns:
            input_sequence: [batch_size, 1 + chunk_size, d_model]
        """
        # Ensure actions are 3D
        if action.dim() == 2:
            action = action.unsqueeze(1)  # [batch, action_dim] -> [batch, 1, action_dim]

        if action.dim() != 3:
            raise ValueError(f"Action must be 3D [batch, chunk_size, action_dim], got shape {action.shape}")

        chunk_size = action.size(1)

        # Project observation to d_model as conditioning token
        if self.obs_projection is not None:
            obs_token = self.obs_projection(obs_features)  # [batch_size, d_model]
        else:
            obs_token = obs_features  # Already d_model
        obs_token = obs_token.unsqueeze(1)  # [batch_size, 1, d_model]

        # Embed each action in the sequence and project to d_model
        action_flat = action.reshape(-1, action.size(-1))  # [batch_size * chunk_size, action_dim]
        action_features_flat = self.action_embedding(action_flat)  # [batch_size * chunk_size, action_embedding_dim]
        action_tokens = self.action_projection(action_features_flat)  # [batch_size * chunk_size, d_model]
        action_tokens = action_tokens.view(batch_size, chunk_size, -1)  # [batch_size, chunk_size, d_model]

        # Concatenate: [obs_token, action_tokens] -> [batch_size, 1 + chunk_size, d_model]
        input_sequence = torch.cat([obs_token, action_tokens], dim=1)

        return input_sequence

    def _pool_transformer_output(self, transformer_output: torch.Tensor) -> torch.Tensor:
        """
        Pool transformer output based on configured strategy.

        Args:
            transformer_output: [batch_size, seq_len, d_model]

        Returns:
            pooled: [batch_size, d_model]
        """
        if self.pooling_strategy == "first":
            # Use first token (observation/CLS token) - preserves state information
            return transformer_output[:, 0, :]

        elif self.pooling_strategy == "attention":
            # Learned attention pooling
            # attention_scores: [batch_size, seq_len, 1]
            attention_scores = self.attention_pool(transformer_output)
            attention_weights = F.softmax(attention_scores, dim=1)
            # Weighted sum: [batch_size, d_model]
            return (transformer_output * attention_weights).sum(dim=1)

        elif self.pooling_strategy == "weighted_mean":
            # Exponentially weighted mean - recent tokens (later in sequence) matter more
            seq_len = transformer_output.size(1)
            # Create weights: [1, seq_len, 1] with exponential increase
            positions = torch.arange(seq_len, device=transformer_output.device, dtype=transformer_output.dtype)
            # Weight obs token (position 0) highly, then increase for later actions
            weights = torch.exp(positions * 0.1)  # Gentle exponential
            weights[0] = weights.max()  # Ensure obs token has high weight
            weights = weights / weights.sum()
            weights = weights.view(1, -1, 1)
            return (transformer_output * weights).sum(dim=1)

        else:  # "mean" (default)
            return transformer_output.mean(dim=1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for value estimation.

        Args:
            obs: Observations [batch_size, obs_dim] or dict
            action: Actions [batch_size, chunk_size, action_dim] (3D required for Q-function)

        Returns:
            Values [batch_size, 1]
        """
        pooled_output = self.compute_pooled(obs, action)
        # Generate value
        values = self.value_head(pooled_output)

        return values

    def compute_pooled(self, obs: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """
        Compute shared features up to pooled representation.

        Returns a tensor of shape [batch_size, d_model] that is the output of the transformer
        encoder (mean pooled). This is the input to output_mlp.

        Note: SAC's modeling_sac.py calls this and then applies each critic's output_mlp separately.
        """
        # Same batch-size safety as forward()
        obs = self._remove_obs_keys(obs)

        obs_features = self.obs_feature_extractor(obs)
        batch_size = int(obs_features.size(0))

        if self.use_action and action is not None:
            input_sequence = self._process_actions(obs_features, action, batch_size)
        else:
            if self.obs_projection is not None:
                obs_token = self.obs_projection(obs_features)
            else:
                obs_token = obs_features
            input_sequence = obs_token.unsqueeze(1)

        # Apply LayerNorm, positional encoding, and transformer
        input_sequence = self.input_norm(input_sequence)
        sequence_with_pos = self.position_embedding(input_sequence)
        transformer_output = self.transformer_encoder(sequence_with_pos)

        # Pool using configured strategy
        pooled_output = self._pool_transformer_output(transformer_output)

        # Apply output MLP if present
        if self.output_mlp is not None:
            pooled_output = self.output_mlp(pooled_output)

        return pooled_output

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))
