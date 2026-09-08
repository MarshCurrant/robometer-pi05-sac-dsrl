import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, Any, Dict, List
import gymnasium as gym

from robometer_policy_learning.modules.base import BaseActor
from robometer_policy_learning.modules.transformer import TransformerActorConfig
from robometer_policy_learning.modules.transformer.transformer_utils import (
    _build_mlp_layers,
    PositionalEncoding,
    TransformerFeatureExtractor,
)
from robometer_policy_learning.modules.base.distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
)


class TransformerActor(BaseActor):
    """
    Transformer-based chunking actor for action sequence prediction.

    Architecture:
    obs -> [feature_mlp] -> repeat -> positional_encoding -> transformer_decoder -> [output_mlp] -> action_heads

    Takes a single observation and predicts a sequence of actions (chunking).
    Unlike RNN which needs observation history, this repeats the current observation
    embedding for each action step and uses positional encoding.
    """

    def __init__(self, config: TransformerActorConfig):
        super().__init__(config)
        self.config = config

        # Feature extraction
        self.feature_extractor = TransformerFeatureExtractor(
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
            # Disable modality projections to avoid bottlenecking high-dim states
            use_modality_projections=False,
        )
        # Project to transformer dimension if needed
        if self.feature_extractor.output_dim != config.d_model:
            self.obs_projection = nn.Linear(self.feature_extractor.output_dim, config.d_model)
        else:
            self.obs_projection = None

        # LayerNorm on input before positional encoding
        self.input_norm = nn.LayerNorm(config.d_model)

        # Positional encoding
        self.position_embedding = PositionalEncoding(
            d_model=config.d_model,
            max_len=self.config.chunk_size,
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

        # LayerNorm before action heads
        self.pre_output_norm = nn.LayerNorm(final_hidden_dim)

        # Action heads
        if self.is_continuous:
            action_dim = int(np.prod(config.action_space.shape))

            # Separate processors for mean and log_std (like in inspiration code)
            self.mean_processor = nn.Sequential(
                nn.Linear(final_hidden_dim, final_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(final_hidden_dim // 2, action_dim),
            )
            self.log_std_processor = nn.Sequential(
                nn.Linear(final_hidden_dim, final_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(final_hidden_dim // 2, action_dim),
            )

            # Initialize log std
            with torch.no_grad():
                self.log_std_processor[-1].weight.fill_(0.0)
                self.log_std_processor[-1].bias.fill_(config.log_std_init)

            if config.use_tanh_output:
                self.action_dist = SquashedDiagGaussianDistribution(action_dim=action_dim)
            else:
                self.action_dist = DiagGaussianDistribution(action_dim=action_dim)
        else:
            action_dim = config.action_space.n
            self.logits_processor = nn.Linear(final_hidden_dim, action_dim)
            self.action_dist = CategoricalDistribution(action_dim=action_dim)

    def _forward(
        self, obs: torch.Tensor, actor_state: Any = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for training - predict action sequence from single observation.

        Args:
            obs: Observations [batch_size, obs_dim] or dict
            actor_state: Not used for transformer (stateless)

        Returns:
            Action distribution parameters for entire chunk sequence
        """

        def _infer_batch_size(o):
            if isinstance(o, dict):
                # Prefer clearly batched tensors; skip unbatched HWC images
                for v in o.values():
                    if isinstance(v, torch.Tensor):
                        if v.dim() >= 4:
                            return v.size(0)
                        if v.dim() == 3:
                            # If image-like (HWC or CHW), treat as unbatched
                            if v.size(0) == 3 or v.size(-1) == 3:
                                return 1
                            return v.size(0)
                        if v.dim() == 2:
                            return v.size(0)
                return 1
            elif isinstance(o, torch.Tensor):
                if o.dim() >= 2:
                    return o.size(0)
                return 1
            else:
                return 1

        batch_size = _infer_batch_size(obs)

        # Extract features from observations
        obs_features = self.feature_extractor(obs)  # [batch_size, feature_dim]

        # Project to transformer dimension if needed
        if self.obs_projection is not None:
            obs_features = self.obs_projection(obs_features)  # [batch_size, d_model]

        # Repeat observation for each action step in the chunk
        # [batch_size, d_model] -> [batch_size, chunk_size, d_model]
        repeated_obs = obs_features.unsqueeze(1).repeat(1, self.config.chunk_size, 1)

        # Apply LayerNorm on input before positional encoding
        repeated_obs = self.input_norm(repeated_obs)

        # Add positional encoding
        sequence_with_pos = self.position_embedding(repeated_obs)

        transformer_output = self.transformer_encoder(
            src=sequence_with_pos,
        )  # [batch_size, chunk_size, d_model]

        # Apply output MLP if present
        if self.output_mlp is not None:
            # Reshape for MLP: [batch_size * chunk_size, d_model]
            transformer_output_flat = transformer_output.view(-1, transformer_output.size(-1))
            output_features = self.output_mlp(transformer_output_flat)
            # Reshape back: [batch_size, chunk_size, output_dim]
            output_features = output_features.view(batch_size, self.config.chunk_size, -1)
        else:
            output_features = transformer_output

        # Apply LayerNorm before action heads
        output_features = self.pre_output_norm(output_features)

        # Generate action distribution parameters
        if self.is_continuous:
            # Reshape for processors: [batch_size * chunk_size, feature_dim]
            output_flat = output_features.view(-1, output_features.size(-1))

            mean = self.mean_processor(output_flat)  # [batch_size * chunk_size, action_dim]
            log_std = self.log_std_processor(output_flat)  # [batch_size * chunk_size, action_dim]

            # Clamp log_std
            log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)

            # Reshape back to [batch_size, chunk_size, action_dim]
            mean = mean.view(batch_size, self.config.chunk_size, -1)
            log_std = log_std.view(batch_size, self.config.chunk_size, -1)

            return mean, log_std
        else:
            # Reshape for processor: [batch_size * chunk_size, feature_dim]
            output_flat = output_features.view(-1, output_features.size(-1))
            logits = self.logits_processor(output_flat)  # [batch_size * chunk_size, action_dim]

            # Reshape back to [batch_size, chunk_size, action_dim]
            logits = logits.view(batch_size, self.config.chunk_size, -1)
            return logits

    def _act(self, obs: torch.Tensor, deterministic: bool = False, actor_state: Any = None) -> Tuple[torch.Tensor, Any]:
        """
        Generate action chunk for inference.

        Args:
            obs: Single observation [batch_size, obs_dim]
            deterministic: Whether to use deterministic actions
            actor_state: Not used for transformer (stateless)

        Returns:
            (action_chunk, new_actor_state)
        """

        mean, log_std, kwargs = self.get_action_dist_params(obs, actor_state)

        if self.is_continuous:
            actions = self.action_dist.actions_from_params(mean, log_std, deterministic=deterministic)
        else:
            raise NotImplementedError("Discrete action spaces not supported for transformer actor")

        # if self.is_continuous:
        #     mean, log_std = self._forward(obs, actor_state)
        #     # Reshape for action sampling: [batch_size * chunk_size, action_dim]
        #     batch_size = mean.size(0)
        #     action_dim = mean.size(-1)
        #     mean_flat = mean.view(-1, action_dim)
        #     log_std_flat = log_std.view(-1, action_dim)

        #     breakpoint()

        #     # Sample actions
        #     # actions_flat = self.action_dist.actions_from_params(
        #     #     mean_flat, log_std_flat, deterministic=deterministic
        #     # )
        #     actions_flat = mean_flat
        #     print(
        #         "actions_flat",
        #         actions_flat.min(),
        #         actions_flat.max(),
        #     )

        #     # Reshape back to [batch_size, chunk_size, action_dim]
        #     actions = actions_flat.view(batch_size, self.config.chunk_size, action_dim)
        # else:
        #     logits = self._forward(obs, actor_state)
        #     # Reshape for action sampling: [batch_size * chunk_size, action_dim]
        #     batch_size = logits.size(0)
        #     logits_flat = logits.view(-1, logits.size(-1))

        #     # Sample actions
        #     actions_flat = self.action_dist.actions_from_params(
        #         logits_flat, deterministic=deterministic
        #     )

        #     # Reshape back to [batch_size, chunk_size]
        #     actions = actions_flat.view(batch_size, self.config.chunk_size)

        # Actor state remains None (transformer is stateless)
        return actions, None

    def get_action_dist_params(self, obs, hidden=None):
        """Get action distribution parameters (compatibility method)."""
        if self.is_continuous:
            mean, log_std = self._forward(obs, actor_state=hidden)
            return mean, log_std, {}
        else:
            logits = self._forward(obs, actor_state=hidden)
            return logits, None, {}

    def get_initial_state(self) -> Dict[str, Any]:
        """Get initial actor state (transformer is stateless)."""
        return {}

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))
