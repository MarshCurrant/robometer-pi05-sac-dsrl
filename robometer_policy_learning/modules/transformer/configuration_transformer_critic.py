from dataclasses import dataclass
from typing import Optional, List, Union, Any
import gymnasium as gym
import torch.nn as nn

from robometer_policy_learning.modules.base import BaseCriticConfig


@dataclass
class TransformerCriticConfig(BaseCriticConfig):
    """Configuration for Transformer-based critic."""

    # Transformer-specific parameters
    d_model: int = 128  # Transformer model dimension
    nhead: int = 8  # Number of attention heads
    num_encoder_layers: int = 1  # Number of transformer encoder layers
    transformer_dropout: float = 0.1  # Transformer dropout
    transformer_activation: str = "gelu"  # Transformer activation (relu, gelu)

    # Action sequence parameters
    chunk_size: int = 10  # Expected action chunk size for Q-functions

    # MLP parameters for feature extraction and output
    feature_hidden_dims: List[int] = (128,)  # MLP before transformer, None means direct input->transformer
    output_hidden_dims: List[int] = None  # MLP after transformer, None means direct transformer->value

    # Standard MLP parameters
    activation: str = "relu"  # Activation for MLPs (not transformer)
    use_layer_norm: bool = True  # Changed to match actor config
    dropout_rate: float = 0.0

    # Positional encoding parameters
    positional_dropout: float = 0.0  # Dropout for positional encoding

    # Action processing parameters (for Q-functions)
    action_embedding_dim: int = 128  # Dimension for action embeddings

    # Pooling strategy for transformer output
    # "mean": mean over all tokens (dilutes state with longer sequences)
    # "first": use first token only (state/CLS token)
    # "attention": learned attention pooling over all tokens
    # "weighted_mean": exponentially weighted mean (recent tokens matter more)
    pooling_strategy: str = "attention"

    # Featurizer for dict observations
    featurizer: Optional[dict] = None
    preprocess_obs_transform: Optional[List[Any]] = None

    # Image encoder parameters
    image_encoder_type: str = None  # "resnet", "dinov2", "impala", or "flatten"
    resnet_backbone: str = "ResNet18"  # "ResNet18", "ResNet34", "ResNet50"
    resnet_pretrained: bool = True  # Whether to use pretrained ResNet weights
    image_feature_dim: int = 128  # Output dimension for image features
    spatial_softmax_num_kp: int = 32  # Number of keypoints for spatial softmax pooling

    # Language embedding parameters
    use_language_embeddings: bool = True  # Whether to use language embeddings
    lang_embedding_dim: int = 384  # Dimension of language embeddings (384 for MiniLM-L6)
    lang_embedding_device: str = "cpu"  # Device for language encoder

    # DINOv2 encoder parameters (used when image_encoder_type == "dinov2")
    dinov2_model: Any = "facebook/dinov2-base"
    dinov2_processor: Any = "facebook/dinov2-base"

    # IMPALA encoder parameters (used when image_encoder_type == "impala")
    impala_nn_scale: int = 1  # Scaling factor for channel sizes
    impala_num_blocks_per_stack: int = 2  # Number of residual blocks per stack
    impala_use_smaller: bool = False  # Whether to use SmallerImpalaEncoder variant
    impala_output_dim: int = None

    def __post_init__(self):
        # Set default hidden dimensions if not provided
        if self.feature_hidden_dims is None:
            self.feature_hidden_dims = []  # No MLP before transformer

        if self.output_hidden_dims is None:
            self.output_hidden_dims = []  # No MLP after transformer

        # Validate action chunk size
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        # Validate model dimension
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")

        # Validate number of heads
        if self.nhead <= 0:
            raise ValueError(f"nhead must be positive, got {self.nhead}")

        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})")

        # Validate transformer activation
        if self.transformer_activation not in ["relu", "gelu"]:
            raise ValueError(f"transformer_activation must be 'relu' or 'gelu', got {self.transformer_activation}")

        # Validate number of encoder layers
        if self.num_encoder_layers <= 0:
            raise ValueError(f"num_encoder_layers must be positive, got {self.num_encoder_layers}")

        # Validate action embedding dimension
        if self.action_embedding_dim <= 0:
            raise ValueError(f"action_embedding_dim must be positive, got {self.action_embedding_dim}")

    @property
    def critic_class(self):
        from robometer_policy_learning.modules.transformer import TransformerCritic

        return TransformerCritic
