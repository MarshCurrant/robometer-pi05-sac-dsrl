from dataclasses import dataclass
from typing import Optional, List, Union, Any
import gymnasium as gym
import torch.nn as nn

from robometer_policy_learning.modules.base import BaseActorConfig


@dataclass
class TransformerActorConfig(BaseActorConfig):
    """Configuration for Transformer-based chunking actor."""

    # Transformer-specific parameters
    d_model: int = 256  # Transformer model dimension
    nhead: int = 8  # Number of attention heads
    num_encoder_layers: int = 1  # Number of transformer encoder layers
    transformer_dropout: float = 0.1  # Transformer dropout
    transformer_activation: str = "gelu"  # Transformer activation (relu, gelu)

    # Chunking parameters
    chunk_size: int = 10  # Number of action chunks to predict

    # MLP parameters for feature extraction and output
    feature_hidden_dims: List[int] = None  # MLP before transformer, None means direct obs->transformer
    output_hidden_dims: List[int] = None  # MLP after transformer, None means direct transformer->action

    # Standard MLP parameters
    activation: str = "relu"  # Activation for MLPs (not transformer)
    use_layer_norm: bool = True
    dropout_rate: float = 0.0

    # Action distribution parameters
    use_tanh_output: bool = True
    log_std_init: float = 0
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    # Positional encoding parameters
    positional_dropout: float = 0.0  # Dropout for positional encoding

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
    impala_use_smaller: bool = True  # Whether to use SmallerImpalaEncoder variant
    impala_output_dim: int = None

    def __post_init__(self):
        # Set default hidden dimensions if not provided
        if self.feature_hidden_dims is None:
            self.feature_hidden_dims = []  # No MLP before transformer

        if self.output_hidden_dims is None:
            self.output_hidden_dims = []  # No MLP after transformer

        # Validate chunk size
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

        # Validate number of decoder layers
        if self.num_encoder_layers <= 0:
            raise ValueError(f"num_encoder_layers must be positive, got {self.num_encoder_layers}")

    @property
    def actor_class(self):
        from robometer_policy_learning.modules.transformer import TransformerActor

        return TransformerActor
