from dataclasses import dataclass
from typing import Optional, List, Union, Any
import gymnasium as gym
import torch.nn as nn

from robometer_policy_learning.modules.base import BaseActorConfig


@dataclass
class RNNActorConfig(BaseActorConfig):
    """Configuration for RNN-based actor."""

    # RNN-specific parameters
    rnn_type: str = "LSTM"  # "LSTM", "GRU", or "RNN"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1
    rnn_dropout: float = 0.0
    rnn_bidirectional: bool = False

    # MLP parameters for feature extraction and output
    feature_hidden_dims: List[int] = None  # MLP before RNN, None means direct obs->RNN
    output_hidden_dims: List[int] = None  # MLP after RNN, None means direct RNN->action

    # Standard MLP parameters
    activation: str = "relu"
    use_layer_norm: bool = False
    dropout_rate: float = 0.0

    # Action distribution parameters
    log_std_init: float = -0.5
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    # Training parameters
    chunk_size: int = 30  # Expected chunk size for training

    # Featurizer for dict observations (same as MLP actor)
    featurizer: Optional[dict] = None
    preprocess_obs_transform: Optional[List[Any]] = None

    # IMPALA encoder parameters (optional)
    image_encoder_type: str = None  # "impala" to enable IMPALA for image keys
    impala_nn_scale: int = 1
    impala_num_blocks_per_stack: int = 2
    impala_use_smaller: bool = False
    impala_output_dim: int = None

    def __post_init__(self):
        # super().__post_init__()

        # Set default hidden dimensions if not provided
        if self.feature_hidden_dims is None:
            self.feature_hidden_dims = []  # No MLP before RNN

        if self.output_hidden_dims is None:
            self.output_hidden_dims = []  # No MLP after RNN

        # Validate RNN type
        if self.rnn_type not in ["LSTM", "GRU", "RNN"]:
            raise ValueError(f"rnn_type must be one of ['LSTM', 'GRU', 'RNN'], got {self.rnn_type}")

        # Validate chunk size
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

    @property
    def actor_class(self):
        from robometer_policy_learning.modules.rnn import RNNActor

        return RNNActor
