from dataclasses import dataclass
from typing import Optional, List, Union, Any
import gymnasium as gym
import torch.nn as nn

from robometer_policy_learning.modules.base import BaseCriticConfig


@dataclass
class RNNCriticConfig(BaseCriticConfig):
    """Configuration for RNN-based critic."""

    # RNN-specific parameters
    rnn_type: str = "LSTM"  # "LSTM", "GRU", or "RNN"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1
    rnn_dropout: float = 0.0
    rnn_bidirectional: bool = False

    # MLP parameters for feature extraction and output
    feature_hidden_dims: List[int] = None  # MLP before RNN, None means direct obs->RNN
    output_hidden_dims: List[int] = None  # MLP after RNN, None means direct RNN->value

    # Standard MLP parameters
    activation: str = "relu"
    use_layer_norm: bool = False
    dropout_rate: float = 0.0

    # Training parameters
    chunk_size: int = 30  # Expected chunk size for training

    # Featurizer for dict observations
    featurizer: Optional[dict] = None
    preprocess_obs_transform: Optional[List[Any]] = None

    # IMPALA encoder parameters (optional)
    image_encoder_type: str = None  # "impala" to enable IMPALA for image keys
    impala_nn_scale: int = 1
    impala_num_blocks_per_stack: int = 2
    impala_use_smaller: bool = False
    impala_output_dim: int = None

    def __post_init__(self):
        # Don't call super().__post_init__() to avoid base class validation issues

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
    def critic_class(self):
        from robometer_policy_learning.modules.rnn import RNNCritic

        return RNNCritic
