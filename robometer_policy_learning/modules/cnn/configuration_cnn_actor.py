import torch
import gymnasium as gym
from torch import nn
from dataclasses import dataclass

from robometer_policy_learning.modules.base import BaseActorConfig


@dataclass
class CNNActorConfig(BaseActorConfig):
    """
    Configuration for CNN-based actor networks with ResNet backbone.
    Handles dictionary observations with images and state features.
    """

    observation_space: gym.Space = None
    action_space: gym.Space = None

    # ResNet backbone parameters
    resnet_model: str = "resnet18"  # ResNet model from torch hub (resnet18, resnet34, resnet50, etc.)
    pretrained: bool = True  # Whether to use pretrained weights
    freeze_backbone: bool = True  # Whether to freeze ResNet weights

    # Image processing parameters
    image_channels: int = 3  # Expected number of channels for images
    image_size: tuple = (224, 224)  # Expected image size (height, width)

    # MLP head parameters
    hidden_dims: tuple = (512,)  # Hidden layer dimensions after feature concatenation
    activation: str = "relu"  # Activation function

    # Output parameters
    use_tanh_output: bool = True  # Whether to use tanh activation on output
    log_std_init: float = 0.0  # Initial log std for stochastic policies
    log_std_min: float = -20.0  # Minimum log std
    log_std_max: float = 2.0  # Maximum log std
