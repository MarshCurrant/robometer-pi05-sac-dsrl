"""
IMPALA-style CNN encoder for image observations.

Based on the IMPALA encoder from:
https://github.com/nakamotoo/dsrl_pi0/blob/main/jaxrl2/networks/encoders/impala_encoder.py

Adapted to PyTorch for use with RL agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ResnetStack(nn.Module):
    """
    ResNet stack block used in IMPALA encoder.

    Each stack consists of:
    - Initial conv layer
    - Optional max pooling
    - Multiple residual blocks
    """

    def __init__(
        self,
        num_ch: int,
        num_blocks: int,
        in_channels: int,
        use_max_pooling: bool = True,
    ):
        super().__init__()
        self.num_ch = num_ch
        self.num_blocks = num_blocks
        self.use_max_pooling = use_max_pooling

        # Initial convolution with Xavier uniform initialization
        self.initial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        nn.init.xavier_uniform_(self.initial_conv.weight)
        if self.initial_conv.bias is not None:
            nn.init.zeros_(self.initial_conv.bias)

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.residual_blocks.append(self._make_residual_block())

    def _make_residual_block(self) -> nn.Module:
        """Create a residual block with two conv layers."""
        block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                self.num_ch,
                self.num_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.num_ch,
                self.num_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        # Initialize conv layers with Xavier uniform
        for module in block:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet stack.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, num_ch, H', W')
        """
        # Initial convolution
        out = self.initial_conv(x)

        # Max pooling if enabled
        if self.use_max_pooling:
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

        # Residual blocks
        for block in self.residual_blocks:
            residual = out
            out = block(out)
            out = out + residual  # Residual connection

        return out


class ImpalaEncoder(nn.Module):
    """
    IMPALA encoder for image observations.

    Architecture:
    - Normalizes input to [0, 1] by dividing by 255.0
    - Three ResNet stacks with increasing channel sizes
    - Final ReLU activation
    - Flattens output to (B, D) feature vector
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],  # (C, H, W) or (H, W, C)
        nn_scale: int = 1,
        num_blocks_per_stack: int = 2,
        output_dim: Optional[int] = None,
    ):
        """
        Initialize IMPALA encoder.

        Args:
            input_shape: Input image shape. Can be (C, H, W) or (H, W, C).
                         If (H, W, C), will be converted to (C, H, W).
            nn_scale: Scaling factor for channel sizes (default: 1)
            num_blocks_per_stack: Number of residual blocks per stack (default: 2)
            output_dim: Optional output dimension. If None, uses the number of channels
                        after global average pooling. If provided, adds a linear projection
                        layer to map to this dimension.
        """
        super().__init__()
        self.nn_scale = nn_scale
        self.num_blocks_per_stack = num_blocks_per_stack

        # Normalize input shape to (C, H, W)
        if len(input_shape) == 3:
            # Check if channels are last (H, W, C)
            if input_shape[2] <= 4:  # Likely (H, W, C)
                self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
            else:  # Likely (C, H, W)
                self.input_shape = input_shape
        elif len(input_shape) == 2:  # (H, W) - grayscale
            self.input_shape = (1, input_shape[0], input_shape[1])
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

        # Stack channel sizes (scaled by nn_scale)
        stack_sizes = [16, 32, 32]
        input_channels = self.input_shape[0]
        self.stack_blocks = nn.ModuleList(
            [
                ResnetStack(
                    num_ch=stack_sizes[0] * self.nn_scale,
                    num_blocks=num_blocks_per_stack,
                    in_channels=input_channels,
                    use_max_pooling=True,
                ),
                ResnetStack(
                    num_ch=stack_sizes[1] * self.nn_scale,
                    num_blocks=num_blocks_per_stack,
                    in_channels=stack_sizes[0] * self.nn_scale,
                    use_max_pooling=True,
                ),
                ResnetStack(
                    num_ch=stack_sizes[2] * self.nn_scale,
                    num_blocks=num_blocks_per_stack,
                    in_channels=stack_sizes[1] * self.nn_scale,
                    use_max_pooling=True,
                ),
            ]
        )

        # Global average pooling from IMPOOLA: https://openreview.net/forum?id=Kkw4nqaM9Y#discussion
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Compute feature dimension after pooling (number of channels)
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            dummy_output = self._forward_features(dummy_input)
            # After pooling: (B, C, 1, 1) -> C channels
            pooled_output = self.global_pool(dummy_output)
            feature_dim = int(pooled_output.shape[1])  # Number of channels

        # Optional MLP projection to output_dim
        if output_dim is not None:
            self.projection = nn.Linear(feature_dim, output_dim)
        else:
            self.projection = None
            output_dim = feature_dim

        self.output_dim = output_dim

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder stacks (without final flatten).

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Feature tensor before flattening
        """
        # Pass through all stacks
        out = x
        for stack in self.stack_blocks:
            out = stack(out)

        # Final ReLU
        out = F.relu(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through IMPALA encoder.

        Args:
            x: Input tensor. Can be:
               - (B, C, H, W) or (B, H, W, C)
               - (C, H, W) or (H, W, C) (single image)
               - Values in [0, 255] or [0, 1]

        Returns:
            Feature vector of shape (B, output_dim) after global average pooling
            and optional projection
        """
        # Handle different input formats
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension

        # Convert to (B, C, H, W) format
        if x.dim() == 4:
            # Check if channels are last (B, H, W, C)
            if x.size(-1) <= 4:  # Likely channels last
                x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            # Otherwise assume (B, C, H, W)

        # Normalize to [0, 1] if values are in [0, 255]
        if x.max() > 1.0:
            x = x.float() / 255.0

        # Forward through stacks
        features = self._forward_features(x)  # (B, C, H, W)

        # Global average pooling: (B, C, H, W) -> (B, C, 1, 1)
        features = self.global_pool(features)

        # Flatten spatial dimensions: (B, C, 1, 1) -> (B, C)
        batch_size = features.size(0)
        features = features.view(batch_size, -1)

        # Optional projection to output_dim
        if self.projection is not None:
            features = self.projection(features)

        return features


class SmallerImpalaEncoder(nn.Module):
    """
    Smaller variant of IMPALA encoder with fewer residual blocks.

    Uses the same architecture but with reduced number of blocks per stack.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        nn_scale: int = 1,
        output_dim: Optional[int] = None,
    ):
        """
        Initialize smaller IMPALA encoder.

        Args:
            input_shape: Input image shape. Can be (C, H, W) or (H, W, C).
            nn_scale: Scaling factor for channel sizes (default: 1)
            output_dim: Optional output dimension. If None, uses the number of channels
                        after global average pooling. If provided, adds a linear projection
                        layer to map to this dimension.
        """
        super().__init__()
        self.nn_scale = nn_scale

        # Normalize input shape to (C, H, W)
        if len(input_shape) == 3:
            if input_shape[2] <= 4:  # Likely (H, W, C)
                self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
            else:  # Likely (C, H, W)
                self.input_shape = input_shape
        elif len(input_shape) == 2:  # (H, W) - grayscale
            self.input_shape = (1, input_shape[0], input_shape[1])
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

        # Stack channel sizes with fewer blocks
        stack_sizes = [16, 32, 32]
        input_channels = self.input_shape[0]
        self.stack_blocks = nn.ModuleList(
            [
                ResnetStack(
                    num_ch=stack_sizes[0] * self.nn_scale,
                    num_blocks=2,
                    in_channels=input_channels,
                    use_max_pooling=True,
                ),
                ResnetStack(
                    num_ch=stack_sizes[1] * self.nn_scale,
                    num_blocks=1,
                    in_channels=stack_sizes[0] * self.nn_scale,
                    use_max_pooling=True,
                ),
                ResnetStack(
                    num_ch=stack_sizes[2] * self.nn_scale,
                    num_blocks=1,
                    in_channels=stack_sizes[1] * self.nn_scale,
                    use_max_pooling=True,
                ),
            ]
        )

        # Global average pooling from IMPOOLA: https://openreview.net/forum?id=Kkw4nqaM9Y#discussion
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Compute feature dimension after pooling (number of channels)
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            dummy_output = self._forward_features(dummy_input)
            # After pooling: (B, C, 1, 1) -> C channels
            pooled_output = self.global_pool(dummy_output)
            feature_dim = int(pooled_output.shape[1])  # Number of channels

        # Optional MLP projection to output_dim
        if output_dim is not None:
            self.projection = nn.Linear(feature_dim, output_dim)
        else:
            self.projection = None
            output_dim = feature_dim

        self.output_dim = output_dim

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder stacks (without final flatten)."""
        out = x
        for stack in self.stack_blocks:
            out = stack(out)
        out = F.relu(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through smaller IMPALA encoder."""
        # Handle different input formats
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.dim() == 4:
            if x.size(-1) <= 4:  # Likely channels last
                x = x.permute(0, 3, 1, 2)

        # Normalize to [0, 1] if values are in [0, 255]
        if x.max() > 1.0:
            x = x.float() / 255.0

        features = self._forward_features(x)  # (B, C, H, W)

        # Global average pooling: (B, C, H, W) -> (B, C, 1, 1)
        features = self.global_pool(features)

        # Flatten spatial dimensions: (B, C, 1, 1) -> (B, C)
        batch_size = features.size(0)
        features = features.view(batch_size, -1)

        # Optional projection to output_dim
        if self.projection is not None:
            features = self.projection(features)

        return features
