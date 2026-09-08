import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Dict, Union

from robometer_policy_learning.modules.base import BaseCritic
from robometer_policy_learning.modules.cnn import CNNCriticConfig


class CNNCritic(BaseCritic):
    """
    CNN-based critic for dictionary observations with images and state.
    Uses ResNet backbone for image processing.
    """

    def __init__(self, config: CNNCriticConfig):
        super().__init__()
        self.config = config

        # Load ResNet backbone
        self.resnet = torch.hub.load("pytorch/vision:v0.10.0", config.resnet_model, pretrained=config.pretrained)

        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Freeze backbone if specified
        if config.freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Get ResNet feature size
        resnet_feature_size = self._get_resnet_feature_size()

        # Calculate total input size for MLP (features + action)
        total_feature_size = self._calculate_total_feature_size(resnet_feature_size)
        action_dim = np.prod(config.action_space.shape)
        input_size = total_feature_size + action_dim

        # Build single critic head
        self.critic = self._build_critic_head(input_size)

    def _get_resnet_feature_size(self) -> int:
        """Calculate the output feature size of ResNet backbone."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.config.image_channels, *self.config.image_size)
            features = self.resnet(dummy_input)
            return features.view(1, -1).shape[1]

    def _calculate_total_feature_size(self, resnet_feature_size: int) -> int:
        """Calculate total feature size including images and state."""
        # Count image keys in observation space
        image_keys = self._get_image_keys()
        num_images = len(image_keys)

        # Get state dimension
        state_dim = self._get_state_dimension()

        # Total features: (num_images * resnet_features) + state_features
        return num_images * resnet_feature_size + state_dim

    def _get_image_keys(self) -> list:
        """Extract image keys from observation space."""
        if not isinstance(self.config.observation_space, gym.spaces.Dict):
            raise ValueError("Observation space must be a gymnasium Dict space")

        image_keys = []
        for key in self.config.observation_space.spaces.keys():
            if key.startswith("observation.images."):
                image_keys.append(key)
        return image_keys

    def _get_state_dimension(self) -> int:
        """Calculate state feature dimension."""
        if not isinstance(self.config.observation_space, gym.spaces.Dict):
            raise ValueError("Observation space must be a gymnasium Dict space")

        state_dim = 0
        for key, space in self.config.observation_space.spaces.items():
            if key.startswith("observation.state"):
                if isinstance(space, gym.spaces.Box):
                    state_dim += np.prod(space.shape)
                else:
                    raise ValueError(f"State space {key} must be a Box space")
        return state_dim

    def _build_critic_head(self, input_size: int) -> nn.Module:
        layers = []
        prev_size = input_size

        for hidden_size in self.config.hidden_dims:
            layers.append(nn.Linear(prev_size, hidden_size))
            if self.config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(self._get_activation())
            prev_size = hidden_size

        # Output layer (single Q-value)
        layers.append(nn.Linear(prev_size, 1))

        return nn.Sequential(*layers)

    def _get_activation(self) -> nn.Module:
        if self.config.activation.lower() == "relu":
            return nn.ReLU()
        elif self.config.activation.lower() == "tanh":
            return nn.Tanh()
        elif self.config.activation.lower() == "elu":
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {self.config.activation}")

    def _process_observations(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process dictionary observations into a single feature vector."""
        batch_size = None
        features = []

        # Get the device of the model
        device = next(self.parameters()).device

        # Process images
        image_keys = self._get_image_keys()
        for key in image_keys:
            if key in obs_dict:
                image = obs_dict[key]
                if batch_size is None:
                    batch_size = image.shape[0]

                # Ensure image has correct shape and type
                if image.dim() == 3:
                    image = image.unsqueeze(0)

                # Move image to correct device and convert to float
                image = image.to(device).float()

                # Process through ResNet
                image_features = self.resnet(image)
                image_features = image_features.view(image_features.size(0), -1)
                features.append(image_features)

        # Process state features
        state_features = []
        for key, value in obs_dict.items():
            if key.startswith("observation.state"):
                if batch_size is None:
                    batch_size = value.shape[0]

                # Flatten state features and move to device
                if value.dim() > 2:
                    value = value.view(value.size(0), -1)
                elif value.dim() == 1:
                    value = value.unsqueeze(0)

                value = value.to(device).float()
                state_features.append(value)

        # Concatenate state features
        if state_features:
            state_concat = torch.cat(state_features, dim=1)
            features.append(state_concat)

        # Concatenate all features
        if features:
            return torch.cat(features, dim=1)
        else:
            raise ValueError("No valid features found in observation dictionary")

    def forward(self, obs: Union[Dict[str, torch.Tensor], torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        # Handle dictionary observations
        if isinstance(obs, dict):
            combined_features = self._process_observations(obs)
        else:
            # Fallback for non-dict observations (shouldn't happen with this config)
            raise ValueError("Expected dictionary observation but got tensor")

        # Concatenate features with action
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if combined_features.size(0) != action.size(0):
            # Handle batch size mismatch
            if combined_features.size(0) == 1:
                combined_features = combined_features.expand(action.size(0), -1)
            elif action.size(0) == 1:
                action = action.expand(combined_features.size(0), -1)

        combined = torch.cat([combined_features, action], dim=1)

        # Pass through critic head
        q_value = self.critic(combined)

        return q_value

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))
