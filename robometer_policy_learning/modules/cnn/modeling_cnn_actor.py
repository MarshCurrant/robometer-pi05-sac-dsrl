import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Union

from robometer_policy_learning.modules.base import BaseActor
from robometer_policy_learning.modules.cnn import CNNActorConfig


class CNNActor(BaseActor):
    """
    CNN-based actor for dictionary observations with images and state.
    Uses ResNet backbone for image processing.
    """

    def __init__(self, config: CNNActorConfig):
        super().__init__(action_space=config.action_space)
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

        # Calculate total input size for MLP
        total_feature_size = self._calculate_total_feature_size(resnet_feature_size)

        # Build MLP head
        self.mlp = self._build_mlp(total_feature_size)

        # Output layers
        action_dim = int(np.prod(config.action_space.shape))
        # Ensure hidden_dims[-1] is an integer
        final_hidden_dim = int(config.hidden_dims[-1])
        self.mean_layer = nn.Linear(final_hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(final_hidden_dim, action_dim)

        # Initialize log std
        self.log_std_layer.weight.data.fill_(0.0)
        self.log_std_layer.bias.data.fill_(config.log_std_init)

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

    def _build_mlp(self, input_size: int) -> nn.Module:
        layers = []
        prev_size = input_size

        for hidden_size in self.config.hidden_dims:
            # Ensure hidden_size is an integer (fixes numpy.float64 issues)
            hidden_size = int(hidden_size)

            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation())
            prev_size = hidden_size

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

    def _forward(self, obs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle dictionary observations
        if isinstance(obs, dict):
            combined_features = self._process_observations(obs)
        else:
            # Fallback for non-dict observations (shouldn't happen with this config)
            raise ValueError("Expected dictionary observation but got tensor")

        # MLP processing
        hidden = self.mlp(combined_features)

        # Output mean and log std
        mean = self.mean_layer(hidden)
        log_std = self.log_std_layer(hidden)

        # Clamp log std
        log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)

        if self.config.use_tanh_output:
            mean = torch.tanh(mean)

        return mean, log_std

    def _act(
        self,
        obs: Union[Dict[str, torch.Tensor], Dict[str, np.ndarray]],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Internal forward pass that returns actions in [-1, 1] range."""
        # Convert numpy arrays to tensors if needed
        if isinstance(obs, dict):
            obs_tensors = {}
            for key, value in obs.items():
                if not isinstance(value, torch.Tensor):
                    obs_tensors[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    obs_tensors[key] = value
            obs = obs_tensors

        mean, log_std = self._forward(obs)

        if deterministic:
            action = mean
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()

        # Apply tanh to get actions in [-1, 1] range for consistent normalization
        if self.config.use_tanh_output:
            action = torch.tanh(action)
        else:
            # If not using tanh, we need to clamp to a reasonable range
            action = torch.clamp(action, -1.0, 1.0)

        return action

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def get_action_dist_params(self, obs):
        """
        Get the parameters of the action distribution for SAC training.
        Returns mean_actions, log_std, kwargs to match SB3 interface.
        """
        # Handle dictionary observations
        if isinstance(obs, dict):
            combined_features = self._process_observations(obs)
        else:
            raise ValueError("Expected dictionary observation but got tensor")

        # MLP processing
        hidden = self.mlp(combined_features)

        # For continuous actions: return mean, log_std, kwargs
        mean = self.mean_layer(hidden)
        log_std = self.log_std_layer(hidden)

        # Clamp log std
        log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)

        kwargs = {}
        return mean, log_std, kwargs
