import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from robometer_policy_learning.modules.base import BaseCritic
from robometer_policy_learning.modules.mlp import MLPCriticConfig
from robometer_policy_learning.utils.featurizers import ObservationFeaturizer
from robometer_policy_learning.utils.featurizers import _build_mlp_layers


class MLPCritic(BaseCritic):
    """
    MLP-based critic for vector observations.
    Supports dict obs flattening and per-key featurizers.
    """

    def __init__(self, config: MLPCriticConfig):
        super().__init__(config)
        self.config = config
        self.preprocess_obs_transform = config.preprocess_obs_transform
        # API compatibility with transformer critic
        self.output_mlp = None

        # Initialize observation featurizer only if featurizer config is provided
        if config.featurizer is not None or config.image_encoder_type == "impala":
            self.obs_featurizer = ObservationFeaturizer(
                observation_space=config.observation_space,
                featurizer_cfg=config.featurizer,
                activation=config.activation,
                use_layer_norm=config.use_layer_norm,
                dropout_rate=config.dropout_rate,
                image_encoder_type=config.image_encoder_type,
                impala_nn_scale=config.impala_nn_scale,
                impala_num_blocks_per_stack=config.impala_num_blocks_per_stack,
                impala_use_smaller=config.impala_use_smaller,
                impala_output_dim=config.impala_output_dim,
            )
            # Input dimension - compute by passing example observation through featurizer
            obs_dim = self.obs_featurizer.output_dim
            if obs_dim <= 0:
                raise ValueError("ObservationFeaturizer produced invalid output dimension for critic.")
            self._flatten_keys = None
        else:
            # No featurizer: calculate obs_dim directly from observation_space
            self.obs_featurizer = None
            if isinstance(config.observation_space, gym.spaces.Dict):
                self._flatten_keys = [
                    k
                    for k, space in config.observation_space.spaces.items()
                    if getattr(space, "shape", None) is not None
                ]
                obs_dim = sum(int(np.prod(config.observation_space.spaces[k].shape)) for k in self._flatten_keys)
            else:
                self._flatten_keys = None
                obs_dim = int(np.prod(config.observation_space.shape))
            if obs_dim <= 0:
                raise ValueError(f"Invalid observation dimension calculated from observation_space: {obs_dim}")

        # Handle discrete vs continuous action spaces
        if isinstance(config.action_space, gym.spaces.Discrete):
            action_dim = 1
        else:
            action_dim = np.prod(config.action_space.shape)

        if config.use_action:
            if obs_dim is not None:
                input_dim = obs_dim + action_dim
            else:
                input_dim = None  # Will be determined dynamically
        else:
            input_dim = obs_dim

        # Build critic trunk and value head
        if input_dim is not None:
            self.critic_trunk, self.value_head = self._build_critic_trunk_and_head(int(input_dim))
        else:
            self.critic_trunk = None  # Will build dynamically in forward if needed
            self.value_head = None

    def _build_critic_trunk_and_head(self, input_size: int) -> tuple[nn.Module, nn.Module]:
        mlp_layers = _build_mlp_layers(
            input_size,
            self.config.hidden_dims,
            self.config.activation,
            self.config.use_layer_norm,
            self.config.dropout_rate,
        )
        trunk = nn.Sequential(*mlp_layers)

        # 2 layer MLP for value head that goes from hidden_dims[-1] to hidden_dims[-1]//2 to 1
        # value_head_layers = _build_mlp_layers(
        #    self.config.hidden_dims[-1],
        #   [self.config.hidden_dims[-1]//2],
        #    self.config.activation,
        #    self.config.use_layer_norm,
        #    dropout_rate=0.0,
        # )
        # value_head_layers.append(nn.Linear(self.config.hidden_dims[-1]//2, 1))
        # value_head = nn.Sequential(*value_head_layers)
        value_head = nn.Linear(self.config.hidden_dims[-1], 1)
        return trunk, value_head

    def _flatten_obs(self, obs):
        device = next(self.parameters()).device
        if self.obs_featurizer is not None:
            return self.obs_featurizer.flatten_obs(obs, device=device)
        else:
            # Manual flattening when no featurizer
            if isinstance(obs, dict):
                # Infer batch size from any clearly-batched tensor (dim >= 2)
                batch_size = None
                for k in self._flatten_keys:
                    v = obs.get(k)
                    if torch.is_tensor(v) and v.dim() >= 2:
                        batch_size = int(v.size(0))
                        break
                if batch_size is None:
                    batch_size = 1
                feats = []
                for k in self._flatten_keys:
                    v = obs.get(k)
                    if v is not None:
                        # Handle common scalar-per-batch tensors: shape (B,) should become (B, 1),
                        # not (1, B).
                        if torch.is_tensor(v):
                            if v.dim() == 0:
                                v = v.view(1, 1)
                            elif v.dim() == 1:
                                v = v.unsqueeze(-1) if int(v.size(0)) == batch_size else v.unsqueeze(0)
                        v_flat = v.reshape(v.size(0), -1) if torch.is_tensor(v) and v.dim() > 1 else v
                        feats.append(v_flat.to(device))
                return torch.cat(feats, dim=-1) if feats else torch.empty(0, device=device)
            else:
                # Tensor observation
                if obs.dim() > 2:
                    return obs.reshape(obs.size(0), -1).to(device)
                elif obs.dim() == 1:
                    return obs.unsqueeze(0).to(device)
                else:
                    return obs.to(device)

    def forward(self, obs, action: torch.Tensor = None) -> torch.Tensor:
        if self.preprocess_obs_transform is not None:
            for transform in self.preprocess_obs_transform:
                obs = transform(obs)
        # Ensure observations are flattened and featurized as needed
        obs = self._remove_obs_keys(obs)
        obs = self._flatten_obs(obs)
        device = next(self.parameters()).device
        if self.config.use_action:
            if action is None and self.config.use_action:
                raise ValueError(f"Action must be provided when use_action=True, {self.config.use_action}")
            # Handle discrete actions (convert to float and ensure proper shape)
            if isinstance(self.config.action_space, gym.spaces.Discrete):
                if action.dim() == 1:
                    action = action.unsqueeze(-1).float()
                elif action.dim() == 0:
                    action = action.unsqueeze(0).unsqueeze(-1).float()
                else:
                    action = action.float()
            else:
                if action.dim() > 2:
                    action = action.reshape(action.size(0), -1)
                elif action.dim() == 1:
                    action = action.unsqueeze(0)
            # Handle batch size mismatch
            if obs.size(0) != action.size(0):
                if obs.size(0) == 1:
                    obs = obs.expand(action.size(0), -1)
                elif action.size(0) == 1:
                    action = action.expand(obs.size(0), -1)
            action = action.to(device)
            combined = torch.cat([obs, action], dim=1)
        else:
            combined = obs

        if self.critic_trunk is None or self.value_head is None:
            # Build critic dynamically if needed
            input_dim = combined.size(-1)
            self.critic_trunk, self.value_head = self._build_critic_trunk_and_head(input_dim)
            self.critic_trunk = self.critic_trunk.to(device)
            self.value_head = self.value_head.to(device)

        features = self.critic_trunk(combined)
        q_value = self.value_head(features)
        return q_value

    def compute_pooled(self, obs, action: torch.Tensor = None) -> torch.Tensor:
        """Compute shared features up to the final hidden representation.

        Returns a tensor of shape [batch_size, hidden_dim] that is the input to the
        last value head layer. Mirrors TransformerCritic.compute_pooled semantics.
        """
        if self.preprocess_obs_transform is not None:
            for transform in self.preprocess_obs_transform:
                obs = transform(obs)

        obs = self._remove_obs_keys(obs)

        # Flatten/featurize observations
        obs = self._flatten_obs(obs)
        device = next(self.parameters()).device

        # Concatenate action if configured
        if self.config.use_action:
            if action is None:
                raise ValueError("Action must be provided when use_action=True")
            if isinstance(self.config.action_space, gym.spaces.Discrete):
                if action.dim() == 1:
                    action = action.unsqueeze(-1).float()
                elif action.dim() == 0:
                    action = action.unsqueeze(0).unsqueeze(-1).float()
                else:
                    action = action.float()
            else:
                if action.dim() > 2:
                    action = action.reshape(action.size(0), -1)
                elif action.dim() == 1:
                    action = action.unsqueeze(0)
            if obs.size(0) != action.size(0):
                if obs.size(0) == 1:
                    obs = obs.expand(action.size(0), -1)
                elif action.size(0) == 1:
                    action = action.expand(obs.size(0), -1)
            action = action.to(device)
            combined = torch.cat([obs, action], dim=1)
        else:
            combined = obs

        # Ensure critic trunk exists
        if self.critic_trunk is None or self.value_head is None:
            input_dim = combined.size(-1)
            self.critic_trunk, self.value_head = self._build_critic_trunk_and_head(input_dim)
            self.critic_trunk = self.critic_trunk.to(device)
            self.value_head = self.value_head.to(device)

        pooled = self.critic_trunk(combined)
        if self.output_mlp is not None:
            pooled = self.output_mlp(pooled)
        return pooled

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))
