import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import copy
from typing import Dict, List, Optional, Union

try:
    # Reuse helper from transformer utilities to identify image-like keys
    from robometer_policy_learning.modules.transformer.transformer_utils import identify_image_keys
except Exception:

    def identify_image_keys(obs_keys: List[str]) -> List[str]:
        image_keywords = ["image", "rgb", "camera", "vision", "visual"]
        return [k for k in obs_keys if any(s in k.lower() for s in image_keywords)]


class FlattenFeaturizerWrapper(nn.Module):
    """Wrap a module so it always receives a flattened tensor (B, D)."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            elif x.dim() == 1:
                x = x.unsqueeze(0)
        return self.module(x)


class DinoImageFeaturizer(nn.Module):
    """
    Image featurizer using a DINOv2 model. Accepts tensors in (B, H, W, C) or (B, C, H, W)
    and returns a (B, D) embedding using pooler_output or CLS token.
    """

    def __init__(
        self,
        dinov2_model: nn.Module,
        dinov2_processor,
        normalize_inputs: bool = True,
        requires_grad: bool = False,
    ):
        super().__init__()
        self.dino = dinov2_model
        self.processor = dinov2_processor
        self.normalize_inputs = normalize_inputs

        # Typically we use DINO as a frozen encoder
        for p in self.dino.parameters():
            p.requires_grad = requires_grad

        self.dino.eval()

    @property
    def device(self):
        return next(self.dino.parameters()).device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure batch dimension and channel-last for processor
        if not isinstance(x, torch.Tensor):
            raise ValueError("DinoImageFeaturizer expects torch.Tensor inputs")

        # Accept (H, W, C) or (C, H, W) by adding batch dim
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Convert to (B, H, W, C)
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got shape {tuple(x.shape)}")
        if x.size(-1) == 3:  # already NHWC
            images = x
        elif x.size(1) in (1, 3):  # NCHW -> NHWC
            images = x.permute(0, 2, 3, 1)
        else:
            # Try to infer last dim as channels; fallback to treat as NHWC
            images = x

        # Convert to CPU numpy for the processor for maximum compatibility
        images = images.detach().to("cpu")

        # Scale to [0, 255] uint8 if values appear to be 0..1 floats
        img_np_list: List[np.ndarray] = []
        for i in range(images.size(0)):
            arr = images[i].numpy()
            if self.normalize_inputs:
                if arr.dtype != np.uint8:
                    if arr.max() <= 1.0:
                        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        arr = arr.clip(0, 255).astype(np.uint8)
            img_np_list.append(arr)

        processed = self.processor(images=img_np_list, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.device)

        outputs = self.dino(pixel_values=pixel_values)

        emb = outputs.pooler_output

        return emb


class ImpalaImageFeaturizer(nn.Module):
    """
    Image featurizer using IMPALA encoder. Accepts tensors in (B, H, W, C) or (B, C, H, W)
    and returns a (B, D) embedding.
    """

    def __init__(
        self,
        input_shape: tuple,
        nn_scale: int = 1,
        num_blocks_per_stack: int = 2,
        use_smaller: bool = False,
        requires_grad: bool = True,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        from robometer_policy_learning.modules.cnn import ImpalaEncoder, SmallerImpalaEncoder

        if use_smaller:
            self.encoder = SmallerImpalaEncoder(
                input_shape=input_shape,
                nn_scale=nn_scale,
                output_dim=output_dim,
            )
        else:
            self.encoder = ImpalaEncoder(
                input_shape=input_shape,
                nn_scale=nn_scale,
                num_blocks_per_stack=num_blocks_per_stack,
                output_dim=output_dim,
            )

        # Set requires_grad for all parameters
        for p in self.encoder.parameters():
            p.requires_grad = requires_grad

        if not requires_grad:
            self.encoder.eval()

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @property
    def output_dim(self):
        return self.encoder.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through IMPALA encoder.

        Args:
            x: Input tensor of shape (B, H, W, C), (B, C, H, W), (H, W, C), or (C, H, W)

        Returns:
            Feature tensor of shape (B, output_dim)
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError("ImpalaImageFeaturizer expects torch.Tensor inputs")

        # Ensure batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")

        # Move to device
        x = x.to(self.device)

        # Forward through encoder (handles format conversion and normalization internally)
        features = self.encoder(x)

        return features


def build_mlp_dino_featurizers(
    observation_space,
    dinov2_model: nn.Module,
    dinov2_processor,
    keys: Optional[List[str]] = None,
    requires_grad: bool = False,
) -> Dict[str, nn.Module]:
    """
    Build a dict of featurizers for MLP models that maps image-like observation keys
    to DINO image featurizers. Returns a plain dict suitable for `featurizer` config.
    """
    if hasattr(observation_space, "spaces"):
        obs_keys = list(observation_space.spaces.keys())
    else:
        obs_keys = ["obs"]

    image_keys = keys if keys is not None else identify_image_keys(obs_keys)
    featurizers: Dict[str, nn.Module] = {}
    for k in image_keys:
        featurizers[k] = DinoImageFeaturizer(
            dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor,
            requires_grad=requires_grad,
        )
    return featurizers


def build_mlp_impala_featurizers(
    observation_space,
    keys: Optional[List[str]] = None,
    nn_scale: int = 1,
    num_blocks_per_stack: int = 2,
    use_smaller: bool = False,
    requires_grad: bool = True,
    output_dim: Optional[int] = None,
) -> Dict[str, nn.Module]:
    """
    Build a dict of featurizers for MLP models that maps image-like observation keys
    to IMPALA image featurizers. Returns a plain dict suitable for `featurizer` config.

    Args:
        observation_space: Gym observation space
        keys: Optional list of keys to create featurizers for. If None, auto-detects image keys.
        nn_scale: Scaling factor for channel sizes (default: 1)
        num_blocks_per_stack: Number of residual blocks per stack (default: 2)
        use_smaller: Whether to use SmallerImpalaEncoder variant (default: False)
        requires_grad: Whether encoder parameters should be trainable (default: True)
        output_dim: Optional output dimension. If provided, adds a linear projection layer
                    to map from pooled features to this dimension (default: None)

    Returns:
        Dictionary mapping observation keys to ImpalaImageFeaturizer instances
    """
    if hasattr(observation_space, "spaces"):
        obs_keys = list(observation_space.spaces.keys())
    else:
        obs_keys = ["obs"]

    image_keys = keys if keys is not None else identify_image_keys(obs_keys)
    featurizers: Dict[str, nn.Module] = {}
    for k in image_keys:
        if isinstance(observation_space, gym.spaces.Dict):
            input_shape = observation_space.spaces[k].shape
        else:
            input_shape = observation_space.shape

        featurizers[k] = ImpalaImageFeaturizer(
            input_shape=input_shape,
            nn_scale=nn_scale,
            num_blocks_per_stack=num_blocks_per_stack,
            use_smaller=use_smaller,
            requires_grad=requires_grad,
            output_dim=output_dim,
        )
    return featurizers


def _build_mlp_layers(input_size, hidden_dims, activation, use_layer_norm=False, dropout_rate=0.0):
    """Build MLP layers for featurizers."""
    layers = []
    prev_size = int(input_size)
    for hidden_size in hidden_dims:
        hidden_size = int(hidden_size)
        layers.append(nn.Linear(prev_size, hidden_size))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        if activation.lower() == "relu":
            layers.append(nn.ReLU())
        elif activation.lower() == "tanh":
            layers.append(nn.Tanh())
        elif activation.lower() == "elu":
            layers.append(nn.ELU())
        elif activation.lower() == "leaky_relu":
            layers.append(nn.LeakyReLU())
        else:
            raise ValueError(f"Unknown activation: {activation}")
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        prev_size = hidden_size
    return layers


class ObservationFeaturizer(nn.Module):
    """
    Common featurizer for MLP-based models that handles observation flattening and featurization.
    Supports dict obs with per-key featurizers and automatic flattening.

    Can automatically use IMPALA encoders for image keys when configured via image_encoder_type.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        featurizer_cfg: Optional[Dict[str, Union[List, nn.Module]]] = None,
        activation: str = "relu",
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        # IMPALA encoder parameters (optional)
        image_encoder_type: Optional[str] = None,  # "impala" to enable IMPALA for image keys
        impala_nn_scale: int = 1,
        impala_num_blocks_per_stack: int = 2,
        impala_use_smaller: bool = False,
        impala_output_dim: int = None,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.featurizer_cfg = copy.deepcopy(featurizer_cfg)

        # If image_encoder_type is "impala", build IMPALA featurizers for image keys
        # and merge with existing featurizer_cfg
        if image_encoder_type == "impala":
            if not isinstance(observation_space, gym.spaces.Dict):
                raise ValueError("IMPALA encoder requires Dict observation space")

            # Build IMPALA featurizers for image keys
            impala_featurizers = build_mlp_impala_featurizers(
                observation_space=observation_space,
                keys=None,  # Auto-detect image keys
                nn_scale=impala_nn_scale,
                num_blocks_per_stack=impala_num_blocks_per_stack,
                use_smaller=impala_use_smaller,
                requires_grad=True,
                output_dim=impala_output_dim,
            )

            # Merge with existing featurizer_cfg (featurizer_cfg takes precedence)
            if self.featurizer_cfg is None:
                self.featurizer_cfg = {}

            # Update featurizer_cfg with IMPALA featurizers for image keys not already specified
            image_keys = identify_image_keys(list(observation_space.spaces.keys()))
            for key, value in impala_featurizers.items():
                self.featurizer_cfg[key] = value.output_dim

            # combine self.featurizers and impala image featurizer
            self.featurizers = nn.ModuleDict()
            for key, value in impala_featurizers.items():
                self.featurizers[key] = value

        else:
            self.featurizers = nn.ModuleDict() if self.featurizer_cfg else None

        # Build featurizers if config provided
        if self.featurizer_cfg:
            for key, value in self.featurizer_cfg.items():
                if key not in self.featurizers:
                    if isinstance(value, (list, tuple)):
                        # Build an MLP featurizer for this key
                        obs_dim = int(np.prod(observation_space.spaces[key].shape))
                        mlp_seq = _build_mlp_layers(
                            obs_dim,
                            value,
                            activation,
                            use_layer_norm,
                            dropout_rate,
                        )
                        # Wrap in Sequential and FlattenFeaturizerWrapper
                        self.featurizers[key] = FlattenFeaturizerWrapper(nn.Sequential(*mlp_seq))
                    elif isinstance(value, nn.Module):
                        self.featurizers[key] = value
                    else:
                        raise ValueError(f"Featurizer for key {key} must be list/tuple or nn.Module")

        # Determine flatten keys for non-featurizer case
        if self.featurizer_cfg:
            # When using featurizers, only use keys in featurizer_cfg
            self._flatten_keys = list(self.featurizer_cfg.keys())
        elif isinstance(observation_space, gym.spaces.Dict):
            self._flatten_keys = [
                k for k, space in observation_space.spaces.items() if getattr(space, "shape", None) is not None
            ]
        else:
            self._flatten_keys = None

        self._output_dim = self._compute_output_dim()

    def _create_example_obs(self) -> Union[dict, torch.Tensor, None]:
        """Create a dummy observation matching the observation space."""
        if isinstance(self.observation_space, gym.spaces.Dict):
            example_obs = {}
            # If using featurizers, only create examples for keys in featurizer_cfg
            keys_to_process = self._flatten_keys if self.featurizer_cfg else list(self.observation_space.spaces.keys())
            for key in keys_to_process:
                space = self.observation_space.spaces.get(key)
                if space is None:
                    continue
                shape = getattr(space, "shape", None)
                if shape is None:
                    continue
                example_obs[key] = torch.zeros((1,) + shape, dtype=torch.float32)
            return example_obs if example_obs else None
        else:
            shape = getattr(self.observation_space, "shape", None)
            if shape is None:
                return None
            return torch.zeros((1,) + shape, dtype=torch.float32)

    def _compute_output_dim(self) -> int:
        """Infer the flattened observation dimensionality."""
        if self.featurizer_cfg:
            # If using featurizers, compute by passing example through featurizer
            example_obs = self._create_example_obs()
            if example_obs is None:
                return 0
            with torch.no_grad():
                flattened = self.flatten_obs(example_obs)
            return int(flattened.shape[-1]) if flattened.numel() > 0 else 0
        else:
            # If no featurizers, use direct computation (old logic)
            if isinstance(self.observation_space, gym.spaces.Dict):
                if self._flatten_keys is not None:
                    return sum(int(np.prod(self.observation_space.spaces[k].shape)) for k in self._flatten_keys)
                else:
                    return 0
            else:
                return int(np.prod(self.observation_space.shape))

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def flatten_obs(self, obs: Union[dict, torch.Tensor], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Flatten and optionally featurize observations.

        Args:
            obs: Observation dict or tensor
            device: Device to move tensors to (if None, uses device from first parameter)

        Returns:
            Flattened observation tensor of shape [batch, features]
        """
        if isinstance(obs, dict):
            if self.featurizer_cfg:
                feats = []
                # Only process keys that are in featurizer_cfg
                for k in self._flatten_keys:
                    if k not in obs:
                        # Skip missing keys - they should be in the observation
                        continue
                    v = obs[k]
                    if k in self.featurizers:
                        # Move to device if provided
                        if device is not None:
                            v = v.to(device).float()
                        feats.append(self.featurizers[k](v))
                    else:
                        # This shouldn't happen if featurizer_cfg is set up correctly
                        # But fallback to flattening if needed
                        v_flat = v.reshape(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0)
                        if device is not None:
                            v_flat = v_flat.to(device)
                        feats.append(v_flat)
                if feats:
                    return torch.cat(feats, dim=-1)
                else:
                    raise ValueError(
                        f"No valid features found in observation dict. "
                        f"Keys present: {list(obs.keys())}, expected featurizers: {list(self.featurizers.keys()) if self.featurizers else None}"
                    )
            elif self._flatten_keys is not None and len(self._flatten_keys) > 0:
                feats = []
                batch_size = None
                for k in self._flatten_keys:
                    v = obs.get(k, None)
                    shape = self.observation_space.spaces[k].shape
                    flat_dim = int(np.prod(shape))
                    if v is None:
                        # Fill with zeros if missing
                        if batch_size is None:
                            # Try to infer batch size from any present value
                            for vv in obs.values():
                                if vv is not None:
                                    batch_size = vv.size(0) if vv.dim() > 1 else 1
                                    break
                            if batch_size is None:
                                batch_size = 1
                        zero_tensor = torch.zeros(batch_size, flat_dim)
                        if device is not None:
                            zero_tensor = zero_tensor.to(device)
                        feats.append(zero_tensor)
                    else:
                        v_flat = v.reshape(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0)
                        if device is not None:
                            v_flat = v_flat.to(device)
                        feats.append(v_flat)
                return torch.cat(feats, dim=-1) if feats else torch.empty(0)
            else:
                # Just flatten and concatenate all values (skip None shapes)
                feats = []
                for v in obs.values():
                    if v is not None:
                        v_flat = v.reshape(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0)
                        if device is not None:
                            v_flat = v_flat.to(device)
                        feats.append(v_flat)
                return torch.cat(feats, dim=-1) if feats else torch.empty(0)
        else:
            # Tensor observation
            if obs.dim() > 2:
                result = obs.reshape(obs.size(0), -1)
            elif obs.dim() == 1:
                result = obs.unsqueeze(0)
            elif obs.dim() == 2:
                # Ensure shape is (batch, features); if it's (features, 1), transpose
                if obs.size(1) == 1 and obs.size(0) != 1:
                    result = obs.t()
                else:
                    result = obs
            else:
                result = obs
            if device is not None:
                result = result.to(device)
            return result
