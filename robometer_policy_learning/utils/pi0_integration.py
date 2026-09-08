"""
Pi0 Integration Utilities for DSRL

This module provides utilities to load and use the Pi0 policy
from the openpi library within the PyTorch-based rfm_rl framework.
"""

from typing import Dict, Any
import copy
import torch
import numpy as np
from openpi_client import image_tools

from scipy.interpolate import interp1d


def load_pi0_policy(checkpoint_dir: str, config_name: str | None = None):
    """
    Load a pre-trained Pi0 policy from checkpoint directory.

    Args:
        checkpoint_dir: Path to the Pi0 checkpoint directory

    Returns:
        Loaded Pi0 policy object (JAX-based) that can be called from PyTorch
    """
    from openpi.policies import policy_config
    from openpi.training import config

    # Prefer an explicit config. Path-name guessing is retained for the
    # repository's existing DROID/LIBERO/Bridge entry points.
    if config_name is not None:
        pi0_config = config.get_config(config_name)
    elif "libero" in checkpoint_dir:
        if "pi05" in checkpoint_dir:
            pi0_config = config.get_config("pi05_libero")
        else:
            pi0_config = config.get_config("pi0_libero")
    elif "bridge" in checkpoint_dir:
        if "pi05" in checkpoint_dir:
            raise ValueError("pi05_bridge is not supported for now")
        else:
            if "lora" in checkpoint_dir:
                pi0_config = config.get_config("pi0_lora_bridge_1_cam")
            else:
                pi0_config = config.get_config("pi0_bridge_1_cam")
    elif "droid" in checkpoint_dir:
        if "jointpos" in checkpoint_dir:
            pi0_config = config.get_config("pi05_droid_jointpos")
        elif "pi05" in checkpoint_dir:
            pi0_config = config.get_config("pi05_droid")
        else:
            pi0_config = config.get_config("pi0_droid")
    else:
        raise ValueError(f"Invalid checkpoint directory: {checkpoint_dir}")

    # Create the trained policy
    policy = policy_config.create_trained_policy(pi0_config, checkpoint_dir)

    print(f"✓ Successfully loaded Pi0 policy from {checkpoint_dir}")

    return policy


def extract_vlm_features(pi0_policy, observations: Dict[str, Any]) -> torch.Tensor:
    """
    Extract VLM hidden states from Pi0's vision-language model.

    Args:
        pi0_policy: Loaded Pi0 policy object
        observations: Dictionary containing observation data
            - 'observation/image': np.ndarray of shape (B, 224, 224, 3)
            - 'observation/wrist_image': np.ndarray of shape (B, 224, 224, 3)
            - 'observation/state': np.ndarray of shape (B, 8)
            - 'prompt': List[str] or str - language instruction(s)

    Returns:
        vlm_features: torch.Tensor of shape (B, 2048) - VLM hidden states
    """
    # Call Pi0's get_prefix_rep to extract VLM features
    with torch.no_grad():
        # Pi0 returns JAX arrays, so we'll convert to PyTorch
        vlm_hidden_states, _ = pi0_policy.get_prefix_rep(observations)

        # Take the last token (sequence position -1)
        # Shape: (B, seq_len, 2048) -> (B, 2048)
        vlm_hidden_states = vlm_hidden_states[:, -1, :]

        # Convert JAX array to PyTorch tensor
        if hasattr(vlm_hidden_states, "__array__"):
            # Convert to numpy array, then to float32 to handle bfloat16
            vlm_hidden_states = np.array(vlm_hidden_states, dtype=np.float32)
            vlm_hidden_states = torch.from_numpy(vlm_hidden_states)
        elif not isinstance(vlm_hidden_states, torch.Tensor):
            vlm_hidden_states = torch.tensor(vlm_hidden_states, dtype=torch.float32)
        else:
            vlm_hidden_states = vlm_hidden_states.float()

    return vlm_hidden_states


def pi0_infer_with_noise(
    pi0_policy,
    observations: Dict[str, Any],
    noise: np.ndarray | None,
) -> Dict[str, np.ndarray]:
    """
    Run Pi0 inference with injected noise for DSRL steering.

    Args:
        pi0_policy: Loaded Pi0 policy object
        observations: Dictionary containing observation data (same format as extract_vlm_features)
        noise: np.ndarray of shape (B, noise_dim) - noise to inject

    Returns:
        result: Dictionary containing:
            - 'actions': np.ndarray of shape (B, action_len, 7) - predicted robot actions
    """
    # Ensure noise is in the shape expected by Pi0.
    # Pi0's Policy.infer passes `noise` directly to `Pi0.sample_actions`,
    # which expects shape (B, action_horizon, action_dim).
    # Our SAC actor currently outputs noise of shape (B, action_dim),
    # so we broadcast it along the action_horizon dimension.
    # Use Pi0 policy metadata to determine horizon and action_dim.
    if noise is None:
        print("no noise passed, doing regular pi0/pi0.5 inference")
    else:
        horizon = getattr(pi0_policy, "action_horizon", None)
        #print(f"horizon: {horizon}, noise.shape: {noise.shape}, noise.min: {noise.min()}, noise.max: {noise.max()}")
        if horizon is None:
            raise ValueError("Pi0 policy is missing `action_horizon` attribute; cannot reshape noise to (B, T, D).")
        if noise.ndim == 2:
            # Repeat noise along the time (horizon) axis.
            noise = np.repeat(noise[:, None, :], horizon, axis=1)
        elif noise.ndim == 3:
            if horizon - noise.shape[1] > 0:
                num_to_repeat = horizon - noise.shape[1]

                # Repeating the full existing noise sequence
                if num_to_repeat > 0:
                    # Repeat the entire noise sequence along the time axis to fill the gap
                    full_repeats = num_to_repeat // noise.shape[1]
                    remainder = num_to_repeat % noise.shape[1]

                    repeat_part = []
                    if full_repeats > 0:
                        repeat_part.append(np.tile(noise, (1, full_repeats, 1)))
                    if remainder > 0:
                        repeat_part.append(noise[:, :remainder, :])
                    if repeat_part:
                        repeated_noise = np.concatenate(repeat_part, axis=1)
                        noise = np.concatenate([noise, repeated_noise], axis=1)

                # # Noise Interpolation Method
                # # Interpolate the noise sequence along the time (horizon) axis to match horizon length
                # # noise shape: (B, T0, D) where T0 < horizon
                # B, T0, D = noise.shape
                # # Original time indices
                # old_time = np.linspace(0, 1, T0)
                # new_time = np.linspace(0, 1, horizon)
                # # Interpolate along axis 1 (time/horizon)
                # interp_noise = []
                # for i in range(B):
                #     f = interp1d(old_time, noise[i], axis=0, kind='linear', fill_value="extrapolate")
                #     interp_noise.append(f(new_time))
                # noise = np.stack(interp_noise, axis=0)
                # # (Optional) Shuffle elements along time-axis for each batch as in the original code
                # for i in range(B):
                #     np.random.shuffle(noise[i])

                # # Last Noise Vector Repeating Method
                # repeated_noise = np.repeat(noise[:, -1:, :], num_to_repeat, axis=1)
                # noise = np.concatenate([noise, repeated_noise], axis=1)
            elif horizon - noise.shape[1] < 0:
                raise ValueError(
                    f"Pi0 noise must have ndim 2 or 3, with at most length {horizon} along the time axis, shape {noise.shape} "
                    "(after conversion to numpy)."
                )

    # Prepare inference kwargs
    infer_kwargs = {"noise": noise}

    # Run Pi0 inference with noise injection
    result = pi0_policy.infer(observations, **infer_kwargs)

    # Convert actions to float32 to handle bfloat16 from JAX/TPU models
    if "actions" in result:
        result["actions"] = np.array(result["actions"], dtype=np.float32)

    return result


def preprocess_obs_for_pi0(raw_obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess observations from LIBERO environment to Pi0 format.

    Args:
        raw_obs: Dictionary from LIBERO environment
            Keys may include: 'agentview_image', 'robot0_eye_in_hand_image',
                            'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'

    Returns:
        pi0_obs: Dictionary in Pi0's expected format
    """
    from openpi_client import image_tools

    # Helper function to convert quaternion to axis-angle
    def quat2axisangle(quat):
        """Convert quaternion to axis-angle representation"""
        import math

        # Clip quaternion
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0

        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)

        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    # Process images: flip both horizontally and vertically (Pi0 convention)
    img = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1])

    # Resize and convert to uint8
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))

    # Process state: [eef_pos (3), eef_axis_angle (3), gripper_qpos (2)] = 8-dim
    if "robot0_eef_quat" in raw_obs:
        state = np.concatenate(
            [raw_obs["robot0_eef_pos"], quat2axisangle(raw_obs["robot0_eef_quat"]), raw_obs["robot0_gripper_qpos"]]
        )
    elif "robot0_eef_ori" in raw_obs:
        state = np.concatenate(
            [raw_obs["robot0_eef_pos"], raw_obs["robot0_eef_ori"], raw_obs["robot0_gripper_qpos"]]
        )
    else:
        raise ValueError(f"No eef orientation key found. Available keys: {raw_obs.keys()}")

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": raw_obs.get("prompt", raw_obs.get("language_instruction", "unknown task")),
    }


class Pi0Wrapper:
    """
    Wrapper class for Pi0 policy to manage state and provide clean interface.
    """

    def __init__(self, checkpoint_dir: str, device: str = "cuda", config_name: str | None = None):
        """
        Initialize Pi0 wrapper.

        Args:
            checkpoint_dir: Path to Pi0 checkpoint
            device: PyTorch device (Pi0 itself runs in JAX, but this is for feature conversion)
        """
        self.config_name = config_name
        self.policy = load_pi0_policy(checkpoint_dir, config_name=config_name)
        self.device = torch.device(device)

        # Freeze Pi0 - it should never be trained
        if hasattr(self.policy, "_model") and hasattr(self.policy._model, "parameters"):
            for param in self.policy._model.parameters():
                param.requires_grad = False

        print(f"✓ Pi0 wrapper initialized (device: {device})")

    def _prepare_observations(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        if self.config_name == "pi0_base_aloha_robotwin_lora":
            def unbatch_one(value):
                value = np.asarray(value)
                if value.ndim > 0 and value.shape[0] == 1:
                    return value[0]
                return value

            def aloha_image(key):
                image = unbatch_one(observations[key])
                if image.ndim != 3:
                    raise ValueError(f"Expected one RoboTwin image for {key}, got shape {image.shape}")
                if image.shape[-1] in (3, 4):
                    image = image_tools.resize_with_pad(image, 224, 224)
                    image = np.transpose(image, (2, 0, 1))
                elif image.shape[0] in (3, 4):
                    image = np.transpose(image, (1, 2, 0))
                    image = image_tools.resize_with_pad(image, 224, 224)
                    image = np.transpose(image, (2, 0, 1))
                else:
                    raise ValueError(f"Expected HWC or CHW image for {key}, got shape {image.shape}")
                return np.asarray(image, dtype=np.uint8)

            state = unbatch_one(observations["observation.state"])
            if state.shape != (14,):
                raise ValueError(f"Expected one 14-D RoboTwin ALOHA state, got shape {state.shape}")

            prompt = observations.get("prompt", "")
            if isinstance(prompt, np.ndarray):
                prompt = prompt.item() if prompt.ndim == 0 else prompt.reshape(-1)[0]
            elif isinstance(prompt, (list, tuple)):
                prompt = prompt[0]

            # OpenPI's inference-time AlohaInputs consumes this canonical
            # structure directly. Training-only RepackTransform entries are
            # intentionally not applied by create_trained_policy().
            return {
                "images": {
                    "cam_high": aloha_image("observation.images.cam_high"),
                    "cam_left_wrist": aloha_image("observation.images.cam_left_wrist"),
                    "cam_right_wrist": aloha_image("observation.images.cam_right_wrist"),
                },
                "state": np.asarray(state, dtype=np.float32),
                "prompt": str(prompt),
            }

        prepared = copy.deepcopy(observations)
        for key, value in prepared.items():
            prepared[key] = self.resize_images_if_image(value)
        return prepared

    def get_features(self, observations: Dict[str, Any]) -> torch.Tensor:
        """Extract VLM features from observations"""
        # copy the obs dict
        pi0_obs = self._prepare_observations(observations)
        vlm_features = extract_vlm_features(self.policy, pi0_obs)
        return vlm_features.to(self.device)

    @property
    def action_horizon(self):
        return getattr(self.policy, "action_horizon", None)

    def resize_images_if_image(self, images: np.ndarray) -> np.ndarray:
        """Resize images to 224x224"""
        if isinstance(images, np.ndarray) and images.ndim == 4 and images.shape[-1] in (3, 4):
            return image_tools.resize_with_pad(images, 224, 224)
        if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[-1] in (3, 4):
            return image_tools.resize_with_pad(images.cpu().numpy(), 224, 224)
        # not an image, return as is
        return images

    def infer(
        self,
        observations: Dict[str, Any],
        noise: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Run inference with noise steering"""
        return pi0_infer_with_noise(self.policy, self._prepare_observations(observations), noise)

    def __call__(self, observations, noise):
        """Convenience method for inference"""
        return self.infer(observations, noise)
