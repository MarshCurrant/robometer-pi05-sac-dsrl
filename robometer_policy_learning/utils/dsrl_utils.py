"""
DSRL Utility Functions

Helper functions for DSRL training including observation processing,
batch merging, and data handling.
"""

from typing import Dict, List, Any, Optional
import torch
import numpy as np
from PIL import Image


def merge_obs_with_proprio(vlm_features: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
    """
    Merge VLM features with proprioception.

    Args:
        vlm_features: torch.Tensor of shape (B, 2048) - from Pi0 VLM
        proprio: torch.Tensor of shape (B, 8) - robot state

    Returns:
        merged: torch.Tensor of shape (B, 2056)
    """
    return torch.cat([vlm_features, proprio], dim=-1)


def prepare_obs_for_actor(obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Prepare observations in the format expected by DSRL actor.

    Args:
        obs: Dictionary with keys:
            - 'obs': torch.Tensor of shape (B, obs_dim) - VLM features + proprio
            - 'images': torch.Tensor of shape (B, H, W, 3) - raw images

    Returns:
        actor_input: Dictionary ready for actor forward pass
    """
    # Ensure images are in the right format and normalized
    images = obs["images"]

    # Convert to float and normalize if needed
    if images.dtype == torch.uint8:
        images = images.float() / 255.0

    # Rearrange from (B, H, W, C) to (B, C, H, W) if needed
    if images.dim() == 4 and images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)

    return {
        "obs": obs["obs"],
        "images": images,
    }


def compute_chunk_rewards(rewards: torch.Tensor, discount: float, action_len: int) -> torch.Tensor:
    """
    Compute cumulative discounted rewards over action chunks.

    Args:
        rewards: torch.Tensor of shape (B, action_len) - per-step rewards
        discount: Discount factor (gamma)
        action_len: Length of action chunks

    Returns:
        chunk_rewards: torch.Tensor of shape (B,) - cumulative rewards
    """
    # Create discount weights: [1, gamma, gamma^2, ..., gamma^(action_len-1)]
    discount_weights = torch.pow(discount, torch.arange(action_len, device=rewards.device))

    # Compute weighted sum
    chunk_rewards = (rewards * discount_weights).sum(dim=-1)

    return chunk_rewards


def to_numpy_dict(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """
    Convert dictionary of tensors to numpy arrays.

    Args:
        tensor_dict: Dictionary with torch.Tensor values

    Returns:
        numpy_dict: Dictionary with np.ndarray values
    """
    numpy_dict = {}
    for k, v in tensor_dict.items():
        if isinstance(v, dict):
            numpy_dict[k] = to_numpy_dict(v)
        elif isinstance(v, torch.Tensor):
            numpy_dict[k] = v.detach().cpu().numpy()
        else:
            numpy_dict[k] = v
    return numpy_dict


def to_torch_dict(numpy_dict: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Convert dictionary of numpy arrays to torch tensors.

    Args:
        numpy_dict: Dictionary with np.ndarray values
        device: Target device for tensors

    Returns:
        tensor_dict: Dictionary with torch.Tensor values
    """
    tensor_dict = {}
    for k, v in numpy_dict.items():
        if isinstance(v, dict):
            tensor_dict[k] = to_torch_dict(v, device)
        elif isinstance(v, np.ndarray):
            tensor_dict[k] = torch.from_numpy(v).to(device)
        elif isinstance(v, torch.Tensor):
            tensor_dict[k] = v.to(device)
        else:
            tensor_dict[k] = v
    return tensor_dict


def format_log_dict(log_dict: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Format logging dictionary with prefix.

    Args:
        log_dict: Dictionary of metrics
        prefix: Prefix to add to keys (e.g., "dsrl/train/")

    Returns:
        formatted: Dictionary with prefixed keys
    """
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    formatted = {}
    for k, v in log_dict.items():
        # Convert tensors to scalars
        if isinstance(v, torch.Tensor):
            v = v.item() if v.numel() == 1 else v.detach().cpu().numpy()
        formatted[prefix + k] = v

    return formatted


class ActionQueue:
    """
    Queue for managing action chunks in DSRL.

    Pi0 predicts chunks of actions (e.g., 10 steps), but we execute them
    one at a time. This queue manages the buffering.
    """

    def __init__(self, n_envs: int):
        """
        Initialize action queues.

        Args:
            n_envs: Number of parallel environments
        """
        self.n_envs = n_envs
        self.queues = [[] for _ in range(n_envs)]

    def add(self, env_idx: int, actions: np.ndarray):
        """
        Add actions to queue for specific environment.

        Args:
            env_idx: Environment index
            actions: np.ndarray of shape (action_len, action_dim) - action chunk
        """
        self.queues[env_idx].extend(list(actions))

    def pop(self, env_idx: int) -> Optional[np.ndarray]:
        """
        Pop next action from queue.

        Args:
            env_idx: Environment index

        Returns:
            action: np.ndarray of shape (action_dim,) or None if empty
        """
        if len(self.queues[env_idx]) > 0:
            return self.queues[env_idx].pop(0)
        return None

    def is_empty(self, env_idx: int) -> bool:
        """Check if queue is empty for given environment"""
        return len(self.queues[env_idx]) == 0

    def get_empty_env_ids(self) -> List[int]:
        """Get list of environment indices with empty queues"""
        return [i for i in range(self.n_envs) if self.is_empty(i)]

    def clear(self, env_idx: int):
        """Clear queue for specific environment"""
        self.queues[env_idx] = []

    def clear_all(self):
        """Clear all queues"""
        self.queues = [[] for _ in range(self.n_envs)]


def format_obs_for_storage(
    vlm_features: torch.Tensor, images: np.ndarray, proprio: Optional[torch.Tensor] = None
) -> Dict[str, List[np.ndarray]]:
    """
    Format observations for storage in replay buffer.

    Args:
        vlm_features: torch.Tensor of shape (B, 2048) - VLM features
        images: np.ndarray of shape (B, H, W, 3) - images
        proprio: Optional torch.Tensor of shape (B, 8) - proprioception

    Returns:
        obs_dict: Dictionary ready for buffer storage
    """
    # Merge VLM features with proprio
    if proprio is not None:
        obs = torch.cat([vlm_features, proprio], dim=-1)
    else:
        obs = vlm_features

    # Convert to numpy
    obs_np = obs.detach().cpu().numpy()

    # Split into list (one per environment)
    obs_list = list(obs_np)
    images_list = list(images)

    return {
        "obs": obs_list,
        "images": images_list,
    }


def compute_multistep_return(
    rewards: np.ndarray, terminals: np.ndarray, discount: float, action_len: int
) -> np.ndarray:
    """
    Compute multi-step returns for trajectory.

    Args:
        rewards: np.ndarray of shape (T,) - per-step rewards
        terminals: np.ndarray of shape (T,) - terminal flags
        discount: Discount factor
        action_len: Number of steps to look ahead

    Returns:
        returns: np.ndarray of shape (T,) - multi-step returns
    """
    T = len(rewards)
    returns = np.zeros(T)

    for t in range(T):
        ret = 0
        for k in range(min(action_len, T - t)):
            ret += (discount**k) * rewards[t + k]
            if terminals[t + k]:
                break
        returns[t] = ret

    return returns


def resize_images(images: np.ndarray, target_size: int = 64) -> np.ndarray:
    """Resize images to target size"""
    B, H, W, C = images.shape
    if H == target_size and W == target_size:
        return images

    resized = []
    for img in images:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((target_size, target_size))
        resized.append(np.array(pil_img))
    return np.array(resized)
