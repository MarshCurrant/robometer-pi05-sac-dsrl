import gymnasium as gym
import numpy as np
from typing import Dict


def center_crop(image, size):
    """Center crop an image or a batch of images to the specified size.

    Supports both 3D (H, W, C) and 4D (N, H, W, C) arrays.
    """
    if image is None or size is None:
        return image

    if image.ndim == 3:
        h, w = image.shape[:2]
        crop = min(size, h, w)
        x = (w - crop) // 2
        y = (h - crop) // 2
        return image[y : y + crop, x : x + crop, :]
    elif image.ndim == 4:
        h, w = image.shape[1:3]
        crop = min(size, h, w)
        x = (w - crop) // 2
        y = (h - crop) // 2
        return image[:, y : y + crop, x : x + crop, :]
    else:
        return image


class FlatToDictObsWrapper(gym.ObservationWrapper):
    """
    Wraps a flat observation (e.g., np.ndarray) into a dict observation
    with key 'state'.
    """

    def __init__(self, env):
        super().__init__(env)
        # The new observation space is a Dict with a single key
        self.observation_space = gym.spaces.Dict({"state": env.observation_space})

    def observation(self, obs):
        return {"state": obs}


class ImageDictObsWrapper(gym.ObservationWrapper):
    """
    Wraps an environment to provide proprioceptive state (under 'state')
    and the rendered image (under 'image') in a dictionary.
    Note: the rendered image is always flipped vertically.
    Captures proprioceptive information as the first 4 elements of the original observation.
    """

    def __init__(self, env: gym.Env, size: int = 224):
        super().__init__(env)
        self.size = size

        # Update observation space to include both 'state' and 'image'
        self._update_observation_space()

    def _update_observation_space(self):
        """Update the observation space to include both the proprioceptive state (first 4 elements) and rendered image."""
        orig_space = self.env.observation_space
        # Assume original state is a Box with at least 4 elements
        if isinstance(orig_space, gym.spaces.Box):
            # New state space is just 4D slice of original
            state_low = orig_space.low[:4]
            state_high = orig_space.high[:4]
            state_shape = (4,)
            state_space = gym.spaces.Box(low=state_low, high=state_high, shape=state_shape, dtype=orig_space.dtype)
        else:
            # Fallback to using the original observation space (may need refinement for other types)
            state_space = orig_space

        self.observation_space = gym.spaces.Dict(
            {
                "state": state_space,
                "image": gym.spaces.Box(low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8),
            }
        )

    def _render_image(self) -> np.ndarray:
        """Render an image from the environment."""
        image = self.env.render()

        # Ensure the image is the right shape and type
        if image is None:
            raise ValueError("Failed to render image")
        image = np.flipud(image)

        # Perform center crop
        image = center_crop(image, self.size)
        image = image.copy()

        return image

    def observation(self, obs) -> dict:
        """
        Return a dict with:
        - 'state': the first 4 elements of the raw observation (proprioceptive info)
        - 'image': the rendered image
        """
        # Get first 4 elements for proprioceptive state
        state_obs = obs[:4] if isinstance(obs, np.ndarray) and obs.shape[0] >= 4 else obs
        return {"state": state_obs, "image": self._render_image()}
