import gymnasium as gym
import gymnasium.vector as gym_vector
from robometer_policy_learning.envs.gym_compat import VectorWrapperBase
import numpy as np
import torch
from typing import List, Optional
from transformers import AutoModel, AutoImageProcessor
from robometer.utils.embedding_utils import compute_video_embeddings


class DinoEmbeddingWrapper(gym.ObservationWrapper):
    """
    Wrapper that computes DINO embeddings for image observations and replaces them with embeddings.
    This ensures observations are in the same format as the buffer (with precomputed embeddings).

    Args:
        env: The environment to wrap
        dinov2_model: DINOv2 model for computing embeddings
        dinov2_processor: DINOv2 processor for preprocessing images
        image_keys: List of observation keys that contain images (default: auto-detect)
    """

    def __init__(
        self, env, dinov2_model, dinov2_processor, device: Optional[str] = None, image_keys: Optional[List[str]] = None
    ):
        super().__init__(env)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load model and processor internally
        self.dinov2_model = dinov2_model
        self.dinov2_processor = dinov2_processor

        self.image_keys = image_keys if image_keys is not None else ["image"]
        # to not have a warning about the model images already being scaled
        fake_image = torch.ones((1, 3, 84, 84)).to(device).float() * 2
        fake_image = self.dinov2_processor(images=fake_image, return_tensors="pt")
        fake_image = {k: v.to(device, non_blocking=True) for k, v in fake_image.items()}
        fake_image = self.dinov2_model(**fake_image)
        self.output_dim = fake_image.pooler_output.shape[1] * len(image_keys)
        # Update observation space - add "dino_embedding" space while keeping original keys
        if isinstance(env.observation_space, gym.spaces.Dict):
            new_spaces = dict(env.observation_space.spaces)
            # Add "dino_embedding" key (DINOv2-base outputs self.output_dim-dim embeddings)
            new_spaces["dino_embedding"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.output_dim,),
                dtype=np.float32,
            )
            self.observation_space = gym.spaces.Dict(new_spaces)
        else:
            # If not a dict space, create a dict with both original obs and "dino_embedding"
            self.observation_space = gym.spaces.Dict(
                {
                    "obs": env.observation_space,
                    "dino_embedding": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.output_dim,),
                        dtype=np.float32,
                    ),
                }
            )

    def reset(self, **kwargs):
        """Reset the environment and process observations."""
        obs, info = super().reset(**kwargs)
        obs = self.observation(obs)
        return obs, info

    def step(self, action):
        """Step the environment and process observations."""
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self.observation(obs)
        return obs, reward, terminated, truncated, info

    def _compute_dino_embeddings_batch(self, images: np.ndarray) -> np.ndarray:
        """Compute DINO embeddings for a batch of images."""
        if images.ndim == 3:
            # add env dimension if needed
            images = images[None, ...]
        num_envs = images.shape[0]

        # Process all images to ensure [H, W, C] format
        processed_images = []
        for i in range(num_envs):
            img = images[i] if images.ndim >= 3 else images

            # Ensure [H, W, C] format
            if img.ndim == 3:
                if img.shape[0] == 3:  # [C, H, W] -> [H, W, C]
                    img = np.transpose(img, (1, 2, 0))
            elif img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)

            # Ensure uint8
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)

            processed_images.append(img)

        # Stack into frames array: [num_envs, H, W, C]
        frames_array = np.stack(processed_images, axis=0)

        # Use helper function to compute embeddings
        embeddings = compute_video_embeddings(
            frames_array=frames_array,
            dinov2_model=self.dinov2_model,
            dinov2_processor=self.dinov2_processor,
            batch_size=num_envs,
            use_autocast=True,
            use_tqdm=False,
        )

        # Convert to numpy: [num_envs, self.output_dim]
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)

        return embeddings_np

    def observation(self, obs):
        """Convert observation images to DINO embeddings."""
        if isinstance(obs, dict):
            new_obs = dict(obs)  # Keep all original keys
            # Add "dino_embedding" for each image key
            img_batch = []
            for key in self.image_keys:
                if key in obs:
                    img_batch.append(obs[key])
                else:
                    raise ValueError(f"Image key {key} not found in observation")
            # Compute DINO embedding for this image and add as "dino_embedding"
            # For non-vectorized env, squeeze the batch dimension
            if img_batch[0].ndim == 3:
                img_batch = [img[None, ...] for img in img_batch]
            dino_embeddings = self._compute_dino_embeddings_batch(np.concatenate(img_batch, axis=0))
            # if there are multiple images, concatenate the embeddings
            if len(img_batch) > 1:
                new_obs["dino_embedding"] = np.concatenate(
                    [dino_embeddings[i] for i in range(len(dino_embeddings))], axis=0
                )  # Concatenate all embeddings for single env
            else:
                new_obs["dino_embedding"] = dino_embeddings.squeeze(0)  # Remove batch dim for single env
            return new_obs
        else:
            # If observation is not a dict, assume it's an image
            embedding = self._compute_dino_embedding(obs)
            return {"dino_embedding": embedding, **obs}

    def _compute_dino_embedding(self, image: np.ndarray) -> np.ndarray:
        """Compute DINO embedding for a single image."""
        # Ensure image is in the right format
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = image

        # Ensure image is in [H, W, C] format
        if image_np.ndim == 3:
            if image_np.shape[0] == 3:  # [C, H, W] -> [H, W, C]
                image_np = np.transpose(image_np, (1, 2, 0))
        elif image_np.ndim == 2:
            # Grayscale, add channel dimension
            image_np = np.expand_dims(image_np, axis=-1)
            if image_np.shape[-1] == 1:
                # Convert grayscale to RGB by repeating
                image_np = np.repeat(image_np, 3, axis=-1)

        # Ensure uint8 in [0, 255] range
        if image_np.dtype != np.uint8:
            if image_np.max() <= 1.0:
                image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image_np = image_np.clip(0, 255).astype(np.uint8)

        # Add time dimension: [H, W, C] -> [1, H, W, C]
        frames_array = np.expand_dims(image_np, axis=0)

        # Use helper function to compute embeddings
        embeddings = compute_video_embeddings(
            frames_array=frames_array,
            dinov2_model=self.dinov2_model,
            dinov2_processor=self.dinov2_processor,
            batch_size=1,
            use_autocast=True,
            use_tqdm=False,
        )

        # Convert to numpy and remove time dimension: [1, 768] -> [768]
        embedding_np = embeddings.cpu().numpy().astype(np.float32)[0]

        return embedding_np

    def __getattr__(self, name):
        """Forward unknown attributes/methods to wrapped environment."""
        return getattr(self.env, name)


class VectorDinoEmbeddingWrapper(VectorWrapperBase):
    """
    Vectorized version of DinoEmbeddingWrapper for gymnasium.vector.VectorEnv.

    Args:
        env: The vectorized environment to wrap
        dinov2_model: DINOv2 model for computing embeddings
        dinov2_processor: DINOv2 processor for preprocessing images
        image_keys: List of observation keys that contain images (default: auto-detect)
    """

    def __init__(
        self, env, dinov2_model, dinov2_processor, device: Optional[str] = None, image_keys: Optional[List[str]] = None
    ):
        super().__init__(env)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dinov2_model = dinov2_model
        self.dinov2_processor = dinov2_processor

        # get output  by projecting a fake image through the model
        fake_image = torch.ones((1, 3, 84, 84)).to(device).float() * 2
        fake_image = self.dinov2_processor(images=fake_image, return_tensors="pt")
        fake_image = {k: v.to(device, non_blocking=True) for k, v in fake_image.items()}
        fake_image = self.dinov2_model(**fake_image)
        self.output_dim = fake_image.pooler_output.shape[1]

        self.image_keys = image_keys if image_keys is not None else ["image"]

        # Update observation space - add "dino_embedding" space while keeping original keys
        # Update single_observation_space
        if isinstance(env.single_observation_space, gym.spaces.Dict):
            new_single_spaces = dict(env.single_observation_space.spaces)
            # Add "dino_embedding" key (DINOv2-base outputs self.output_dim-dim embeddings)
            new_single_spaces["dino_embedding"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self.image_keys) * self.output_dim,),
                dtype=np.float32,
            )
            self.single_observation_space = gym.spaces.Dict(new_single_spaces)
        else:
            # If not a dict space, create a dict with both original obs and "dino_embedding"
            self.single_observation_space = gym.spaces.Dict(
                {
                    "obs": env.single_observation_space,
                    "dino_embedding": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.output_dim,),
                        dtype=np.float32,
                    ),
                }
            )

        # Update observation_space (batched) if it exists
        if hasattr(env, "observation_space"):
            if isinstance(env.observation_space, gym.spaces.Dict):
                new_spaces = dict(env.observation_space.spaces)
                # Add "dino_embedding" key
                new_spaces["dino_embedding"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self.image_keys) * self.output_dim,),
                    dtype=np.float32,
                )
                self.observation_space = gym.spaces.Dict(new_spaces)
            else:
                # If not a dict space, create a dict with both original obs and "dino_embedding"
                self.observation_space = gym.spaces.Dict(
                    {
                        "obs": env.observation_space,
                        "dino_embedding": gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(self.output_dim,),
                            dtype=np.float32,
                        ),
                    }
                )

    def reset(self, **kwargs):
        """Reset the environment and process observations."""
        obs, info = super().reset(**kwargs)
        obs = self.observation(obs)
        return obs, info

    def step(self, actions):
        """Step the environment and process observations."""
        obs, reward, terminated, truncated, info = super().step(actions)
        obs = self.observation(obs)
        return obs, reward, terminated, truncated, info

    def observation(self, obs):
        """Convert observation images to DINO embeddings for all envs."""
        if isinstance(obs, dict):
            new_obs = dict(obs)  # Keep all original keys
            # Add "dino_embedding" for each image key
            img_batch = []
            for key in self.image_keys:
                if key in obs:
                    assert obs[key].ndim == 4, (
                        "Observation must be a 4D array as it should be wrapped around a vectorized environment"
                    )
                    num_envs = obs[key].shape[0]
                    for env in range(num_envs):
                        img_batch.append(obs[key][env])
                else:
                    raise ValueError(f"Image key {key} not found in observation")
            # Process all images in batch and add as "dino_embedding"
            imgs_per_env = len(img_batch) // num_envs
            # Shape: [n_envs, embedding_dim] - keep batch dimension for vectorized env
            # Use np.stack to create (N, H, W, C) from list of (H, W, C) arrays
            dino_embeddings = self._compute_dino_embeddings_batch(np.stack(img_batch, axis=0))
            # if there are multiple images, concatenate the embeddings for each env
            # Slicing dino_embeddings[i::num_envs] gets all image keys for env i
            per_env_embeddings = [dino_embeddings[i::num_envs] for i in range(num_envs)]
            # Inner concatenate: combine embeddings from different image keys for each env -> (num_keys * embedding_dim,)
            # Outer stack: preserve batch dimension -> (num_envs, num_keys * embedding_dim)
            new_obs["dino_embedding"] = np.stack(
                [np.concatenate(embeddings, axis=0) for embeddings in per_env_embeddings], axis=0
            )
            return new_obs
        else:
            # If observation is not a dict, assume it's an image
            embeddings = self._compute_dino_embeddings_batch(obs)
            return {"dino_embedding": embeddings, **obs}

    def _compute_dino_embeddings_batch(self, images: np.ndarray) -> np.ndarray:
        """Compute DINO embeddings for a batch of images."""
        if images.ndim == 3:
            # add env dimension if needed
            images = images[None, ...]
        num_envs = images.shape[0]

        # Process all images to ensure [H, W, C] format
        processed_images = []
        for i in range(num_envs):
            img = images[i] if images.ndim >= 3 else images

            # Ensure [H, W, C] format
            if img.ndim == 3:
                if img.shape[0] == 3:  # [C, H, W] -> [H, W, C]
                    img = np.transpose(img, (1, 2, 0))
            elif img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)

            # Ensure uint8
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)

            processed_images.append(img)

        # Stack into frames array: [num_envs, H, W, C]
        frames_array = np.stack(processed_images, axis=0)

        # Use helper function to compute embeddings
        embeddings = compute_video_embeddings(
            frames_array=frames_array,
            dinov2_model=self.dinov2_model,
            dinov2_processor=self.dinov2_processor,
            batch_size=num_envs,
            use_autocast=True,
            use_tqdm=False,
        )

        # Convert to numpy: [num_envs, self.output_dim]
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)

        return embeddings_np

    def __getattr__(self, name):
        """Forward unknown attributes/methods to wrapped environment."""
        return getattr(self.env, name)
