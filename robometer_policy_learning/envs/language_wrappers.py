import gymnasium as gym
import gymnasium.vector as gym_vector
from robometer_policy_learning.envs.gym_compat import VectorWrapperBase
import numpy as np
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = Any

from .metaworld_utils import TASK_TO_LANG
from robometer.utils.embedding_utils import compute_text_embeddings


class LanguageInstructionWrapper(gym.ObservationWrapper):
    """
    Observation wrapper that adds a fixed language encoding to observations
    and exposes it in the observation_space for single (non-vector) envs.
    """

    def __init__(self, env: gym.Env, task_name: str, sentence_model: SentenceTransformer):
        super().__init__(env)
        self.language_instruction = TASK_TO_LANG[task_name]
        # Convert to numpy float32 for compatibility with Box dtype
        enc = compute_text_embeddings(self.language_instruction, sentence_model)
        self.language_encoding = enc.cpu().numpy().astype(np.float32)

        # Update observation space to include 'language'
        orig_space = self.env.observation_space
        if isinstance(orig_space, gym.spaces.Dict):
            spaces_dict = dict(orig_space.spaces)
        else:
            spaces_dict = {"obs": orig_space}
        spaces_dict["language"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.language_encoding.shape, dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(spaces_dict)

    def observation(self, obs):
        if isinstance(obs, dict):
            out = dict(obs)
            out["language"] = self.language_encoding
            return out
        return obs

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # Add language_instruction to info
        if isinstance(info, dict):
            info = dict(info)
            info["language_instruction"] = self.language_instruction
        else:
            info = {"language_instruction": self.language_instruction}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Apply observation wrapper processing
        obs = self.observation(obs)
        # Add language_instruction to info
        if isinstance(info, dict):
            info = dict(info)
            info["language_instruction"] = self.language_instruction
        else:
            info = {"language_instruction": self.language_instruction}
        return obs, reward, terminated, truncated, info

    def get_language_instruction(self) -> str:
        return self.language_instruction

    def __getattr__(self, name):
        """Forward unknown attributes/methods to wrapped environment."""
        return getattr(self.env, name)


class LanguageInstructionVectorWrapper(VectorWrapperBase):
    """
    Vectorized version of LanguageInstructionWrapper for gymnasium.vector.VectorEnv.
    Attaches a per-env language instruction into the info dict and in the observation dict under 'language'.
    Also updates the observation space to include the 'language' key (shape=384).
    """

    def __init__(self, env: gym_vector.VectorEnv, task_name: str = None, sentence_model: SentenceTransformer = None):
        super().__init__(env)
        print("Using language instruction: ", TASK_TO_LANG[task_name])
        self.language_instruction = TASK_TO_LANG[task_name]
        enc = compute_text_embeddings(self.language_instruction, sentence_model)
        # Convert to numpy array for consistency with image embeddings
        self.language_encoding = enc.cpu().numpy().astype(np.float32)

        # Update observation spaces to add 'language' for both single and batched views
        # Batched observation_space
        orig_space = self.env.observation_space
        if isinstance(orig_space, gym.spaces.Dict):
            obs_space_dict = dict(orig_space.spaces)
        else:
            obs_space_dict = {"obs": orig_space}
        obs_space_dict["language"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(obs_space_dict)

        # Single env observation space (critical for callers using single_observation_space)
        orig_single = getattr(self.env, "single_observation_space", None)
        if orig_single is None:
            orig_single = orig_space
        if isinstance(orig_single, gym.spaces.Dict):
            single_obs_space_dict = dict(orig_single.spaces)
        else:
            single_obs_space_dict = {"obs": orig_single}
        single_obs_space_dict["language"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32)
        self.single_observation_space = gym.spaces.Dict(single_obs_space_dict)

    def _add_language_to_obs(self, obs, n):
        # Adds language instruction under key 'language' for each env
        # language_encoding is already a numpy array from __init__
        if isinstance(obs, dict):
            obs = dict(obs)
            # Create numpy array with shape (n, embedding_dim) instead of list
            obs["language"] = np.tile(self.language_encoding, (n, 1))  # (n, 384)
            return obs
        elif isinstance(obs, (list, tuple)):
            # for list/tuple of dicts
            for i in range(n):
                if isinstance(obs[i], dict):
                    obs[i] = dict(obs[i])
                    obs[i]["language"] = self.language_encoding  # (384,)
            return obs
        else:
            return obs  # in case of unknown obs format

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            n = self.env.num_envs
        except Exception:
            n = len(obs) if hasattr(obs, "__len__") else 1
        instructions = [self.language_instruction] * n
        # Add language to observation
        obs = self._add_language_to_obs(obs, n)
        # Add language to info
        if isinstance(info, dict):
            info = dict(info)
            info["language_instruction"] = instructions
        else:
            info = [{"language_instruction": self.language_instruction} for _ in range(n)]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            n = self.env.num_envs
        except Exception:
            n = len(obs) if hasattr(obs, "__len__") else 1
        instructions = [self.language_instruction] * n
        # Add language to observation
        obs = self._add_language_to_obs(obs, n)
        # Add language to info
        if isinstance(info, dict):
            info = dict(info)
            info["language_instruction"] = instructions
        else:
            info = [{"language_instruction": self.language_instruction} for _ in range(n)]
        return obs, reward, terminated, truncated, info

    def get_language_instruction(self) -> str:
        return self.language_instruction

    def __getattr__(self, name):
        return getattr(self.env, name)
