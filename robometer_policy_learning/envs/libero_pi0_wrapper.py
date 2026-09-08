"""
LIBERO Environment Wrapper for DSRL + Pi0

Wraps LIBERO environments to work with Pi0 and DSRL,
including RFM reward model integration.
"""

import hashlib

import gymnasium as gym
import gymnasium.vector as gym_vector
import numpy as np
from typing import Dict, Any, Optional, List
from robometer_policy_learning.envs.gym_compat import VectorWrapperBase

from libero.libero.envs import OffScreenRenderEnv
from robometer_policy_learning.utils.pi0_integration import preprocess_obs_for_pi0
from robometer.utils.embedding_utils import compute_text_embeddings


class LiberoPI0Wrapper(gym.Wrapper):
    """
    Wrapper for LIBERO environments to work with Pi0.

    - Converts LIBERO observations to Pi0 format
    - Handles image preprocessing (resizing, flipping)
    - Manages state vector extraction
    - Optional: Integrates RFM for reward relabeling
    """

    def __init__(
        self,
        env,
        ignore_environment_termination: bool = False,
        ignore_environment_reward: bool = False,
        reset_settle_steps: int = 0,
        procedural_seed: Optional[int] = None,
    ):
        """
        Initialize LIBERO wrapper.

        Args:
            env: Base LIBERO environment
        """
        super().__init__(env)
        self.ignore_environment_termination = bool(ignore_environment_termination)
        self.ignore_environment_reward = bool(ignore_environment_reward)
        self.reset_settle_steps = int(reset_settle_steps)
        if self.reset_settle_steps < 0:
            raise ValueError("reset_settle_steps must be non-negative")
        # Standard LIBERO implements env.seed() with global np.random.seed().
        # Keep a wrapper-local reset stream so train/eval environments and
        # actor initialization cannot silently perturb one another.
        self._procedural_rng_state = (
            np.random.RandomState(int(procedural_seed)).get_state()
            if procedural_seed is not None
            else None
        )
        self.sim_success_once = False

        # Define observation space (Pi0 format)
        self.observation_space = gym.spaces.Dict(
            {
                "observation/state": gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
                "observation/image": gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
                "observation/wrist_image": gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            }
        )

        # Action space remains the same
        if not hasattr(self.env, "action_space"):
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Task metadata (set during reset)
        self.task_id = None
        self.task_suite = None
        self.language_instruction = None

        print(f"✓ LIBERO Pi0 Wrapper initialized")

        self.dsrl_key_mapping = {
            "image": ["observation/image"],
            "state": "observation/state",
            "language": "language",
        }

    def reset(self, **kwargs):
        """Reset environment and return Pi0-format observation"""
        caller_rng_state = None
        if self._procedural_rng_state is not None:
            caller_rng_state = np.random.get_state()
            np.random.set_state(self._procedural_rng_state)
        try:
            raw_obs, info = self.env.reset()
            if self.reset_settle_steps:
                settle_action = np.zeros(7, dtype=np.float32)
                settle_action[-1] = -1.0
                for _ in range(self.reset_settle_steps):
                    raw_obs, _reward, terminated, truncated, _info = self.env.step(
                        settle_action
                    )
                    if terminated or truncated:
                        raise RuntimeError(
                            "LIBERO terminated during reset settling; cannot reproduce "
                            "the RLinf initial-observation contract"
                        )
                # RLinf performs settling below its episode accounting layer. The
                # policy-learning compatibility wrapper owns the counter, so reset
                # it after the equivalent raw environment steps.
                if hasattr(self.env, "current_step"):
                    self.env.current_step = 0
        finally:
            if caller_rng_state is not None:
                self._procedural_rng_state = np.random.get_state()
                np.random.set_state(caller_rng_state)
        self.sim_success_once = False

        info = dict(info or {})
        if "agentview_image" in raw_obs:
            image = np.ascontiguousarray(raw_obs["agentview_image"])
            digest = hashlib.blake2b(image.tobytes(), digest_size=8).digest()
            info["initial_observation_fingerprint"] = (
                int.from_bytes(digest, byteorder="little") & ((1 << 24) - 1)
            )

        # Get language instruction
        self.language_instruction = self.env.language_instruction

        # Convert to Pi0 format
        obs = preprocess_obs_for_pi0(raw_obs)
        obs["prompt"] = self.language_instruction

        return obs, info

    def step(self, action: np.ndarray):
        """
        Execute action and return Pi0-format observation.

        Args:
            action: np.ndarray of shape (7,) - robot action

        Returns:
            obs: Pi0-format observation
            reward: Reward (sparse or RFM-based)
            done: Terminal flag
            truncated: Truncation flag
            info: Info dictionary
        """
        # LIBERO's procedural RNG is process-global. Keep each wrapper's stream
        # active for steps as well as resets, matching RLinf's subprocess envs.
        caller_rng_state = None
        if self._procedural_rng_state is not None:
            caller_rng_state = np.random.get_state()
            np.random.set_state(self._procedural_rng_state)
        try:
            raw_obs, reward, done, truncated, info = self.env.step(action)
        finally:
            if caller_rng_state is not None:
                self._procedural_rng_state = np.random.get_state()
                np.random.set_state(caller_rng_state)

        # Convert observation to Pi0 format
        obs = preprocess_obs_for_pi0(raw_obs)
        obs["prompt"] = self.language_instruction

        info["language_instruction"] = self.language_instruction
        info["sparse_reward"] = reward

        # Training can hide the simulator label so RoboMeter is the only
        # termination source. Evaluation keeps the native LIBERO behavior.
        simulator_done = bool(done)
        self.sim_success_once = self.sim_success_once or simulator_done
        info["sim_success"] = simulator_done
        info["sim_success_once"] = self.sim_success_once
        if self.ignore_environment_termination:
            info.pop("success", None)
            info.pop("is_success", None)
        if not self.ignore_environment_termination and "success" not in info:
            info["success"] = simulator_done
        if simulator_done:
            assert reward == 1.0, "Reward should be 1.0 when task succeeds"

        if self.ignore_environment_reward:
            reward = 0.0
        else:
            reward -= 1  # reward is -1, 0
        if self.ignore_environment_termination:
            done = False

        return obs, np.float64(reward), done, truncated, info

    def close(self):
        """Close environment"""
        self.env.close()


class VectorLiberoPromptWrapper(VectorWrapperBase):
    """
    Adds 'prompt' (language instruction) to each observation in a vectorized LIBERO env,
    using env.language_instruction from one of the underlying environments.
    'prompt' is included in each observation dict returned by the vector env.
    """

    def __init__(self, env: gym_vector.VectorEnv, sentence_model: Any = None):
        # Always set on the wrapper itself so __getattr__ never forwards it to the base env
        self.sentence_model: Optional[Any] = sentence_model
        self.language_encoding: Optional[np.ndarray] = None

        super().__init__(env)
        # Try to extract language_instruction from the base env
        # This assumes all vectorized envs have the same language_instruction
        self.language_instruction = None
        # Handle SyncVectorEnv/AsyncVectorEnv where envs is a list of callables or environments
        self.language_instruction = env.envs[0].language_instruction

        # Pre-compute language embedding if a model is provided
        if self.sentence_model is not None and self.language_instruction is not None:
            enc = compute_text_embeddings(self.language_instruction, self.sentence_model)
            self.language_encoding = enc.cpu().numpy().astype(np.float32)
        else:
            self.language_encoding = np.zeros((384,), dtype=np.float32)

        # Update observation space to include 'prompt'
        if hasattr(self, "single_observation_space"):
            orig_space = self.single_observation_space
        else:
            orig_space = self.observation_space

        # Defensive: only update if it's a Dict space and doesn't already have 'prompt'
        if isinstance(orig_space, gym.spaces.Dict) and "prompt" not in orig_space.spaces:
            # Guess dtype and shape: a language prompt is a string, so we use gym.spaces.Text if available, else ignore in obs space
            try:
                # gymnasium >=0.29 has spaces.Text
                prompt_space = gym.spaces.Text(min_length=1, max_length=512)
                spaces_dict = dict(orig_space.spaces)
                spaces_dict["prompt"] = prompt_space
                # Only add 'language' to the observation space if we actually have an embedding
                spaces_dict["language"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self.language_encoding.shape if self.language_encoding is not None else (384,),
                    dtype=np.float32,
                )
                new_space = gym.spaces.Dict(spaces_dict)
                if hasattr(self, "single_observation_space"):
                    self.single_observation_space = new_space
                else:
                    self.observation_space = new_space
            except Exception:
                # If gym.spaces.Text is not available, just ignore in space (will still insert into dict at runtime)
                pass

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Access language_instruction from the first env, which may be wrapped
        try:
            self.language_instruction = getattr(self.env.envs[0], "language_instruction", None)
        except (AttributeError, IndexError):
            self.language_instruction = None
        obs = self._add_prompt(obs)
        try:
            n = self.env.num_envs
        except Exception:
            n = len(obs) if hasattr(obs, "__len__") else 1
        # (Re-)compute language encoding if we have a model; keep the last encoding otherwise
        if self.sentence_model is not None:
            enc = compute_text_embeddings(self.language_instruction, self.sentence_model)
            self.language_encoding = enc.cpu().numpy().astype(np.float32)

        # Add language encoding to observation only if it exists
        if self.language_encoding is not None:
            obs = self._add_language_to_obs(obs, n)
        # Attach to info too, if desired
        if isinstance(info, dict):
            info = {k: dict(v, prompt=self.language_instruction) if isinstance(v, dict) else v for k, v in info.items()}
        return obs, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        obs = self._add_prompt(obs)
        try:
            n = self.env.num_envs
        except Exception:
            n = len(obs) if hasattr(obs, "__len__") else 1
        # Add language to observation only if we actually have an encoding
        if self.language_encoding is not None:
            obs = self._add_language_to_obs(obs, n)
        # Attach to info for each env
        if isinstance(info, dict):
            info = {k: dict(v, prompt=self.language_instruction) if isinstance(v, dict) else v for k, v in info.items()}
        elif isinstance(info, list):
            info = [dict(i, prompt=self.language_instruction) if isinstance(i, dict) else i for i in info]
        return obs, reward, terminated, truncated, info

    def _add_prompt(self, obs: Dict[str, Any]):
        obs["prompt"] = [self.language_instruction] * self.env.num_envs
        return obs

    def __getattr__(self, name):
        """Forward unknown attributes/methods to wrapped environment."""
        return getattr(self.env, name)

    def get_language_instruction(self) -> str:
        return self.language_instruction

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
