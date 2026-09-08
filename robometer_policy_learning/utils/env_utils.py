from __future__ import annotations

import gymnasium as gym
import numpy as np
from transformers import AutoModel, AutoImageProcessor
from typing import Any, Optional, Tuple, List
from robometer_policy_learning.envs.action_wrappers import ActionChunkingWrapper, VectorActionChunkingWrapper
from robometer_policy_learning.envs.obs_wrappers import FlatToDictObsWrapper, ImageDictObsWrapper
from robometer_policy_learning.envs.language_wrappers import LanguageInstructionWrapper, LanguageInstructionVectorWrapper
from robometer_policy_learning.envs.dino_wrapper import DinoEmbeddingWrapper, VectorDinoEmbeddingWrapper


def create_dummy_vectorized_env(
    env_name: str = "CartPole-v1",
    num_envs: int = 4,
    render_mode: str = "rgb_array",
    use_full_state: bool = False,
    kwargs: dict = {},
):
    """Create a simple Gymnasium vectorized environment for testing."""

    def make_env():
        env = gym.make(env_name, render_mode=render_mode, **kwargs)
        if use_full_state:
            env = FlatToDictObsWrapper(env)
        else:
            env = ImageDictObsWrapper(env)
        # Don't use RecordEpisodeStatistics - we track episodes in RolloutWorker
        return env

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    return envs


def make_metaworld_vectorized_env(
    task_suite: str = "Meta-World",
    task_name: str = "faucet-open-v3",
    num_envs: int = 4,
    render_mode: str = "rgb_array",
    kwargs: dict = {},
    chunk_size: int = 1,
    n_action_steps: int = 1,
    sentence_model: Any = None,
    use_full_state: bool = False,
):
    """Create a Meta-World vectorized environment with optional wrappers."""

    import metaworld  # noqa: F401 - imported lazily for environment registration

    def make_env():
        env = gym.make(task_suite, env_name=task_name, render_mode=render_mode, **kwargs)
        if use_full_state:
            env = FlatToDictObsWrapper(env)
        else:
            env = ImageDictObsWrapper(env)
        # Don't use RecordEpisodeStatistics - we track episodes in RolloutWorker
        return env

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    if sentence_model is not None:
        envs = LanguageInstructionVectorWrapper(envs, task_name, sentence_model)
    if chunk_size is not None:
        envs = VectorActionChunkingWrapper(envs, chunk_size=chunk_size, n_action_steps=n_action_steps)
    return envs


def _make_env(
    env_name: str,
    vectorized: bool = False,
    num_envs: int = 4,
    max_episode_steps: int = 400,
    chunk_size: Optional[int] = None,
    use_full_state: bool = False,
    dinov2_model: Optional[AutoModel] = None,
    dinov2_processor: Optional[AutoImageProcessor] = None,
    device: Optional[str] = None,
    sentence_model: Optional[Any] = None,
    render_mode: str = "rgb_array",
    terminate_on_success: bool = True,
    dino_image_keys: Optional[List[str]] = None,
    seed: Optional[int] = None,
    reset_settle_steps: int = 0,
) -> gym.Env:
    """
    Internal helper to create a single environment (vectorized or not) with appropriate wrappers.

    Args:
        env_name: Name of the environment (e.g., "Meta-World/faucet-open-v3" or "Pendulum-v1")
        vectorized: Whether to create a vectorized environment (for training)
        num_envs: Number of parallel environments (only used if vectorized=True)
        max_episode_steps: Maximum number of steps per episode primarily used for LIBERO and Meta-World environments
        chunk_size: Action chunk size (None for no chunking)
        use_full_state: Whether to use full state observations (for Meta-World)
        dinov2_model: DINOv2 model for embedding wrapper
        dinov2_processor: DINOv2 processor for preprocessing images
        device: Device to load DINOv2 model on
        sentence_model: Sentence transformer model for language embeddings (for Meta-World)
        render_mode: Render mode for the environment
        terminate_on_success: Whether to terminate the environment when the goal is reached (only for Meta-World)
        dino_image_keys: List of dino image keys to use for DINO embedding wrapper
        seed: Seed for the environment
    Returns:
        The created environment
    """
    # Determine device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    if "Meta-World" in env_name:
        task_suite, task_name = env_name.rsplit("/", 1)

        if vectorized:
            # Create vectorized Meta-World environment
            env = make_metaworld_vectorized_env(
                task_suite=task_suite,
                task_name=task_name,
                num_envs=num_envs,
                render_mode=render_mode,
                kwargs={
                    "terminate_on_success": terminate_on_success,
                    "camera_name": "corner2",
                    "max_episode_steps": max_episode_steps,
                },
                chunk_size=chunk_size,
                n_action_steps=1,
                sentence_model=sentence_model,
                use_full_state=use_full_state,
            )
            # Wrap with DINO embedding wrapper if provided
            if dinov2_model is not None and "image" in env.observation_space.spaces:
                env = VectorDinoEmbeddingWrapper(
                    env, dinov2_model, dinov2_processor, device=device, image_keys=dino_image_keys
                )
        else:
            # Create single Meta-World environment
            env = gym.make(
                task_suite,
                env_name=task_name,
                terminate_on_success=terminate_on_success,
                render_mode=render_mode,
                camera_name="corner2",
                max_episode_steps=max_episode_steps,
            )
            if use_full_state:
                env = FlatToDictObsWrapper(env)
            else:
                env = ImageDictObsWrapper(env)
            # Don't use RecordEpisodeStatistics - we track episodes in RolloutWorker
            if sentence_model is not None:
                env = LanguageInstructionWrapper(env, task_name, sentence_model)
            # Wrap with DINO embedding wrapper if provided
            if dinov2_model is not None and "image" in env.observation_space.spaces:
                env = DinoEmbeddingWrapper(
                    env, dinov2_model, dinov2_processor, device=device, image_keys=dino_image_keys
                )
            if chunk_size is not None:
                env = ActionChunkingWrapper(env, chunk_size=chunk_size, n_action_steps=1)
    elif "libero" in env_name:
        task_suite, task_id = env_name.split("/")
        task_id = int(task_id)

        from robometer_policy_learning.envs.dsrl_env_wrappers import setup_libero_env
        import random
        if seed is None:
            seed = random.randint(1,100)
        env, _ = setup_libero_env(task_suite_name=task_suite,
                                task_id=task_id,
                                n_envs=num_envs,
                                dinov2_model=dinov2_model,
                                dinov2_processor=dinov2_processor,
                                sentence_model=sentence_model,
                                device=device,
                                max_episode_steps=max_episode_steps, 
                                seed=seed, 
                                ignore_environment_termination=not terminate_on_success,
                                ignore_environment_reward=not terminate_on_success,
                                reset_settle_steps=reset_settle_steps,
                                )
    else:
        # Regular gym environment
        if vectorized:
            # Create vectorized regular gym environment
            def make_single_env():
                single_env = gym.make(env_name, render_mode=render_mode)
                if use_full_state:
                    single_env = FlatToDictObsWrapper(single_env)
                else:
                    single_env = ImageDictObsWrapper(single_env)
                # Don't use RecordEpisodeStatistics - we track episodes in RolloutWorker
                return single_env

            env = gym.vector.SyncVectorEnv([make_single_env for _ in range(num_envs)])

            if dinov2_model is not None:
                single_space = getattr(env, "single_observation_space", env.observation_space)
                if isinstance(single_space, gym.spaces.Dict) and "image" in single_space.spaces:
                    env = VectorDinoEmbeddingWrapper(
                        env, dinov2_model, dinov2_processor, device=device, image_keys=dino_image_keys
                    )
            if chunk_size is not None:
                env = VectorActionChunkingWrapper(env, chunk_size=chunk_size, n_action_steps=1)
        else:
            # Create single regular gym environment
            env = gym.make(env_name, render_mode=render_mode)

            if use_full_state:
                env = FlatToDictObsWrapper(env)
            else:
                env = ImageDictObsWrapper(env)
            # Don't use RecordEpisodeStatistics - we track episodes in RolloutWorker

            # Wrap with DINO embedding wrapper if provided and image observations are available
            if dinov2_model is not None:
                obs_space = env.observation_space
                if isinstance(obs_space, gym.spaces.Dict) and "image" in obs_space.spaces:
                    env = DinoEmbeddingWrapper(
                        env, dinov2_model, dinov2_processor, device=device, image_keys=dino_image_keys
                    )

            if chunk_size is not None:
                env = ActionChunkingWrapper(env, chunk_size=chunk_size, n_action_steps=1)

    return env


def make_env(
    env_name: str,
    num_envs: int = 4,
    max_episode_steps: int = 400,
    chunk_size: Optional[int] = None,
    use_full_state: bool = False,
    dinov2_model: Optional[AutoModel] = None,
    dinov2_processor: Optional[AutoImageProcessor] = None,
    device: Optional[str] = None,
    sentence_model: Optional[Any] = None,
    render_mode: str = "rgb_array",
    terminate_on_success: bool = True,
    eval_terminate_on_success: Optional[bool] = None,
    dino_image_keys: Optional[List[str]] = None,
    seed: Optional[int] = None,
    eval_seed: Optional[int] = None,
    reset_settle_steps: int = 0,
) -> Tuple[gym.Env, gym.Env]:
    """
    Create training and evaluation environments with appropriate wrappers.

    Args:
        env_name: Name of the environment (e.g., "Meta-World/faucet-open-v3" or "Pendulum-v1")
        num_envs: Number of parallel environments for training
        chunk_size: Action chunk size (None for no chunking)
        use_full_state: Whether to use full state observations (for Meta-World)
        dinov2_model: DINOv2 model for embedding wrapper
        dinov2_processor: DINOv2 processor for preprocessing images
        device: Device to load DINOv2 model on (default: "cuda" if available, else "cpu")
        sentence_model: Sentence transformer model for language embeddings (for Meta-World)
        render_mode: Render mode for the environment
        terminate_on_success: Whether to terminate the environment when the goal is reached
    Returns:
        Tuple of (training_env, eval_env)
    """
    if eval_terminate_on_success is None:
        eval_terminate_on_success = terminate_on_success
    if eval_seed is None:
        eval_seed = seed

    # Create training environment (vectorized)
    train_env = _make_env(
        env_name=env_name,
        vectorized=True,
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        chunk_size=chunk_size,
        use_full_state=use_full_state,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        device=device,
        sentence_model=sentence_model,
        render_mode=render_mode,
        terminate_on_success=terminate_on_success,
        dino_image_keys=dino_image_keys,
        seed=seed,
        reset_settle_steps=reset_settle_steps,
    )

    # Create evaluation environment (non-vectorized)
    eval_env = _make_env(
        env_name=env_name,
        vectorized=True,
        num_envs=1,
        max_episode_steps=max_episode_steps,
        chunk_size=chunk_size,
        use_full_state=use_full_state,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        device=device,
        sentence_model=sentence_model,
        render_mode=render_mode,
        terminate_on_success=eval_terminate_on_success,
        dino_image_keys=dino_image_keys,
        seed=eval_seed,
        reset_settle_steps=reset_settle_steps,
    )

    return train_env, eval_env


class GymToGymnasiumWrapper(gym.Env):
    """
    A wrapper to convert a classic Gym environment to a Gymnasium-like interface.
    It adapts `reset()` and `step()` signatures, handles info dict changes, and supports compatibility.
    """

    def __init__(self, env, time_limit: int = None):
        super().__init__()  # make sure Env is initialized
        self.env = env
        # Action space remains the same
        if hasattr(self.env, "action_space"):
            self.action_space = self.env.action_space
        if hasattr(self.env, "observation_space"):
            self.observation_space = self.env.observation_space
        self.reward_range = getattr(env, "reward_range", None)
        self.metadata = getattr(env, "metadata", {})
        self.time_limit = time_limit
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        # Reset step counter
        self.current_step = 0
        # Gym reset sometimes does not support 'seed' or 'options'
        if seed is not None:
            try:
                obs = self.env.reset(seed=seed)
            except TypeError:
                self.env.seed(seed)
                obs = self.env.reset()
        else:
            obs = self.env.reset()
        info = {}
        if isinstance(obs, tuple) and len(obs) == 2:
            obs, info = obs
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        self.current_step += 1
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            # Gymnasium expects terminated, truncated
            if self.time_limit is not None and self.current_step >= self.time_limit:
                truncated = True
            else:
                truncated = info.get("TimeLimit.truncated", False)
            return obs, reward, terminated, truncated, info
        elif len(result) == 5:
            # Already modern API
            return result
        else:
            raise ValueError("Unexpected number of outputs from env.step")

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        # Forward other attributes/methods to original env
        return getattr(self.env, name)
