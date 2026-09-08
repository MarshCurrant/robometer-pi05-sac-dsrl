import gymnasium as gym
import gymnasium.vector as gym_vector
from robometer_policy_learning.envs.gym_compat import VectorWrapperBase
import numpy as np
from typing import Tuple, Dict, List, Optional


# Action Chunking Wrapper
class ActionChunkingWrapper(gym.ActionWrapper):
    """
    Wraps an environment to handle chunked actions for open-loop execution.

    This wrapper receives a chunked action of shape (num_envs, chunk_size, action_dim),
    stores them in a buffer, and executes them sequentially in an open-loop manner.

    Args:
        env: The environment to wrap
        chunk_size: Size of each action chunk
        n_action_steps: Number of action steps to execute from each chunk
    """

    def __init__(self, env: gym.Env, chunk_size: int, n_action_steps: int):
        super().__init__(env)
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.action_buffer = None  # Will be initialized on first action
        self.last_action = None
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate initialization parameters."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.n_action_steps <= 0:
            raise ValueError(f"n_action_steps must be positive, got {self.n_action_steps}")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Process chunked actions and return the next action to execute.

        Args:
            action: Chunked action array of shape (num_envs, chunk_size, action_dim)

        Returns:
            Single action of shape (num_envs, action_dim) to execute
        """
        # Normalize and validate action shape
        if action is not None:
            # Allow an extra leading singleton batch dimension (1, chunk_size, action_dim)
            if isinstance(action, np.ndarray) and action.ndim == 3 and action.shape[0] == 1:
                action = action[0]
            if len(action.shape) != 2:
                raise ValueError(
                    f"Action must be chunked into a sequence of shape (chunk_size, action_dim), got shape {action.shape}"
                )

        # Initialize buffer if empty or if we've used up all actions
        if self.action_buffer is None or self._get_remaining_actions() == 0:
            self._initialize_buffer(action)

        # Get the next action to execute
        next_action = self.action_buffer[0, :]

        # Remove the executed action from buffer
        self.action_buffer = self.action_buffer[1:, :]

        self.last_action = next_action

        return next_action

    def _get_last_action(self) -> np.ndarray:
        return self.last_action

    def _initialize_buffer(self, action: np.ndarray) -> None:
        """Initialize the action buffer with new actions."""
        # Take only the first n_action_steps from the chunk
        self.action_buffer = action[: self.n_action_steps, :]

    def _get_remaining_actions(self) -> int:
        """Get the number of remaining actions in the buffer."""
        if self.action_buffer is None:
            return 0
        return self.action_buffer.shape[0]

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment and clear the action buffer."""
        self.action_buffer = None
        return super().reset(**kwargs)

    @property
    def is_chunk_empty(self) -> bool:
        """Check if the action buffer is empty."""
        return self._get_remaining_actions() == 0


# Vectorized Action Chunking Wrapper
class VectorActionChunkingWrapper(VectorWrapperBase):
    """
    Vectorized wrapper to handle chunked actions for open-loop execution in gymnasium.vector.VectorEnv.

    Expects chunked actions of shape (num_envs, chunk_size, action_dim) when a new
    chunk is provided, and then autonomously executes the next action for each env
    for the next n_action_steps. When no new chunk is provided (actions=None), it
    continues executing from the internal per-env buffers.

    Args:
        env: The vectorized environment to wrap
        chunk_size: Size of each action chunk provided by the policy
        n_action_steps: Number of action steps from each chunk to execute
    """

    def __init__(self, env: gym_vector.VectorEnv, chunk_size: int, n_action_steps: int):
        super().__init__(env)
        self.chunk_size = int(chunk_size)
        self.n_action_steps = int(n_action_steps)
        self._validate_params()

        # Per-env action buffers. Each entry is None or an ndarray of shape (remaining_steps, action_dim)
        self.action_buffers: List[Optional[np.ndarray]] = [None for _ in range(self.env.num_envs)]
        self.last_action: Optional[np.ndarray] = None  # (num_envs, action_dim)

    def _validate_params(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.n_action_steps <= 0:
            raise ValueError(f"n_action_steps must be positive, got {self.n_action_steps}")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

    def _initialize_buffers(self, chunk_actions: np.ndarray) -> None:
        """Initialize per-env buffers from a new chunk of actions.

        chunk_actions: np.ndarray with shape (num_envs, chunk_size, action_dim)
        """
        if not isinstance(chunk_actions, np.ndarray) or chunk_actions.ndim != 3:
            raise ValueError(
                f"Chunk actions must be ndarray with shape (num_envs, chunk_size, action_dim); got {type(chunk_actions)} with shape {getattr(chunk_actions, 'shape', None)}"
            )
        num_envs = self.env.num_envs
        if chunk_actions.shape[0] != num_envs:
            raise ValueError(f"chunk_actions first dim must equal num_envs={num_envs}, got {chunk_actions.shape[0]}")
        if chunk_actions.shape[1] < self.n_action_steps:
            raise ValueError(
                f"chunk_actions second dim (chunk_size={chunk_actions.shape[1]}) must be >= n_action_steps={self.n_action_steps}"
            )

        # Use only the first n_action_steps from each env's chunk
        self.action_buffers = [chunk_actions[i, : self.n_action_steps, :].copy() for i in range(num_envs)]

    def _get_remaining_actions(self, env_index: int) -> int:
        buf = self.action_buffers[env_index]
        if buf is None:
            return 0
        return int(buf.shape[0])

    def _pop_next_actions(self) -> np.ndarray:
        """Pop the next action for all envs from their buffers. Returns (num_envs, action_dim)."""
        num_envs = self.env.num_envs
        # Infer action_dim from spaces
        action_dim = int(np.prod(self.single_action_space.shape))
        next_actions = np.zeros((num_envs, action_dim), dtype=np.float32)

        for i in range(num_envs):
            buf = self.action_buffers[i]
            if buf is None or buf.shape[0] == 0:
                raise RuntimeError(
                    "Action buffer empty for at least one env; a new chunk must be provided before stepping."
                )
            # Take first row
            act = buf[0]
            next_actions[i] = act
            # Remove it
            remaining = buf[1:]
            self.action_buffers[i] = remaining if remaining.shape[0] > 0 else None

        return next_actions

    def reset(self, **kwargs):
        # Clear buffers on reset
        self.action_buffers = [None for _ in range(self.env.num_envs)]
        self.last_action = None
        return super().reset(**kwargs)

    def step(self, actions):
        """
        If actions is None: execute the next action from each env's buffer.
        If actions has shape (num_envs, chunk_size, action_dim): initialize new buffers and execute the first step.
        If actions has shape (num_envs, action_dim): bypass chunking and forward as-is.
        """
        if actions is None:
            # Continue with existing buffers
            next_actions = self._pop_next_actions()
        elif isinstance(actions, np.ndarray) and actions.ndim == 3:
            # New chunk provided
            self._initialize_buffers(actions)
            next_actions = self._pop_next_actions()
        elif isinstance(actions, np.ndarray) and actions.ndim == 2:
            # Direct per-step actions; bypass chunking
            next_actions = actions
            # Invalidate buffers to avoid mixing paradigms
            self.action_buffers = [None for _ in range(self.env.num_envs)]
        else:
            raise ValueError(
                f"Unsupported actions format. Expected None, (num_envs, chunk_size, action_dim) or (num_envs, action_dim); got {type(actions)} with shape {getattr(actions, 'shape', None)}"
            )

        self.last_action = next_actions.copy()
        obs, reward, terminated, truncated, info = self.env.step(next_actions)

        # Clear buffers for envs that terminated this step
        try:
            term_arr = np.asarray(terminated).astype(bool)
            trunc_arr = np.asarray(truncated).astype(bool)
            done_arr = term_arr | trunc_arr
            for i in range(self.env.num_envs):
                if i < done_arr.shape[0] and done_arr[i]:
                    self.action_buffers[i] = None
        except Exception:
            # Be resilient if shapes are unexpected
            pass

        return obs, reward, terminated, truncated, info

    def _get_last_action(self) -> Optional[np.ndarray]:
        return self.last_action

    @property
    def is_chunk_empty(self) -> bool:
        # True when all buffers are empty/None
        for buf in self.action_buffers:
            if buf is not None and buf.shape[0] > 0:
                return False
        return True

    def __getattr__(self, name):
        # Delegate unknown attributes to the underlying env (e.g., language_instruction)
        return getattr(self.env, name)
