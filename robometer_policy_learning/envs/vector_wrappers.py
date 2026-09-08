"""
Vectorization wrappers for environments.
"""

import numpy as np


class SingleEnvVectorWrapper:
    """
    Wrapper that makes a single environment compatible with vectorized environment APIs.

    This wrapper is useful when you have a single environment but need it to behave
    like a vectorized environment (e.g., for compatibility with code expecting vectorized envs).
    """

    def __init__(self, env):
        self.env = env
        self.num_envs = 1
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, [info]

    def step(self, actions):
        # Handle actions from vectorized environment
        # The underlying env is vectorized with num_envs=1, so it expects (1, action_dim)
        if isinstance(actions, np.ndarray):
            if actions.ndim == 2 and actions.shape[0] == 1:
                # Already (1, action_dim) - pass through
                act = actions
            elif actions.ndim == 1:
                # Shape (action_dim,) -> reshape to (1, action_dim) for vectorized env
                act = actions[np.newaxis, :]
            else:
                # Unexpected shape, try to reshape
                if actions.size > 0:
                    act = actions.reshape(1, -1)
                else:
                    act = actions
        elif isinstance(actions, (list, tuple)):
            # Convert list/tuple to numpy array and ensure (1, action_dim) shape
            act = np.array(actions, dtype=np.float32)
            if act.ndim == 1:
                act = act[np.newaxis, :]
        else:
            # Scalar or other type - convert to array
            act = np.array([actions], dtype=np.float32)
            if act.ndim == 1:
                act = act[np.newaxis, :]

        # Ensure action is a numpy array with correct shape (1, action_dim)
        if not isinstance(act, np.ndarray):
            act = np.array(act, dtype=np.float32)
        if act.ndim == 1:
            act = act[np.newaxis, :]

        # Pass to vectorized environment which expects (1, action_dim)
        obs, reward, done, truncated, info = self.env.step(act)
        return obs, [float(reward)], [bool(done)], [bool(truncated)], [info]

    def render(self, mode="rgb_array"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)
