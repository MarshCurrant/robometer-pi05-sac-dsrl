"""Small Gymnasium API compatibility shims used by vector wrappers."""

from __future__ import annotations

import gymnasium.vector as gym_vector


VectorWrapperBase = getattr(
    gym_vector, "VectorWrapper", getattr(gym_vector, "VectorEnvWrapper", None)
)
if VectorWrapperBase is None:
    raise ImportError("Gymnasium provides neither VectorWrapper nor VectorEnvWrapper")
