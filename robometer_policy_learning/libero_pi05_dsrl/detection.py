"""RoboMeter success detection used by the LIBERO Pi0.5 DSRL path."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def combine_environment_and_estimated_reward(
    environment_reward: float,
    estimated_reward: float,
    *,
    add_estimated_reward: bool,
) -> float:
    """Apply the official RoboMeter replay reward composition."""
    if add_estimated_reward:
        return float(environment_reward) + float(estimated_reward)
    return float(estimated_reward)


def success_window_detected(
    probabilities: Sequence[float],
    *,
    threshold: float,
    duration: int,
    rule: str,
) -> bool:
    """Match the detector semantics validated in RLinf's LIBERO wrapper."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if len(probabilities) != duration:
        return False
    if rule == "all_consecutive":
        return all(float(value) > float(threshold) for value in probabilities)
    if rule == "majority_window":
        votes = sum(float(value) >= float(threshold) for value in probabilities)
        return votes > duration / 2
    raise ValueError(f"Unsupported success detection rule: {rule!r}")


def terminal_progress_contexts(
    frames: np.ndarray,
    *,
    context_frames: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Build RLinf's initial and full-trajectory terminal contexts."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError(f"Expected non-empty (T,H,W,C) frames, got {frames.shape}")
    if context_frames <= 0:
        raise ValueError("context_frames must be positive")
    last = int(frames.shape[0]) - 1
    first_indices = np.zeros(context_frames, dtype=np.int64)
    final_indices = np.linspace(0, last, context_frames, dtype=np.int64)
    return (
        np.ascontiguousarray(frames[first_indices], dtype=np.uint8),
        np.ascontiguousarray(frames[final_indices], dtype=np.uint8),
    )


def normalized_terminal_progress_delta(
    first_progress: float,
    final_progress: float,
    *,
    frame_count: int,
) -> float:
    """Return the per-frame endpoint delta used by the validated detector."""
    if frame_count <= 1:
        raise ValueError("frame_count must be greater than one")
    return (float(final_progress) - float(first_progress)) / (frame_count - 1)
