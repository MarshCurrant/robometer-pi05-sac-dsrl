from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np


def extract_rewards_from_output(outputs: Dict[str, Any]) -> np.ndarray:
    """
    Extract rewards from the output dictionary returned by process_batch_helper.

    Args:
        outputs: Dictionary with 'outputs_preference', 'outputs_progress', 'outputs_similarity'
                 Should contain 'outputs_progress' with 'progress_pred' key

    Returns:
        numpy array of rewards (one per sample)
    """
    if outputs.get("outputs_progress") is None:
        raise ValueError("No progress outputs found in batch outputs")

    outputs_progress = outputs["outputs_progress"]
    progress_pred = outputs_progress.get("progress_pred", [])

    # Extract rewards (last value of each progress prediction)
    # Each progress_list contains progress values for each frame in the subsequence
    rewards_list = []
    for progress_list in progress_pred:
        try:
            if isinstance(progress_list, list) and len(progress_list) > 0:
                # The last value is the reward for this subsequence
                reward = float(progress_list[-1])
                # Clamp to [0, 1] range
                reward = max(0.0, min(1.0, reward))
            else:
                # Default to 0.0 if no valid prediction
                reward = 0.0
        except (ValueError, TypeError, IndexError) as e:
            reward = 0.0
        rewards_list.append(reward)

    return np.array(rewards_list, dtype=np.float32)


def extract_success_probs_from_output(outputs: Dict[str, Any]) -> np.ndarray:
    """
    Extract success probabilities from the output dictionary returned by process_batch_helper.

    Args:
        outputs: Dictionary with 'outputs_success'
                 Should contain 'outputs_success' with 'success_probs' key

    Returns:
        numpy array of success probabilities (one per sample)
    """
    if outputs.get("outputs_success") is None:
        raise ValueError("No success probabilities outputs found in batch outputs")

    outputs_success = outputs["outputs_success"]
    success_preds = outputs_success.get("success_probs", [])
    success_probs = []
    for success_pred in success_preds:
        try:
            if isinstance(success_pred, list) and len(success_pred) > 0:
                # The last value is the success probability for this subsequence
                success_prob = float(success_pred[-1])
                success_probs.append(success_prob)
            else:
                success_probs.append(0.0)
        except (ValueError, TypeError, IndexError) as e:
            success_probs.append(0.0)

    return np.array(success_probs, dtype=np.float32)


def extract_preferences_from_output(outputs: Dict[str, Any]) -> np.ndarray:
    """
    Extract preferences from the output dictionary returned by process_batch_helper.

    Args:
        outputs: Dictionary with 'outputs_preference'
                 Should contain 'outputs_preference' with 'predictions' key

    Returns:
        numpy array of preferences (one per sample)
    """
    if outputs.get("outputs_preference") is None:
        raise ValueError("No preferences outputs found in batch outputs")

    outputs_preference = outputs["outputs_preference"]
    preference_pred = outputs_preference.get("predictions", [])
    preference_probs = outputs_preference.get(
        "prediction_probs", []
    )  ## TODO: we might need to use this to account for noise.

    return np.array(preference_pred, dtype=np.float32)


def extract_rewards_from_server_output(outputs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    outputs_progress = outputs.get("outputs_progress")
    if outputs_progress is None:
        raise ValueError("No `outputs_progress` in server response")
    progress_pred = outputs_progress.get("progress_pred", [])

    # Extract progress predictions
    if progress_pred and len(progress_pred) > 0:
        progress_array = np.array(progress_pred[0])  # First sample
    else:
        progress_array = np.array([])

    # Extract success predictions if available
    outputs_success = outputs.get("outputs_success", {})
    success_probs = outputs_success.get("success_probs", []) if outputs_success else None
    if success_probs and len(success_probs) > 0:
        success_array = np.array(success_probs[0])

    return progress_array, success_array
