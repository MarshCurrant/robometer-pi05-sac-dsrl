from __future__ import annotations

from typing import List, Dict, Any


def batch_transitions(transitions: List[Any]) -> Dict[str, Any]:
    """Convert a list of Transition objects into a batched dict-of-lists.

    The returned dict contains keys:
      - obs: dict of lists (per-key observations)
      - next_obs: dict of lists (per-key observations)
      - action: list
      - reward: list[float]
      - done: list[bool]
      - truncated: list[bool]
      - episode_id, step_in_episode, max_steps_in_episode, timestamp, language_instruction: lists
    """
    if not transitions:
        return {}

    first_obs = transitions[0].obs
    is_dict_obs = isinstance(first_obs, dict)

    batched: Dict[str, Any] = {
        "action": [],
        "reward": [],
        "done": [],
        "truncated": [],
        "episode_id": [],
        "step_in_episode": [],
        "max_steps_in_episode": [],
        "timestamp": [],
        "language_instruction": [],
        "info": [],  # Info dict for retroactive updates
    }

    if is_dict_obs:
        batched["obs"] = {k: [] for k in first_obs.keys()}
        batched["next_obs"] = {k: [] for k in transitions[0].next_obs.keys()}
        for tr in transitions:
            for k, v in tr.obs.items():
                batched["obs"][k].append(v)
            for k, v in tr.next_obs.items():
                batched["next_obs"][k].append(v)
            batched["action"].append(tr.action)
            batched["reward"].append(tr.reward)
            batched["done"].append(tr.done)
            batched["truncated"].append(tr.truncated)
            batched["episode_id"].append(tr.episode_id)
            batched["step_in_episode"].append(tr.step_in_episode)
            batched["max_steps_in_episode"].append(tr.max_steps_in_episode)
            batched["timestamp"].append(tr.timestamp)
            batched["language_instruction"].append(tr.language_instruction)
            batched["info"].append(tr.info)
    else:
        batched["obs"] = [tr.obs for tr in transitions]
        batched["next_obs"] = [tr.next_obs for tr in transitions]
        for tr in transitions:
            batched["action"].append(tr.action)
            batched["reward"].append(tr.reward)
            batched["done"].append(tr.done)
            batched["truncated"].append(tr.truncated)
            batched["episode_id"].append(tr.episode_id)
            batched["step_in_episode"].append(tr.step_in_episode)
            batched["max_steps_in_episode"].append(tr.max_steps_in_episode)
            batched["timestamp"].append(tr.timestamp)
            batched["language_instruction"].append(tr.language_instruction)
            batched["info"].append(tr.info)

    return batched


def unbatch_transitions(batched: Dict[str, Any]) -> List[Any]:
    """Convert a batched dict-of-lists back to a list of Transition objects."""
    if not batched:
        return []

    num = len(batched.get("reward", []))
    transitions: List[Any] = []

    is_dict_obs = isinstance(batched.get("obs", {}), dict)

    for i in range(num):
        if is_dict_obs:
            obs_i = {k: batched["obs"][k][i] for k in batched["obs"].keys()}
            next_obs_i = {k: batched["next_obs"][k][i] for k in batched["next_obs"].keys()}
        else:
            obs_i = batched["obs"][i]
            next_obs_i = batched["next_obs"][i]

        # Lazy import to avoid circular import at module import time
        from robometer_policy_learning.buffers.base_replay_buffer import Transition

        tr = Transition(
            obs=obs_i,
            action=batched["action"][i] if "action" in batched else None,
            reward=batched["reward"][i] if "reward" in batched else 0.0,
            next_obs=next_obs_i,
            done=batched["done"][i] if "done" in batched else False,
            truncated=batched["truncated"][i] if "truncated" in batched else False,
            episode_id=batched.get("episode_id", [None] * num)[i],
            step_in_episode=batched.get("step_in_episode", [0] * num)[i],
            max_steps_in_episode=batched.get("max_steps_in_episode", [0] * num)[i],
            timestamp=batched.get("timestamp", [None] * num)[i],
            language_instruction=batched.get("language_instruction", [None] * num)[i],
            info=batched.get("info", [None] * num)[i],
        )
        transitions.append(tr)

    return transitions


def apply_transforms_batched(batched: Dict[str, Any], transforms: List[Any]) -> Dict[str, Any]:
    """Apply post transforms in batch when possible, falling back per-transition.

    If a transform supports batched input (has attribute supports_batch True or
    succeeds with dict input), we use it directly; otherwise fall back to
    element-wise application on the list of Transition objects.
    """
    if not transforms or not batched:
        return batched

    current = batched
    for transform in transforms:
        supports_batch = getattr(transform, "supports_batch", False)
        if supports_batch:
            try:
                current = transform(current)
                continue
            except Exception:
                # Fall back if transform raises for batch
                pass
        # Fallback: unbatch -> apply per-transition -> re-batch
        transitions = unbatch_transitions(current)
        out_list: List[Any] = []
        for tr in transitions:
            try:
                out_list.append(transform(tr))
            except Exception:
                out_list.append(tr)
        current = batch_transitions(out_list)
    return current
