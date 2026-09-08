from typing import Dict, Any, List
import time
import numpy as np
import torch

from robometer_policy_learning.distributed.clients.remote_replay_buffer_client import (
    RemoteReplayBufferClient,
)
from loguru import logger


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


class StreamingBufferAdapter:
    """
    Implements a buffer-like API with add(), but streams transitions to a remote learner.
    Can be dropped into RolloutWorker as the buffer.
    """

    def __init__(self, learner_address: str, flush_every: int = 256, max_message_mb: int = 64):
        self.client = RemoteReplayBufferClient(
            learner_address, max_batch_size=flush_every, max_message_mb=max_message_mb
        )
        self._local: List[Dict[str, Any]] = []
        self._flush_every = flush_every

    def add(self, obs, action, reward, next_obs, done, truncated, **kwargs):
        obs_np = {k: _to_numpy(v) for k, v in (obs or {}).items()} if isinstance(obs, dict) else _to_numpy(obs)
        next_obs_np = (
            {k: _to_numpy(v) for k, v in (next_obs or {}).items()}
            if isinstance(next_obs, dict)
            else _to_numpy(next_obs)
        )
        tr = dict(
            obs=obs_np if isinstance(obs_np, dict) else {"obs": obs_np},
            action=_to_numpy(action),
            reward_env=float(reward),
            next_obs=next_obs_np if isinstance(next_obs_np, dict) else {"obs": next_obs_np},
            done=bool(done),
            truncated=bool(truncated),
            episode_id=kwargs.get("episode_id", ""),
            step_in_episode=kwargs.get("step_in_episode", 0),
            timestamp=kwargs.get("timestamp", int(time.time_ns())),
            info=kwargs.get("info", {}),
        )
        self._local.append(tr)
        if len(self._local) >= self._flush_every:
            self.flush()

    def flush(self):
        if not self._local:
            return
        try:
            # Enqueue to persistent stream; defer ack since it's a server-side return value
            self.client.enqueue(self._local)
        except Exception as e:
            logger.error(f"[STREAM] send error: {e}")
        self._local = []

    def __len__(self):
        # Not meaningful for remote adapter; return 0 to satisfy interfaces that call len(buffer)
        return 0
