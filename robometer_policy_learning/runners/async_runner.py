"""
Async runner entry-points for distributed training/collection.

Modes:
- learner: start gRPC services (policy + ingestion) and run local training loop threads
- robot: run rollout worker and stream transitions to learner; poll latest policy
- eval: periodic evaluation pulling latest policy

This file wires together services; business logic remains in existing modules.
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Optional

import torch

from robometer_policy_learning.distributed.servers.learner_server import LearnerServer
from robometer_policy_learning.distributed.clients.policy_client import PolicyClient
from robometer_policy_learning.distributed.clients.remote_replay_buffer_client import (
    RemoteReplayBufferClient,
)
from robometer_policy_learning.rollouts.rollout_worker import RolloutWorker


def run_learner(algorithm, buffer, reward_model, host: str, port: int, relabel_mode: str):
    server = LearnerServer(buffer=buffer, reward_model=reward_model, host=host, port=port, relabel_mode=relabel_mode)
    server.start()

    # Training thread updates params into policy server periodically
    def training_loop():
        last_push_step = -1
        while True:
            metrics = algorithm.train_step(logging_prefix="online/policy")
            # Push new actor weights every k steps
            step = getattr(algorithm, "_n_updates", 0)
            if step // 50 != last_push_step // 50:  # push every 50 updates
                server.set_actor_state_dict(algorithm.actor.state_dict())
                last_push_step = step

    t = threading.Thread(target=training_loop, daemon=True)
    t.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


def run_robot(env, actor, learner_addr: str, buffer_client: RemoteReplayBufferClient, device: torch.device):
    policy_client = PolicyClient(learner_addr)
    # Initial weights fetch
    state_dict = policy_client.fetch_latest()
    actor.load_state_dict(state_dict)

    rollout_worker = RolloutWorker(
        env=env,
        buffer=None,  # we stream ourselves
        num_rollouts=1,
        actor=actor,
        device=device,
        count_by="step",
    )

    # Simple loop: collect a chunk of transitions and send
    transitions_buffer = []

    def add_transition(obs, action, reward_env, next_obs, done, truncated, episode_id, step_in_episode):
        transitions_buffer.append(
            dict(
                obs=obs,
                action=action,
                reward_env=reward_env,
                next_obs=next_obs,
                done=done,
                truncated=truncated,
                episode_id=episode_id,
                step_in_episode=step_in_episode,
                timestamp=int(time.time_ns()),
                info={},
            )
        )

    # Monkey-patch: intercept buffer.add in worker by wrapping its internal call site would need refactor.
    # For now, we implement a light loop here (pseudocode placeholder; user integrates into their env loop).
    raise NotImplementedError(
        "Integrate run_robot with your collection loop or extend RolloutWorker to call a callback per step."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["learner", "robot", "eval"], required=True)
    parser.add_argument("--addr", default="0.0.0.0:50051")
    parser.add_argument("--relabel-mode", choices=["pre", "post"], default="pre")
    args = parser.parse_args()

    if args.mode == "learner":
        raise NotImplementedError("Wire this into your training script to pass algorithm/buffer/reward_model")
    elif args.mode == "robot":
        raise NotImplementedError("Provide env, actor, and learner addr to run robot mode")
    else:
        raise NotImplementedError("Eval mode to be implemented similarly to robot weights fetching")


if __name__ == "__main__":
    main()
