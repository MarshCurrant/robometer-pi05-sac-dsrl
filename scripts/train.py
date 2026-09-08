#!/usr/bin/env python3
"""Train LIBERO MLP SAC in Pi0.5 noise space with RoboMeter rewards.

This is deliberately a LIBERO-only entrypoint. It reuses the regular
``libero_online_rl`` setup, SAC implementation, replay/reward buffer and W&B
logger, while replacing the environment action space with RLinf's 32D DSRL
noise action and a frozen Pi0.5 decoder.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from functools import partial
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint

from robometer_policy_learning.algorithms.sac import SAC, SACConfig
from robometer_policy_learning.buffers.samplers import RandomSampler
from robometer_policy_learning.libero_pi05_dsrl import (
    FrozenRLinfPi05Policy,
    LiberoPi05EvaluationWorker,
    LiberoPi05RobometerRolloutWorker,
    initialize_residual_gaussian_actor,
)
from robometer_policy_learning.libero_pi05_dsrl.training_semantics import (
    resolve_macro_gamma,
)
from robometer_policy_learning.runners.serial_runner import SerialRunner
from robometer_policy_learning.utils.training_utils import (
    build_actor_critic_models,
    create_buffer,
    load_checkpoint,
    save_checkpoint,
    setup_training,
)


def _single_space(env, name: str):
    single_name = f"single_{name}"
    if hasattr(env, single_name):
        return getattr(env, single_name)
    return getattr(env, name)


def _build_reward_semantics_audit(
    cfg: DictConfig,
    *,
    configured_gamma: float,
    discount_unit: str,
    effective_macro_gamma: float,
) -> dict:
    """Validate the formal MLP-SAC to Pi0.5 DSRL reward mapping."""
    errors = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(not bool(cfg.env.use_gt_rewards), "env.use_gt_rewards must be false")
    require(
        not bool(cfg.env.train_terminate_on_success),
        "simulator success must be hidden during training",
    )
    require(
        not bool(cfg.reward_model.use_relative_rewards),
        "formal RoboMeter reward must use absolute progress",
    )
    require(
        bool(cfg.reward_model.add_estimated_reward),
        "RoboMeter progress must be added to the environment step reward",
    )
    require(
        float(cfg.dsrl.macro_environment_reward) == -1.0,
        "official LIBERO DSRL macro environment reward must be -1",
    )
    require(
        list(cfg.env.dino_image_keys) == ["observation/image"],
        "formal reward and SAC observations must use the main camera only",
    )
    require(int(cfg.reward_model.max_frames) == 8, "Robometer-4B max_frames must be 8")
    require(int(cfg.online_algorithm.batch_size) == 128, "SAC batch_size must be 128")
    require(
        str(cfg.dsrl.training_preset) == "sf73jk43_legacy_control",
        "formal control must use the validated sf73jk43 training preset",
    )
    require(
        discount_unit == "low_level" and configured_gamma == 0.99,
        "sf73jk43 discount must be configured as low-level gamma=0.99",
    )
    require(int(cfg.online_algorithm.num_critics) == 5, "SAC must use 5 critics")
    require(
        int(cfg.online_algorithm.n_critics_to_sample) == 2,
        "SAC target must sample 2 critics",
    )
    require(
        int(cfg.online_algorithm.num_critic_updates_per_actor_update) == 1,
        "formal LIBERO MLP control requires one critic update per actor update",
    )
    require(
        float(cfg.online_algorithm.actor_optimizer_lr) == 1e-5,
        "sf73jk43 actor learning rate must be 1e-5",
    )
    require(
        float(cfg.online_algorithm.critic_optimizer_lr) == 1e-5,
        "sf73jk43 critic learning rate must be 1e-5",
    )
    require(
        int(cfg.online_algorithm.num_updates_per_train_step) == 1,
        "one actor update per collected macro transition must be preserved",
    )
    require(
        int(cfg.online_algorithm.ent_coef_update_interval) == 1,
        "sf73jk43 entropy coefficient must update every actor update",
    )
    require(
        cfg.online_algorithm.actor_clip_grad_norm is None,
        "sf73jk43 actor gradient clipping must remain disabled",
    )
    require(
        cfg.online_algorithm.critic_clip_grad_norm is None,
        "sf73jk43 critic gradient clipping must remain disabled",
    )
    require(
        abs(
            effective_macro_gamma
            - resolve_macro_gamma(
                configured_gamma=configured_gamma,
                action_exec_len=int(cfg.dsrl.action_exec_len),
                discount_unit=discount_unit,
            )
        )
        < 1e-12,
        "macro discount does not match the declared discount unit",
    )
    if errors:
        raise ValueError("Invalid formal reward semantics:\n- " + "\n- ".join(errors))

    return {
        "reference": "validated sf73jk43 RoboMeter LIBERO Pi0.5 DSRL control",
        "allowed_changes": [
            "7D direct MLP actor -> 32D residual actor plus frozen Pi0.5 decoder",
            "native simulator success -> enhanced RoboMeter success detector",
        ],
        "replay_reward_formula": "r_k = -1 + absolute_robometer_progress(H_k)",
        "transition_unit": "one Pi0.5 macro action",
        "reward_history": "initial frame plus every executed low-level frame",
        "simulator_success_used_for_training": False,
        "environment_reward_per_macro": float(cfg.dsrl.macro_environment_reward),
        "robometer_use_relative_rewards": False,
        "robometer_add_estimated_reward": True,
        "robometer_max_frames": int(cfg.reward_model.max_frames),
        "reward_camera_keys": list(cfg.env.dino_image_keys),
        "configured_gamma": configured_gamma,
        "discount_unit": discount_unit,
        "action_exec_len": int(cfg.dsrl.action_exec_len),
        "effective_macro_gamma": effective_macro_gamma,
        "episode_metric_env": "sum of pre-relabel macro environment rewards",
        "episode_metric_training": "sum of replay rewards after RoboMeter relabel",
    }


def run(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    if not str(cfg.env.env_name).startswith("libero"):
        raise ValueError("train_libero_mlp_pi05_dsrl.py only supports LIBERO environments")
    if cfg.training.num_envs != 1:
        raise ValueError("The serial Pi0.5 DSRL path currently requires training.num_envs=1")
    if cfg.training.chunk_size is not None:
        raise ValueError("DSRL macro actions require the MLP policy path (training.chunk_size=null)")

    run_root = str(cfg.runtime.output_dir)
    os.environ["RUN_ROOT"] = run_root
    if run_root:
        resolved_config_path = Path(run_root) / "resolved_config.yaml"
        resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, resolved_config_path, resolve=True)

    components = setup_training(cfg)
    device = components.device
    env = components.env
    eval_env = components.eval_env
    logger = components.logger
    wandb_logger = components.wandb_logger

    action_exec_len = int(cfg.dsrl.action_exec_len)
    action_chunk = int(cfg.dsrl.action_chunk)
    noise_dim = int(cfg.dsrl.noise_dim)
    noise_source = str(cfg.dsrl.get("noise_source", "external"))
    force_zero_actor = bool(cfg.dsrl.get("force_zero_actor", False))
    if action_exec_len > action_chunk:
        raise ValueError("dsrl.action_exec_len cannot exceed dsrl.action_chunk")
    if noise_source == "internal_rlinf" and int(cfg.training.num_rollouts) != 0:
        raise ValueError(
            "dsrl.noise_source=internal_rlinf is an eval-only step-0 diagnostic; "
            "formal SAC training must use the external trainable actor"
        )

    logger.info("Loading frozen RLinf Pi0.5 decoder")
    pi05 = FrozenRLinfPi05Policy(
        checkpoint=cfg.dsrl.pi05_checkpoint,
        device=device,
        action_chunk=action_chunk,
        action_dim=int(cfg.dsrl.env_action_dim),
        noise_dim=noise_dim,
        num_steps=int(cfg.dsrl.num_steps),
        precision=cfg.dsrl.precision,
        config_name=str(cfg.dsrl.config_name),
        backend=str(cfg.dsrl.pi05_backend),
        noise_source=noise_source,
        internal_actor_seed=int(cfg.dsrl.actor_seed),
        force_zero_actor=force_zero_actor,
    )

    # setup_training intentionally remains untouched for other pipelines. Its
    # temporary 7D LIBERO actor/critic are replaced here by the 32D DSRL MLPs.
    del components.actor, components.critic, components.v_net
    gc.collect()
    noise_action_space = gym.spaces.Box(
        low=-float(cfg.dsrl.noise_action_bound),
        high=float(cfg.dsrl.noise_action_bound),
        shape=(noise_dim,),
        dtype=np.float32,
    )
    actor_seed = int(cfg.dsrl.actor_seed)
    random.seed(actor_seed)
    np.random.seed(actor_seed)
    torch.manual_seed(actor_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actor_seed)
    actor, critic, v_net = build_actor_critic_models(
        _single_space(env, "observation_space"),
        noise_action_space,
        cfg,
        device,
        components.remove_obs_keys,
    )
    initialize_residual_gaussian_actor(
        actor,
        output_gain=(
            0.0 if force_zero_actor else float(cfg.dsrl.actor_output_init_gain)
        ),
    )
    del v_net

    online_dict = OmegaConf.to_container(cfg.online_algorithm, resolve=True)
    configured_gamma = float(online_dict["gamma"])
    discount_unit = str(cfg.dsrl.get("discount_unit", "low_level"))
    base_learning_starts = int(online_dict["learning_starts"])
    # One replay row advances action_exec_len low-level environment steps. The
    # discount is mapped to that macro transition, while learning_starts keeps
    # the original LIBERO MLP unit: low-level environment steps.
    online_dict["gamma"] = resolve_macro_gamma(
        configured_gamma=configured_gamma,
        action_exec_len=action_exec_len,
        discount_unit=discount_unit,
    )
    online_dict["learning_starts"] = base_learning_starts
    online_dict["target_entropy"] = float(cfg.dsrl.target_entropy)
    online_cfg = SACConfig(**online_dict)
    reward_semantics_audit = _build_reward_semantics_audit(
        cfg,
        configured_gamma=configured_gamma,
        discount_unit=discount_unit,
        effective_macro_gamma=float(online_dict["gamma"]),
    )
    if run_root:
        (Path(run_root) / "reward_semantics_audit.json").write_text(
            json.dumps(reward_semantics_audit, indent=2) + "\n",
            encoding="utf-8",
        )

    buffer = create_buffer(
        sampler=RandomSampler(),
        use_gt_rewards=components.use_gt_rewards,
        use_relative_rewards=components.use_relative_rewards,
        reward_model_exp_cfg=components.reward_model_exp_cfg,
        capacity=int(cfg.buffer.capacity),
        remove_obs_keys=components.remove_obs_keys,
        post_transforms=[components.success_bonus_fn] if components.success_bonus_fn else [],
        use_eval_server=components.use_eval_server,
        eval_server_url=components.eval_server_url,
        eval_server_timeout=components.eval_server_timeout,
        reward_model=components.reward_model,
        reward_relabeling_keys=components.dino_image_keys,
        use_success_detection=bool(cfg.reward_model.use_success_detection),
        success_detection_duration=int(cfg.reward_model.success_detection_duration),
        success_detection_threshold=float(cfg.reward_model.success_detection_threshold),
        success_detection_rule=str(cfg.reward_model.success_detection_rule),
        terminal_adjacent_delta_detection=bool(
            cfg.reward_model.terminal_adjacent_delta_detection
        ),
        terminal_adjacent_delta_threshold=float(
            cfg.reward_model.terminal_adjacent_delta_threshold
        ),
        terminal_adjacent_delta_context_frames=int(
            cfg.reward_model.terminal_adjacent_delta_context_frames
        ),
        reward_max_frames=int(cfg.reward_model.max_frames),
        add_estimated_reward=bool(cfg.reward_model.add_estimated_reward),
    )

    online_cfg.env = env
    online_cfg.actor = actor
    online_cfg.critic = critic
    online_cfg.buffer = buffer
    online_cfg.action_space = noise_action_space
    online_cfg.logger = wandb_logger
    algorithm = SAC(online_cfg)
    if cfg.training.load_dir is not None:
        load_checkpoint(algorithm, cfg.training.load_dir)

    rollout_worker = LiberoPi05RobometerRolloutWorker(
        env=env,
        buffer=buffer,
        num_rollouts=1,
        actor=actor,
        device=device,
        count_by="step",
        num_envs=1,
        pi05_policy=pi05,
        action_exec_len=action_exec_len,
        macro_environment_reward=float(cfg.dsrl.macro_environment_reward),
        reward_relabeling_keys=components.dino_image_keys,
        reward_autocast_dtype=str(cfg.dsrl.reward_autocast_dtype),
    )
    eval_worker_factory = partial(
        LiberoPi05EvaluationWorker,
        pi05_policy=pi05,
        action_exec_len=action_exec_len,
    )

    wandb_logger.log_hparams(
        {
            "dsrl/configured_gamma": configured_gamma,
            "dsrl/discount_unit": discount_unit,
            "dsrl/effective_macro_gamma": online_dict["gamma"],
            "dsrl/learning_starts_low_level_env_steps": online_dict[
                "learning_starts"
            ],
            "dsrl/noise_dim": noise_dim,
            "dsrl/model_action_horizon": pi05.model_action_horizon,
            "dsrl/action_chunk": action_chunk,
            "dsrl/action_exec_len": action_exec_len,
            "dsrl/macro_environment_reward": float(
                cfg.dsrl.macro_environment_reward
            ),
            "dsrl/pi05_num_steps": int(cfg.dsrl.num_steps),
            "dsrl/pi05_backend": str(cfg.dsrl.pi05_backend),
            "dsrl/target_entropy": float(cfg.dsrl.target_entropy),
            "dsrl/actor_output_init_gain": float(cfg.dsrl.actor_output_init_gain),
            "dsrl/force_zero_actor": force_zero_actor,
            "dsrl/actor_seed": actor_seed,
            "dsrl/noise_source": noise_source,
            "dsrl/reward_autocast_dtype": str(cfg.dsrl.reward_autocast_dtype),
            "reward/success_detection_threshold": float(
                cfg.reward_model.success_detection_threshold
            ),
            "reward/success_detection_duration": int(
                cfg.reward_model.success_detection_duration
            ),
            "reward/success_detection_rule": str(
                cfg.reward_model.success_detection_rule
            ),
            "reward/terminal_adjacent_delta_threshold": float(
                cfg.reward_model.terminal_adjacent_delta_threshold
            ),
            "reward/max_frames": int(cfg.reward_model.max_frames),
            "env/train_terminate_on_success": bool(
                cfg.env.train_terminate_on_success
            ),
            "env/eval_terminate_on_success": bool(
                cfg.env.eval_terminate_on_success
            ),
            "dsrl/updates_per_macro_action": int(
                online_dict["num_updates_per_train_step"]
            ),
        }
    )
    logger.info(
        "LIBERO Pi0.5 DSRL mapping: "
        f"noise_dim={noise_dim} model_horizon={pi05.model_action_horizon} "
        f"action_chunk={action_chunk} execute={action_exec_len} "
        f"macro_environment_reward={float(cfg.dsrl.macro_environment_reward):.3f} "
        f"configured_gamma={configured_gamma:.6f} discount_unit={discount_unit} "
        f"macro_gamma={online_dict['gamma']:.6f} "
        f"base_warmup={base_learning_starts} "
        f"macro_env_warmup={online_dict['learning_starts']} "
        f"target_entropy={online_dict['target_entropy']:.1f} "
        f"updates_per_macro={online_dict['num_updates_per_train_step']}"
    )

    runner = SerialRunner(
        env=env,
        eval_env=eval_env,
        algorithm=algorithm,
        buffer=buffer,
        actor=actor,
        rollout_worker=rollout_worker,
        num_rollouts=int(cfg.training.num_rollouts),
        eval_freq=int(cfg.eval.eval_freq),
        eval_kwargs={
            "num_episodes": int(cfg.eval.eval_num_episodes),
            "record_video": bool(cfg.eval.eval_record_video),
        },
        logger=wandb_logger,
        evaluation_worker_class=eval_worker_factory,
        eval_on_first_step=bool(cfg.eval.eval_on_first_step),
        save_dir=components.save_dir,
        save_interval=int(cfg.training.save_interval),
    )

    rprint(
        f"Starting LIBERO MLP SAC + Pi0.5 DSRL for {cfg.training.num_rollouts} environment steps"
    )
    try:
        runner.run()
        save_checkpoint(algorithm, components.save_dir, "final")
    except KeyboardInterrupt:
        save_checkpoint(algorithm, components.save_dir, "interrupted")
    finally:
        env.close()
        eval_env.close()
        pi05.close()
        try:
            wandb_logger.finish()
        except Exception as exc:  # noqa: BLE001 - logging cannot block checkpoint finalization
            logger.warning(f"W&B finalization failed after checkpoint handling: {exc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Resolved experiment YAML")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable OmegaConf dotlist override",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from robometer_policy_learning.reproducibility import (
        load_experiment_config,
        prepare_run,
    )

    cfg = load_experiment_config(args.config)
    if args.set:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))
    cfg = prepare_run(cfg, source_config=args.config)
    run(cfg)


if __name__ == "__main__":
    main()
