"""Training utility functions."""

import os
import copy
import gymnasium as gym
import torch
import numpy as np
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Tuple, Optional, Dict, List
from omegaconf import DictConfig, OmegaConf

from robometer_policy_learning.utils.transitions_transforms import success_bonus as transition_success_bonus
from robometer_policy_learning.modules.mlp import MLPActor, MLPActorConfig, MLPCritic, MLPCriticConfig
from robometer_policy_learning.modules.rnn import RNNActor, RNNActorConfig, RNNCritic, RNNCriticConfig
from robometer_policy_learning.modules.transformer import TransformerActor, TransformerActorConfig, TransformerCritic, TransformerCriticConfig

from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.robometer_replay_buffer import RobometerReplayBuffer

# Setup imports
from transformers import AutoModel, AutoImageProcessor
from robometer_policy_learning.utils.env_utils import make_env
from robometer_policy_learning.utils.transitions_transforms import SuccessBonusTransform
from PIL import Image
from rich import print as rprint
from robometer.utils.logger import setup_loguru_logging, get_logger
from datetime import datetime
from robometer_policy_learning.loggers.wandb_logger import WandbLogger

logger = get_logger()


def flatten_obs(obs):
    """Flatten everything except for the first dimension, then concatenate all the flattened values."""
    return torch.cat([v.view(v.size(0), -1) for k, v in obs.items()], dim=-1)


def save_checkpoint(algorithm, save_dir, step):
    """Save model checkpoint."""
    checkpoint_dir = os.path.join(save_dir, str(step))
    # Create the base save directory
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    algorithm.save(checkpoint_dir)


def load_checkpoint(algorithm, load_dir):
    """Load model checkpoint."""
    # Use the algorithm's built-in load method
    algorithm.load(load_dir)

    return algorithm.step_counter


def build_actor_critic_models(observation_space, action_space, cfg, device, remove_obs_keys=None):
    """Build actor, critic, and v_net models based on configuration."""

    new_observation_space = copy.deepcopy(observation_space)
    if remove_obs_keys is not None:
        for key in remove_obs_keys:
            new_observation_space.spaces.pop(key, None)

    print(f"----- New observation space -----")
    print(new_observation_space)
    print(f"----- Removed observation keys -----")
    print(remove_obs_keys)

    # if remove_obs_keys is not None:
    #    for key in remove_obs_keys:
    #        featurizer.pop(key, None)

    # Build featurizer config - ObservationFeaturizer will handle IMPALA if configured
    # Default: use MLP featurizers for all keys
    FEATURIZER_DIM = 384
    featurizer = {}
    for key in new_observation_space.spaces:
        featurizer[key] = [FEATURIZER_DIM]

    # Extract IMPALA config if present
    image_encoder_type = OmegaConf.select(cfg, "model.image_encoder_type", default=None)
    impala_config = {}
    if image_encoder_type == "impala":
        impala_config = {
            "image_encoder_type": image_encoder_type,
            "impala_nn_scale": OmegaConf.select(cfg, "model.impala_nn_scale", default=1),
            "impala_num_blocks_per_stack": OmegaConf.select(cfg, "model.impala_num_blocks_per_stack", default=2),
            "impala_use_smaller": OmegaConf.select(cfg, "model.impala_use_smaller", default=False),
            "impala_output_dim": FEATURIZER_DIM,
        }

    if cfg.training.chunk_size is None:
        # MLP architecture
        policy_cfg = cfg.policy
        value_cfg = cfg.value_function

        # Extract policy config fields for unpacking
        policy_kwargs = OmegaConf.to_container(policy_cfg.mlp, resolve=True)
        policy_kwargs["hidden_dims"] = tuple(policy_kwargs["hidden_dims"])
        # Add top-level policy config fields
        policy_kwargs.update(
            {
                "use_tanh_output": policy_cfg.use_tanh_output,
                "deterministic": policy_cfg.deterministic,
                "log_std_init": policy_cfg.log_std_init,
                "log_std_min": policy_cfg.log_std_min,
                "log_std_max": policy_cfg.log_std_max,
            }
        )

        actor_config = MLPActorConfig(
            observation_space=new_observation_space,
            action_space=action_space,
            preprocess_obs_transform=None,
            featurizer=featurizer,
            remove_obs_keys=remove_obs_keys,
            **policy_kwargs,
            **impala_config,
        )
        actor = MLPActor(actor_config)

        value_kwargs = OmegaConf.to_container(value_cfg.mlp, resolve=True)
        value_kwargs["hidden_dims"] = tuple(value_kwargs["hidden_dims"])

        critic_config = MLPCriticConfig(
            observation_space=new_observation_space,
            action_space=action_space,
            preprocess_obs_transform=None,
            featurizer=featurizer,
            use_action=True,
            remove_obs_keys=remove_obs_keys,
            **value_kwargs,
            **impala_config,
        )
        critic = MLPCritic(critic_config)
        v_net_config = dataclasses.replace(critic_config)
        v_net_config.use_action = False
        v_net_config.use_layer_norm = False
        v_net = MLPCritic(v_net_config)
    else:
        # Chunked architecture (RNN or Transformer)
        policy_cfg = cfg.policy
        value_cfg = cfg.value_function

        if cfg.training.use_rnn:
            # RNN architecture
            # Extract RNN policy config fields for unpacking
            rnn_policy_kwargs = OmegaConf.to_container(policy_cfg.rnn, resolve=True)
            # Add top-level policy config fields
            rnn_policy_kwargs.update(
                {
                    "log_std_init": policy_cfg.log_std_init,
                    "log_std_min": policy_cfg.log_std_min,
                    "log_std_max": policy_cfg.log_std_max,
                }
            )

            actor_config = RNNActorConfig(
                observation_space=new_observation_space,
                action_space=action_space,
                chunk_size=cfg.training.chunk_size,
                remove_obs_keys=remove_obs_keys,
                featurizer=featurizer,
                **rnn_policy_kwargs,
                **impala_config,
            )
            actor = RNNActor(actor_config)

            rnn_value_kwargs = OmegaConf.to_container(value_cfg.rnn, resolve=True)

            critic_config = RNNCriticConfig(
                observation_space=new_observation_space,
                action_space=action_space,
                chunk_size=cfg.training.chunk_size,
                remove_obs_keys=remove_obs_keys,
                featurizer=featurizer,
                **rnn_value_kwargs,
                **impala_config,
            )
            critic = RNNCritic(critic_config)
        else:
            # Transformer architecture
            # Extract Transformer policy config fields for unpacking
            transformer_policy_kwargs = OmegaConf.to_container(policy_cfg.transformer, resolve=True)
            # Add top-level policy config fields
            transformer_policy_kwargs.update(
                {
                    "log_std_init": policy_cfg.log_std_init,
                    "log_std_min": policy_cfg.log_std_min,
                    "log_std_max": policy_cfg.log_std_max,
                }
            )
            transformer_policy_kwargs.pop("pooling_strategy", None)

            actor_config = TransformerActorConfig(
                observation_space=new_observation_space,
                action_space=action_space,
                featurizer=featurizer,
                chunk_size=cfg.training.chunk_size,
                dinov2_model=None,
                dinov2_processor=None,
                use_language_embeddings=False,
                remove_obs_keys=remove_obs_keys,
                **transformer_policy_kwargs,
                **impala_config,
            )
            actor = TransformerActor(actor_config)
            transformer_value_kwargs = OmegaConf.to_container(value_cfg.transformer, resolve=True)

            critic_config = TransformerCriticConfig(
                observation_space=new_observation_space,
                action_space=action_space,
                featurizer=featurizer,
                chunk_size=cfg.training.chunk_size,
                dinov2_model=None,
                dinov2_processor=None,
                use_language_embeddings=False,
                remove_obs_keys=remove_obs_keys,
                use_action=True,
                **transformer_value_kwargs,
                **impala_config,
            )
            critic = TransformerCritic(critic_config)

        v_net_config = dataclasses.replace(critic_config)
        v_net_config.use_action = False
        v_net_config.use_layer_norm = False
        if cfg.training.use_rnn:
            v_net = RNNCritic(v_net_config)
        else:
            v_net = TransformerCritic(v_net_config)

    actor = actor.to(device)
    critic = critic.to(device)
    v_net = v_net.to(device)

    # Log model info
    logger.info(f"Actor: {actor.__class__.__name__}")
    rprint(actor)
    logger.info(f"Critic: {critic.__class__.__name__}")
    rprint(critic)
    logger.info(f"V-Net: {v_net.__class__.__name__}")
    rprint(v_net)

    return actor, critic, v_net


@dataclass
class TrainingComponents:
    """Holds all components needed for RL training."""

    # Core
    cfg: DictConfig
    device: torch.device

    # Models
    dinov2_model: Any
    dinov2_processor: Any
    sentence_model: Any

    # Environments
    env: Any
    eval_env: Any
    remove_obs_keys: list
    dino_image_keys: list

    # Actor/Critic
    actor: Any
    critic: Any
    v_net: Any

    # Reward model (optional)
    reward_model: Any = None
    reward_model_exp_cfg: Any = None
    use_gt_rewards: bool = True
    use_relative_rewards: bool = False
    use_eval_server: bool = False
    eval_server_url: Optional[str] = None
    eval_server_timeout: float = 120.0

    # Algorithms
    offline_algo: Any = None
    online_algo: Any = None

    # Buffers
    offline_buffer: Any = None
    online_buffer: Any = None
    buffer: Any = None

    # Logging
    logger: Any = None
    wandb_logger: Any = None
    save_dir: str = ""

    # Success bonus
    success_bonus_fn: Any = None


def create_buffer(
    sampler: Any,
    use_gt_rewards: bool,
    use_relative_rewards: bool,
    capacity: int,
    remove_obs_keys: list,
    post_transforms: list,
    reward_model: Optional[Any] = None,
    reward_model_exp_cfg: Optional[Any] = None,
    h5_paths: Optional[list] = None,
    use_full_state: bool = False,
    sentence_model: Optional[Any] = None,
    dinov2_model: Optional[Any] = None,
    dinov2_processor: Optional[Any] = None,
    reward_relabeling_keys: List[str] = ["image"],
    use_success_fail_buffer: bool = False,
    success_fail_sample_ratio: float = 0.5,
    use_async_reward_relabel: bool = False,
    reward_relabel_address: Optional[str] = None,
    reward_relabel_max_queue_size: int = 100,
    reward_relabel_timeout: float = 60.0,
    reward_relabel_flush_interval: float = 0.1,
    use_eval_server: bool = False,
    eval_server_url: Optional[str] = None,
    eval_server_timeout: float = 120.0,
    use_success_detection: bool = False,
    success_detection_duration: int = 2,
    success_detection_threshold: float = 0.65,
    success_detection_rule: str = "majority_window",
    terminal_adjacent_delta_detection: bool = False,
    terminal_adjacent_delta_threshold: float = float("inf"),
    terminal_adjacent_delta_context_frames: int = 4,
    reward_max_frames: Optional[int] = None,
    add_estimated_reward: bool = False,
) -> Any:
    """
    Create a replay buffer for training.

    Args:
        reward_model: Reward model instance (None for ground truth rewards)
        reward_model_exp_cfg: Experiment config for reward model
        sampler: Sampler for the buffer
        use_gt_rewards: Whether to use ground truth rewards
        use_relative_rewards: Whether to use relative rewards
        capacity: Buffer capacity (for online buffers)
        remove_obs_keys: Keys to remove from observations
        post_transforms: Post-transforms to apply
        h5_paths: List of H5 dataset paths (for offline buffers)
        use_full_state: Whether to use full state (no DINO embeddings)
        sentence_model: Sentence transformer model
        dinov2_model: DINOv2 model
        dinov2_processor: DINOv2 processor
        use_success_fail_buffer: If True, create SuccessFailureReplayBuffer
        success_fail_sample_ratio: Ratio for sampling from success vs failure buffer (default 0.5)
        use_async_reward_relabel: If True, wrap buffer with AsyncRewardRelabelBuffer
        reward_relabel_address: Address for remote reward relabeling server (required if use_async_reward_relabel=True)
        reward_relabel_max_queue_size: Max queue size for reward relabeling client
        reward_relabel_timeout: Timeout for reward relabeling client
        reward_relabel_flush_interval: Flush interval for reward relabeling client
        use_success_detection: Whether to use success detection of the reward model
        success_detection_duration: Number of consecutive time steps to detect success (only used with reward relabeling)
        success_detection_threshold: Threshold for success detection (only used with reward relabeling)
        add_estimated_reward: Whether to add estimated reward to the ground truth reward (useful in sparse reward settings)

    Returns:
        Configured replay buffer
    """
    # Handle distributed reward relabeling
    if use_async_reward_relabel:
        from robometer_policy_learning.buffers.remote_reward_relabel_buffer import (
            AsyncRewardRelabelBuffer,
        )
        from robometer_policy_learning.distributed.clients.reward_relabel_client import (
            RewardRelabelClient,
        )

        if reward_relabel_address is None:
            raise ValueError("reward_relabel_address is required when use_async_reward_relabel=True")
        if h5_paths is not None:
            raise ValueError(
                "Distributed reward relabeling is only supported for online buffers (h5_paths must be None)"
            )

        # Create underlying buffer (without reward relabeling)
        underlying_buffer = ReplayBuffer(
            capacity=capacity,
            remove_obs_keys=remove_obs_keys,
            post_transforms=post_transforms,
            sampler=sampler,
        )

        # Create reward relabeling client
        reward_relabel_client = RewardRelabelClient(
            address=reward_relabel_address,
            max_queue_size=reward_relabel_max_queue_size,
            timeout=reward_relabel_timeout,
            flush_interval=reward_relabel_flush_interval,
        )

        # Wrap with remote reward relabeling
        # Get batch_size from config if available, default to 32
        batch_size = 32  # Default batch size for remote reward relabeling
        # This could be added to config later if needed
        return AsyncRewardRelabelBuffer(
            underlying_buffer=underlying_buffer,
            reward_relabel_client=reward_relabel_client,
            use_relative_rewards=use_relative_rewards,
            batch_size=batch_size,
            remove_obs_keys=remove_obs_keys,
            post_transforms=post_transforms,
            sampler=sampler,
        )

    def _create_single_buffer():
        """Helper to create a single buffer instance."""
        if h5_paths is not None:
            from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer
            from robometer_policy_learning.buffers.robometer_replay_buffer import (
                RobometerH5ReplayBuffer,
            )

            # Offline buffer (from H5 dataset)
            if not use_full_state:
                return RobometerH5ReplayBuffer(
                    reward_model=reward_model,
                    reward_model_config=reward_model_exp_cfg,
                    h5_paths=h5_paths,
                    sampler=sampler,
                    use_relative_rewards=use_relative_rewards,
                    remove_obs_keys=remove_obs_keys,
                    post_transforms=post_transforms,
                    sentence_model=sentence_model,
                    dinov2_model=dinov2_model,
                    dinov2_processor=dinov2_processor,
                    use_eval_server=use_eval_server,
                    eval_server_url=eval_server_url,
                    eval_server_timeout=eval_server_timeout,
                    reward_relabeling_keys=reward_relabeling_keys,
                    use_success_detection=use_success_detection,
                    success_detection_duration=success_detection_duration,
                    success_detection_threshold=success_detection_threshold,
                    add_estimated_reward=add_estimated_reward,
                )
            else:
                assert use_gt_rewards, "use_gt_rewards must be True when use_full_state is True"
                return H5ReplayBuffer(
                    h5_paths=h5_paths,
                    sampler=sampler,
                    remove_obs_keys=remove_obs_keys,
                    post_transforms=post_transforms,
                )
        else:
            # Online buffer (in-memory)
            if reward_model is not None or use_eval_server:
                return RobometerReplayBuffer(
                    reward_model=reward_model,
                    reward_model_config=reward_model_exp_cfg,
                    sampler=sampler,
                    use_relative_rewards=use_relative_rewards,
                    capacity=capacity,
                    remove_obs_keys=remove_obs_keys,
                    post_transforms=post_transforms,
                    use_eval_server=use_eval_server,
                    eval_server_url=eval_server_url,
                    eval_server_timeout=eval_server_timeout,
                    reward_relabeling_keys=reward_relabeling_keys,
                    use_success_detection=use_success_detection,
                    success_detection_duration=success_detection_duration,
                    success_detection_threshold=success_detection_threshold,
                    success_detection_rule=success_detection_rule,
                    terminal_adjacent_delta_detection=terminal_adjacent_delta_detection,
                    terminal_adjacent_delta_threshold=terminal_adjacent_delta_threshold,
                    terminal_adjacent_delta_context_frames=terminal_adjacent_delta_context_frames,
                    max_frames=reward_max_frames,
                    add_estimated_reward=add_estimated_reward,
                )
            else:
                return ReplayBuffer(
                    capacity=capacity,
                    remove_obs_keys=remove_obs_keys,
                    post_transforms=post_transforms,
                    sampler=sampler,
                )

    if not use_success_fail_buffer:
        # Standard single buffer
        logger.info(f"Creating standard buffer (capacity={capacity})")
        return _create_single_buffer()
    else:
        from robometer_policy_learning.buffers.success_failure_replay_buffer import (
            SuccessFailureReplayBuffer,
        )

        # Success/Failure buffer with two underlying buffers
        logger.info(f"Creating SuccessFailureReplayBuffer with sample_ratio={success_fail_sample_ratio}")
        logger.info(f"  Success buffer capacity: {capacity}")
        logger.info(f"  Failure buffer capacity: {capacity}")
        success_buffer = _create_single_buffer()
        failure_buffer = _create_single_buffer()

        return SuccessFailureReplayBuffer(
            success_buffer=success_buffer,
            failure_buffer=failure_buffer,
            sample_ratio=success_fail_sample_ratio,
            obs_keys=None,
            remove_obs_keys=remove_obs_keys,
            rename_obs_keys=None,
            sampler=sampler,
        )


def setup_training(
    cfg: DictConfig,
) -> TrainingComponents:
    """
    Args:
        cfg: Hydra config

    Returns:
        TrainingComponents with all initialized components
    """
    output_dir = str(OmegaConf.select(cfg, "runtime.output_dir", default="outputs/run"))
    os.makedirs(output_dir, exist_ok=True)

    # Setup loguru logging
    log_level = cfg.log_level if hasattr(cfg, "log_level") else "INFO"
    setup_loguru_logging(log_level=log_level, output_dir=output_dir)

    # Setup wandb logger
    string_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = str(cfg.logging.wandb_name)
    wandb_entity = OmegaConf.select(cfg, "logging.wandb_entity", default=None)
    if wandb_entity is not None and not str(wandb_entity).strip():
        wandb_entity = None
    wandb_logger = WandbLogger(
        exp_name=exp_name,
        offline=cfg.logging.wandb_offline,
        project=cfg.logging.wandb_project,
        entity=wandb_entity,
        log_dir=str(cfg.logging.wandb_log_dir_base),
        prefix=None,
        group=OmegaConf.select(cfg, "logging.wandb_group", default=None),
        tags=list(OmegaConf.select(cfg, "logging.wandb_tags", default=[])),
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup models
    logger.info("Setting up models")
    dino_image_keys = cfg.env.get("dino_image_keys", ["image"])
    remove_obs_keys = cfg.env.get("extra_keys_to_drop", [])
    if cfg.model.dinov2_model is not None:
        dinov2_model = AutoModel.from_pretrained(cfg.model.dinov2_model).to(device).eval()
        dinov2_processor = AutoImageProcessor.from_pretrained(cfg.model.dinov2_model)
        remove_obs_keys += dino_image_keys # remove image keys since we use the dino embedding wrapper
    else:
        dinov2_model = None
        dinov2_processor = None
    if cfg.model.sentence_model is not None:
        from sentence_transformers import SentenceTransformer

        sentence_model = SentenceTransformer(cfg.model.sentence_model)
    else:
        sentence_model = None

    # Setup reward model
    logger.info("Setting up reward model")
    reward_model_cfg = OmegaConf.select(cfg, "reward_model", default=None)
    use_gt_rewards = cfg.env.use_gt_rewards
    # use_relative_rewards can be set in config.yaml under reward_model, default to False
    use_relative_rewards = (
        reward_model_cfg.use_relative_rewards
        if reward_model_cfg is not None and hasattr(reward_model_cfg, "use_relative_rewards")
        else False
    )

    reward_model = None
    reward_model_exp_cfg = None
    use_eval_server = False
    eval_server_url = None

    if reward_model_cfg is not None:
        model_path = reward_model_cfg["model_path"]
        if model_path is None:
            # Connect to eval_server instead of loading locally
            use_eval_server = True
            eval_server_url = f"{reward_model_cfg.get('eval_server_url', 'http://localhost')}:{reward_model_cfg.get('eval_server_port', 8000)}"
            logger.info(f"Using eval_server at {eval_server_url} for reward computation")
        else:
            # Load model locally
            from robometer.utils.save import load_model_from_hf

            reward_model_exp_cfg, tokenizer, processor, reward_model = load_model_from_hf(
                model_path=model_path,
                device=device,
            )
            logger.info(f"Loaded reward model locally from {model_path}")
    else:
        use_gt_rewards = True

    if "libero" in cfg.env.env_name:
        # setup so we can parse make_env properly
        env_name = cfg.env.env_name + "/" + str(cfg.env.task_id)
    else:
        env_name = cfg.env.env_name

    # Setup environments
    logger.info("Setting up environments")
    env, eval_env = make_env(
        env_name=env_name,
        num_envs=cfg.training.num_envs,
        max_episode_steps=cfg.env.max_episode_steps,
        chunk_size=cfg.training.chunk_size,
        use_full_state=cfg.env.use_full_state,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        dino_image_keys=dino_image_keys,
        device=device,
        sentence_model=sentence_model,
        render_mode="rgb_array",
        terminate_on_success=bool(cfg.env.get("train_terminate_on_success", True)),
        eval_terminate_on_success=bool(cfg.env.get("eval_terminate_on_success", True)),
        seed=cfg.training.seed,
        eval_seed=cfg.env.get("eval_seed", cfg.training.seed),
        reset_settle_steps=int(cfg.env.get("reset_settle_steps", 0)),
    )

    # Get action and observation spaces
    if hasattr(env, "single_action_space"):
        action_space = env.single_action_space
    else:
        action_space = env.action_space

    if hasattr(env, "single_observation_space"):
        observation_space = env.single_observation_space
    else:
        observation_space = env.observation_space

    # Log environment info
    logger.info(f"Observation space: {env.observation_space if hasattr(env, 'observation_space') else 'N/A'}")
    logger.info(f"Action space: {env.action_space if hasattr(env, 'action_space') else 'N/A'}")

    # Save example image if available
    obs, _ = env.reset()
    if "image" in obs:
        ex_img = obs["image"][0]
        ex_img_pil = Image.fromarray(ex_img)
        ex_img_pil.save("example_image.png")

    # Build models
    logger.info("Building actor, critic, and v_net models")
    actor, critic, v_net = build_actor_critic_models(observation_space, action_space, cfg, device, remove_obs_keys)

    logger.info("Config:")
    rprint(OmegaConf.to_container(cfg))
    if wandb_logger:
        wandb_logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))

    # Define success bonus
    success_bonus_amount = cfg.env.success_bonus_amount
    if success_bonus_amount > 0:
        success_bonus_fn = SuccessBonusTransform(success_bonus_amount)
    else:
        success_bonus_fn = None

    save_dir = f"{output_dir}/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    return TrainingComponents(
        cfg=cfg,
        device=device,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        sentence_model=sentence_model,
        env=env,
        eval_env=eval_env,
        remove_obs_keys=remove_obs_keys,
        dino_image_keys=dino_image_keys,
        actor=actor,
        critic=critic,
        v_net=v_net,
        reward_model=reward_model,
        reward_model_exp_cfg=reward_model_exp_cfg,
        use_gt_rewards=use_gt_rewards,
        use_relative_rewards=use_relative_rewards,
        use_eval_server=use_eval_server,
        eval_server_url=eval_server_url,
        eval_server_timeout=reward_model_cfg.get("eval_server_timeout", 120.0)
        if reward_model_cfg is not None
        else 120.0,
        logger=logger,
        wandb_logger=wandb_logger,
        save_dir=save_dir,
        success_bonus_fn=success_bonus_fn,
    )
