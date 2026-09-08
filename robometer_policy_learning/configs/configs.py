from dataclasses import dataclass, field
from typing import Optional, Any, List


@dataclass
class EnvironmentConfig:
    """Configuration for environment settings."""

    env_name: str = field(default="Meta-World/MT1/button-press-wall-v3", metadata={"help": "Environment name to use"})
    use_full_state: bool = field(
        default=False,
        metadata={
            "help": "Use the full state (default: False). If False, only proprioceptive information will be used. If True, no reward relabeling will be performed."
        },
    )
    use_gt_rewards: bool = field(
        default=False,
        metadata={
            "help": "Use ground truth rewards (default: False). If True, no reward relabeling will be performed. If no reward model is provided, this will be set to True."
        },
    )
    h5_dataset_path: str = field(
        default="/scr/shared/reward_fm/policy_training_datasets/metaworld_generation_converted.h5",
        metadata={"help": "Path to the H5 dataset"},
    )
    # LIBERO / DSRL env parameters (optional for other envs)
    task_id: int = field(default=0, metadata={"help": "Task ID within suite (e.g., LIBERO)"})
    max_episode_steps: int = field(default=400, metadata={"help": "Max episode steps (e.g., LIBERO / remote robot)"})
    use_vlm_features: bool = field(
        default=True, metadata={"help": "Whether to use VLM features (DSRL-specific)"}
    )
    image_keys: Optional[List[str]] = field(
        default=None, metadata={"help": "Observation image keys used for DINO feature extraction (primary field)"}
    )
    dino_image_keys: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Observation image keys used for DINO feature extraction (deprecated: use image_keys)"},
    )
    extra_keys_to_drop: List[str] = field(
        default_factory=list, metadata={"help": "Extra observation keys to drop beyond image_keys"}
    )
    # Async reward relabeling at environment level (before transitions go to buffer)
    use_async_reward_relabel: bool = field(
        default=False, metadata={"help": "Whether to use async reward relabeling at environment level"}
    )
    reward_relabel_batch_size: int = field(
        default=32, metadata={"help": "Batch size for async reward relabeling at environment level"}
    )
    use_placeholder_rewards: bool = field(
        default=True, metadata={"help": "Whether to use placeholder rewards initially and update retroactively"}
    )
    # Remote robot specific
    obs_format: Optional[str] = field(
        default=None, metadata={"help": "Remote robot observation format (if applicable)"}
    )
    num_stages: int = field(
        default=1, metadata={"help": "Number of stages for RemoteEnv (default: 1)"}
    )
    success_bonus_amount: float = field(
        default=0.0, metadata={"help": "Success bonus amount"}
    )

    def __post_init__(self):
        """Sync image_keys and dino_image_keys for backward compatibility."""
        # Default value if neither is set
        default_image_keys = ["image"]

        # If both are None, set both to default
        if self.image_keys is None and self.dino_image_keys is None:
            self.image_keys = default_image_keys
            self.dino_image_keys = default_image_keys
        # If image_keys is set but dino_image_keys is not, sync them
        elif self.image_keys is not None and self.dino_image_keys is None:
            self.dino_image_keys = self.image_keys
        # If dino_image_keys is set but image_keys is not, sync them
        elif self.dino_image_keys is not None and self.image_keys is None:
            self.image_keys = self.dino_image_keys


@dataclass
class TrainingConfig:
    """Configuration for training settings."""

    num_envs: int = field(default=1, metadata={"help": "Number of parallel environments"})
    num_rollouts: int = field(
        default=100_000, metadata={"help": "Number of rollouts to train, this is more like env steps"}
    )
    chunk_size: Optional[int] = field(default=None, metadata={"help": "Action chunking size (None for no chunking)"})
    use_rnn: bool = field(default=False, metadata={"help": "Use RNN actor and critic"})
    seed: int = field(default=0, metadata={"help": "Random seed for training and environment initialization"})

    num_offline_steps: int = field(default=100_000, metadata={"help": "Number of offline training steps"})

    load_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to load RL models from (null for no loading)"}
    )
    save_interval: int = field(default=10000, metadata={"help": "Save interval in steps"})
    continue_training: bool = field(default=False, metadata={"help": "Continue training from the last checkpoint"})
    train_after_episode: bool = field(default=False, metadata={"help": "Train only after episode completion"})


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""

    wandb_project: str = field(default="rfm_rl", metadata={"help": "WandB project name"})
    wandb_entity: Optional[str] = field(default="clvr", metadata={"help": "WandB entity name"})
    wandb_offline: bool = field(default=False, metadata={"help": "Run WandB in offline mode"})
    wandb_log_dir_base: str = field(default="logs/wandb", metadata={"help": "Base directory for WandB logs"})
    wandb_name: str = field(default="train_rl", metadata={"help": "WandB run name"})
    wandb_prefix: Optional[str] = field(
        default="offline", metadata={"help": "Prefix for WandB metric names (e.g., 'offline', 'online', 'learner')"}
    )
    log_level: str = field(
        default="INFO",
        metadata={"help": "Logging level (TRACE, DEBUG2, DEBUG, INFO, WARNING, ERROR, CRITICAL)"},
    )


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""

    eval_freq: int = field(default=10000, metadata={"help": "Evaluation frequency in rollouts"})
    eval_num_episodes: int = field(default=25, metadata={"help": "Number of episodes for evaluation"})
    eval_record_video: bool = field(default=True, metadata={"help": "Whether to record videos during evaluation"})
    eval_on_first_step: bool = field(default=False, metadata={"help": "Evaluate on the first step before any training"})
    eval_at_episode_boundary: bool = field(
        default=False, metadata={"help": "Only evaluate at episode boundaries (useful for remote robot)"}
    )


@dataclass
class ModelConfig:
    """Configuration for model settings."""

    dinov2_model: Optional[str] = field(
        default="facebook/dinov2-base", metadata={"help": "DINOv2 model for video embeddings (None to disable)"}
    )
    sentence_model: Optional[str] = field(
        default="sentence-transformers/all-MiniLM-L12-v2",
        metadata={"help": "Sentence transformer model for text embeddings (null/None to disable)"},
    )
    image_encoder_type: Optional[str] = field(
        default=None,
        metadata={"help": "Image encoder type: 'impala', 'resnet', 'dinov2', or 'flatten'. None means use default."},
    )
    # IMPALA encoder parameters (used when image_encoder_type == "impala")
    impala_nn_scale: int = field(default=1, metadata={"help": "Scaling factor for IMPALA encoder channel sizes"})
    impala_num_blocks_per_stack: int = field(
        default=2, metadata={"help": "Number of residual blocks per stack in IMPALA encoder"}
    )
    impala_use_smaller: bool = field(
        default=False, metadata={"help": "Whether to use SmallerImpalaEncoder variant (fewer blocks)"}
    )
    impala_output_dim: Optional[int] = field(
        default=None, metadata={"help": "Output dimension for IMPALA encoder (None uses default based on architecture)"}
    )


@dataclass
class AlgorithmConfig:
    """Configuration for algorithm settings."""

    offline_alg_name: str = field(default="iql", metadata={"help": "Offline algorithm name (iql, bc, sac)"})
    online_alg_name: str = field(default="sac", metadata={"help": "Online algorithm name (sac, iql, bc)"})


@dataclass
class MLPConfig:
    """Configuration for MLP architecture."""

    hidden_dims: tuple = field(default=(256, 256), metadata={"help": "Hidden layer dimensions"})
    activation: str = field(default="relu", metadata={"help": "Activation function (relu, gelu, tanh, etc.)"})
    use_layer_norm: bool = field(default=False, metadata={"help": "Whether to use layer normalization"})
    dropout_rate: float = field(default=0.0, metadata={"help": "Dropout rate (0.0 means no dropout)"})


@dataclass
class RNNConfig:
    """Configuration for RNN architecture."""

    rnn_type: str = field(default="LSTM", metadata={"help": "RNN type (LSTM, GRU, RNN)"})
    rnn_hidden_size: int = field(default=256, metadata={"help": "RNN hidden size"})
    rnn_num_layers: int = field(default=1, metadata={"help": "Number of RNN layers"})
    rnn_dropout: float = field(default=0.0, metadata={"help": "RNN dropout rate"})
    rnn_bidirectional: bool = field(default=False, metadata={"help": "Whether to use bidirectional RNN"})
    feature_hidden_dims: Optional[list] = field(
        default=None, metadata={"help": "MLP hidden dims before RNN (None means no MLP)"}
    )
    output_hidden_dims: Optional[list] = field(
        default=None, metadata={"help": "MLP hidden dims after RNN (None means no MLP)"}
    )
    activation: str = field(default="relu", metadata={"help": "Activation function for MLP layers"})
    use_layer_norm: bool = field(default=False, metadata={"help": "Whether to use layer normalization"})
    dropout_rate: float = field(default=0.0, metadata={"help": "Dropout rate for MLP layers"})


@dataclass
class TransformerConfig:
    """Configuration for Transformer architecture."""

    d_model: int = field(default=128, metadata={"help": "Transformer model dimension"})
    nhead: int = field(default=8, metadata={"help": "Number of attention heads"})
    num_encoder_layers: int = field(default=1, metadata={"help": "Number of transformer encoder layers"})
    transformer_dropout: float = field(default=0.1, metadata={"help": "Transformer dropout"})
    transformer_activation: str = field(default="gelu", metadata={"help": "Transformer activation (relu, gelu)"})
    feature_hidden_dims: Optional[list] = field(
        default=None, metadata={"help": "MLP hidden dims before transformer (None means direct input->transformer)"}
    )
    output_hidden_dims: Optional[list] = field(
        default=None, metadata={"help": "MLP hidden dims after transformer (None means direct transformer->output)"}
    )
    pooling_strategy: str = field(
        default="first",
        metadata={"help": "Pooling strategy for transformer output (mean, first, attention, weighted_mean)"},
    )
    use_layer_norm: bool = field(default=True, metadata={"help": "Whether to use layer normalization"})
    activation: str = field(default="relu", metadata={"help": "Activation function for MLP layers"})
    dropout_rate: float = field(default=0.0, metadata={"help": "Dropout rate for MLP layers"})


@dataclass
class PolicyConfig:
    """Configuration for policy (actor) network."""

    # MLP configuration (used when chunk_size is None)
    mlp: MLPConfig = field(
        default_factory=lambda: MLPConfig(hidden_dims=(512, 512), activation="relu", use_layer_norm=False)
    )

    # RNN configuration (used when use_rnn=True and chunk_size is not None)
    rnn: RNNConfig = field(default_factory=lambda: RNNConfig(feature_hidden_dims=[512, 512]))

    # Transformer configuration (used when use_rnn=False and chunk_size is not None)
    transformer: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(
            d_model=128,
            feature_hidden_dims=[768, 512, 256],
            transformer_dropout=0.1,
            transformer_activation="gelu",
            use_layer_norm=False,
        )
    )

    # Policy-specific parameters
    use_tanh_output: bool = field(default=True, metadata={"help": "Whether to use tanh activation on output (for MLP)"})
    deterministic: bool = field(default=False, metadata={"help": "Whether this is a deterministic policy"})
    log_std_init: float = field(default=-0.5, metadata={"help": "Initial log std for stochastic policies"})
    log_std_min: float = field(default=-20.0, metadata={"help": "Minimum log std"})
    log_std_max: float = field(default=2.0, metadata={"help": "Maximum log std"})

    def __post_init__(self):
        """Convert dict sub-configs to proper dataclass instances."""
        if isinstance(self.mlp, dict):
            self.mlp = MLPConfig(**self.mlp)
        if isinstance(self.rnn, dict):
            self.rnn = RNNConfig(**self.rnn)
        if isinstance(self.transformer, dict):
            self.transformer = TransformerConfig(**self.transformer)


@dataclass
class ValueFunctionConfig:
    """Configuration for value function (critic and v_net) networks."""

    mlp: MLPConfig = field(
        default_factory=lambda: MLPConfig(hidden_dims=(768, 512, 256), activation="relu", use_layer_norm=True)
    )

    rnn: RNNConfig = field(default_factory=lambda: RNNConfig(feature_hidden_dims=[768, 256], activation="relu"))

    transformer: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(
            d_model=128,
            feature_hidden_dims=[768, 512, 256],
            transformer_dropout=0.1,
            transformer_activation="gelu",
            use_layer_norm=True,
            pooling_strategy="first",
        )
    )

    def __post_init__(self):
        """Convert dict sub-configs to proper dataclass instances."""
        if isinstance(self.mlp, dict):
            self.mlp = MLPConfig(**self.mlp)
        if isinstance(self.rnn, dict):
            self.rnn = RNNConfig(**self.rnn)
        if isinstance(self.transformer, dict):
            self.transformer = TransformerConfig(**self.transformer)


@dataclass
class RewardModelConfig:
    """
    Configuration for reward relabeling.
    """

    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Type of model (robometer or roboreward, etc)"
        },
     )
    # Model loading (only used by server script, not training scripts)
    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to reward model checkpoint (HuggingFace model ID or local path). Only used by reward relabeling server."
        },
    )

    # Reward relabeling settings
    use_relative_rewards: bool = field(
        default=False, metadata={"help": "Whether to use relative rewards (delta from previous step)"}
    )
    add_estimated_reward: bool = field(
        default=False, metadata={"help": "Whether to add estimated reward to the ground truth reward (useful in sparse reward settings)"}
    )
    reward_relabeling_keys: List[str] = field(
        default_factory=lambda: ["image"],
        metadata={"help": "Observation keys to use for reward relabeling (e.g., ['image'] for images)"},
    )

    use_success_detection: bool = field(
        default=False, metadata={"help": "Whether to use success detection"}
    )
    success_detection_duration: int = field(
        default=2,
        metadata={"help": "Number of consecutive steps with high success probability to detect success"},
    )
    success_detection_threshold: float = field(
        default=0.65, metadata={"help": "Success probability threshold for success detection"}
    )
    success_detection_mode: str = field(
        default="standard",
        metadata={"help": "Success detector: standard or enhanced_terminal_delta"},
    )
    terminal_adjacent_delta_detection: bool = field(
        default=False,
        metadata={"help": "Use terminal progress-delta fallback after a complete episode"},
    )
    terminal_adjacent_delta_threshold: float = field(
        default=float("inf"),
        metadata={"help": "Threshold for terminal adjacent progress-delta success fallback"},
    )
    wait_for_completion_on_episode_end: bool = field(
        default=False,
        metadata={"help": "Wait for outstanding reward labels before finalizing an episode"},
    )

    use_async_reward_relabel: bool = field(
        default=False, metadata={"help": "Whether to use async reward relabeling at buffer level"}
    )
    async_reward_relabel_server_address: str = field(
        default="localhost:50052", metadata={"help": "Address for remote reward relabeling server"}
    )


@dataclass
class ReplayBufferConfig:
    """Configuration for replay buffer settings."""

    capacity: int = field(default=1_000_000, metadata={"help": "Replay buffer capacity"})
    sample_ratio: float = field(
        default=0.5,
        metadata={
            "help": "Sample ratio of offline buffer in mixed replay buffer (used only when we do offline-to-online RL)"
        },
    )
    use_success_fail_buffer: bool = field(
        default=False, metadata={"help": "If True, create SuccessFailureReplayBuffer"}
    )
    success_fail_sample_ratio: float = field(
        default=0.5, metadata={"help": "Ratio for sampling from success vs failure buffer (default 0.5)"}
    )
    min_relabeled_ratio: float = field(
        default=0.1, metadata={"help": "Minimum ratio of relabeled transitions required before sampling"}
    )
    save_buffer_on_exit: bool = field(
        default=False, metadata={"help": "Whether to save buffer to npz file on training exit or kill (remote robot mode)"}
    )
    save_buffer_every: int = field(
        default=0, metadata={"help": "Save buffer every N env steps (0 to disable periodic saving)"}
    )
    save_buffer_images: bool = field(
        default=False, metadata={"help": "Whether to save raw image frames when saving buffer (can be large)"}
    )
    save_buffer_image_keys: List[str] = field(
        default_factory=lambda: ["image"], 
        metadata={"help": "List of image keys to save (if save_buffer_images=True)"}
    )
    save_buffer_dir: str = field(
        default="replay_buffers", metadata={"help": "Directory to save buffer files"}
    )


@dataclass
class TrainConfig:
    """Main training configuration that contains all sub-configs."""

    debug: bool = field(default=False, metadata={"help": "Enable debug mode"})

    # Sub-configs
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvaluationConfig = field(default_factory=EvaluationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    alg: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
    buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    value_function: ValueFunctionConfig = field(default_factory=ValueFunctionConfig)

    def __post_init__(self):
        """Convert dict sub-configs to proper dataclass instances."""
        if isinstance(self.env, dict):
            self.env = EnvironmentConfig(**self.env)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)
        if isinstance(self.eval, dict):
            self.eval = EvaluationConfig(**self.eval)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.alg, dict):
            self.alg = AlgorithmConfig(**self.alg)
        if isinstance(self.buffer, dict):
            self.buffer = ReplayBufferConfig(**self.buffer)
        if isinstance(self.reward_model, dict):
            self.reward_model = RewardModelConfig(**self.reward_model)
        if isinstance(self.policy, dict):
            self.policy = PolicyConfig(**self.policy)
        if isinstance(self.value_function, dict):
            self.value_function = ValueFunctionConfig(**self.value_function)


@dataclass
class DSRLSectionConfig:
    """Configuration for DSRL-specific parameters (matches `dsrl:` section in dsrl_config_new.yaml)."""

    noise_dim: int = field(default=32, metadata={"help": "Dimension of noise to predict"})
    noise_action_bound: float = field(default=1.0, metadata={"help": "Action bound for DSRL noise"})
    pi0_checkpoint: Optional[str] = field(default=None, metadata={"help": "Path to Pi0 checkpoint (REQUIRED)"})
    pi0_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Explicit OpenPI training config used to restore the checkpoint"},
    )
    action_exec_len: int = field(default=20, metadata={"help": "How many actions to execute from chunk"})
    use_vlm_features: bool = field(default=True, metadata={"help": "Whether to use VLM features in DSRL env wrapper"})


@dataclass
class DistributedLearnerServerConfig:
    host: str = field(default="0.0.0.0", metadata={"help": "Learner server bind host"})
    port: int = field(default=50051, metadata={"help": "Learner server port"})
    address: str = field(default="127.0.0.1:50051", metadata={"help": "Full learner address for clients (host:port)"})
    max_msg_mb: int = field(default=256, metadata={"help": "Max gRPC message size in MB"})


@dataclass
class DistributedRolloutConfig:
    flush_every: int = field(default=32, metadata={"help": "Flush buffer every N transitions"})
    max_message_mb: int = field(default=64, metadata={"help": "Max message size in MB for streaming"})
    refresh_secs: float = field(default=5.0, metadata={"help": "How often to refresh actor weights (seconds)"})
    target_hz: float = field(default=0.0, metadata={"help": "Target rollout frequency in Hz (0.0 = no limit)"})
    num_rollouts: int = field(default=100, metadata={"help": "Number of rollouts per cycle"})


@dataclass
class DistributedEvalConfig:
    eval_every: float = field(default=120.0, metadata={"help": "Evaluation frequency in seconds"})


@dataclass
class RemoteRobotConfig:
    """Remote robot server configuration."""
    host: str = field(default="localhost", metadata={"help": "Remote robot server host"})
    port: int = field(default=6000, metadata={"help": "Remote robot server port"})
    connect_timeout: float = field(default=300.0, metadata={"help": "Connection timeout in seconds"})


@dataclass
class SimplerConfig:
    """SimplerEnv server configuration."""
    host: str = field(default="0.0.0.0", metadata={"help": "SimplerEnv server host"})
    port: int = field(default=6000, metadata={"help": "SimplerEnv server port"})
    use_dense_reward: bool = field(
        default=False, metadata={"help": "If true, use dense rewards based on ManiSkill info dict"}
    )

@dataclass
class DistributedConfig:
    """Distributed/async training configuration (matches `distributed:` section in dsrl_config_distributed.yaml)."""

    learner_server: DistributedLearnerServerConfig = field(default_factory=DistributedLearnerServerConfig)
    train_target_hz: float = field(default=0.0, metadata={"help": "Target training frequency in Hz (0.0 = no limit)"})
    log_interval: float = field(default=5.0, metadata={"help": "Logging interval in seconds"})
    save_interval: Optional[int] = field(
        default=None, metadata={"help": "Save interval in steps (None = use training.save_interval)"}
    )
    actor_update_freq: int = field(
        default=50, metadata={"help": "Frequency of actor updates served to workers (steps)"}
    )
    rollout: DistributedRolloutConfig = field(default_factory=DistributedRolloutConfig)
    eval: DistributedEvalConfig = field(default_factory=DistributedEvalConfig)

    def __post_init__(self):
        if isinstance(self.learner_server, dict):
            self.learner_server = DistributedLearnerServerConfig(**self.learner_server)
        if isinstance(self.rollout, dict):
            self.rollout = DistributedRolloutConfig(**self.rollout)
        if isinstance(self.eval, dict):
            self.eval = DistributedEvalConfig(**self.eval)


@dataclass
class DSRLConfig:
    """
    Structured config for `scripts/train_dsrl.py` (matches `configs/dsrl_config_new.yaml`).

    Note: We keep a root-level `wandb_name` alias for older overrides (e.g. dsrl_config_distributed.yaml)
    and forward it into `logging.wandb_name` in __post_init__.
    """

    # Hydra defaults will populate this group; type kept flexible on purpose.
    online_algorithm: Any = field(default_factory=dict)

    # Top-level controls
    debug: bool = field(default=False, metadata={"help": "Enable debug mode"})
    mode: str = field(default="serial", metadata={"help": "Mode: serial, learner, rollout, eval"})
    wandb_name: Optional[str] = field(default=None, metadata={"help": "DEPRECATED: use logging.wandb_name"})

    # Sub-configs (hierarchical)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvaluationConfig = field(default_factory=EvaluationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    alg: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    dsrl: DSRLSectionConfig = field(default_factory=DSRLSectionConfig)
    reward_model: Optional[RewardModelConfig] = field(default=None)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    value_function: ValueFunctionConfig = field(default_factory=ValueFunctionConfig)

    # Optional async/distributed config
    distributed: Optional[DistributedConfig] = field(default=None)
    
    # Optional remote robot config (for REMOTE_ROBOT env)
    remote_robot: Optional[RemoteRobotConfig] = field(default=None)
    eval_remote_robot: Optional[RemoteRobotConfig] = field(default=None)
    
    # Optional SimplerEnv config (for SIMPLER env)
    simpler: Optional[SimplerConfig] = field(default=None)

    def __post_init__(self):
        # Convert dict sub-configs to proper dataclass instances.
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.buffer, dict):
            self.buffer = ReplayBufferConfig(**self.buffer)
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)
        if isinstance(self.eval, dict):
            self.eval = EvaluationConfig(**self.eval)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.env, dict):
            self.env = EnvironmentConfig(**self.env)
        if isinstance(self.alg, dict):
            self.alg = AlgorithmConfig(**self.alg)
        if isinstance(self.dsrl, dict):
            self.dsrl = DSRLSectionConfig(**self.dsrl)
        if isinstance(self.policy, dict):
            self.policy = PolicyConfig(**self.policy)
        if isinstance(self.value_function, dict):
            self.value_function = ValueFunctionConfig(**self.value_function)
        if isinstance(self.reward_model, dict):
            self.reward_model = RewardModelConfig(**self.reward_model)
        if isinstance(self.distributed, dict):
            self.distributed = DistributedConfig(**self.distributed)
        if isinstance(self.remote_robot, dict):
            self.remote_robot = RemoteRobotConfig(**self.remote_robot)
        if isinstance(self.eval_remote_robot, dict):
            self.eval_remote_robot = RemoteRobotConfig(**self.eval_remote_robot)
        if isinstance(self.simpler, dict):
            self.simpler = SimplerConfig(**self.simpler)

        # Backward-compatible alias (used by existing dsrl_config_distributed.yaml in this repo)
        if self.wandb_name:
            # Only override if user didn't explicitly set a different value in logging
            if not getattr(self.logging, "wandb_name", None) or self.logging.wandb_name == "train_rl":
                self.logging.wandb_name = self.wandb_name
