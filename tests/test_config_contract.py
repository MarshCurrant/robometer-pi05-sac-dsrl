from pathlib import Path

from omegaconf import OmegaConf

from robometer_policy_learning.loggers.wandb_logger import WandbLogger
from robometer_policy_learning.reproducibility import (
    load_experiment_config,
    scientific_recipe_hash,
)

ROOT = Path(__file__).resolve().parents[1]


def load_reference():
    cfg = load_experiment_config(ROOT / "configs/reproduction/sf73jk43.yaml")
    OmegaConf.resolve(cfg)
    return cfg


def test_reference_recipe_freezes_validated_scientific_contract():
    cfg = load_reference()
    assert cfg.env.env_name == "libero_spatial"
    assert cfg.env.task_id == 4
    assert cfg.training.num_rollouts == 500_000
    assert cfg.online_algorithm.learning_starts == 25_000
    assert cfg.online_algorithm.batch_size == 128
    assert cfg.online_algorithm.num_critics == 5
    assert cfg.online_algorithm.n_critics_to_sample == 2
    assert cfg.dsrl.noise_dim == 32
    assert cfg.dsrl.action_exec_len == 5
    assert cfg.reward_model.max_frames == 8
    assert cfg.reward_model.success_detection_threshold == 0.925
    assert cfg.reward_model.terminal_adjacent_delta_threshold == 0.0013611419747273127


def test_wandb_allowlist_is_small_unique_and_contains_outcome_metrics():
    metrics = list(load_reference().logging.metric_allowlist)
    assert 20 <= len(metrics) <= 30
    assert len(metrics) == len(set(metrics))
    assert "eval/success_rate" in metrics
    assert "buffer/avg_total_reward" in metrics
    assert "online/policy/actor_loss" in metrics


def test_recipe_hash_ignores_paths_and_logging_but_not_learning_rate():
    cfg = load_reference()
    baseline = scientific_recipe_hash(cfg)
    cfg.runtime.output_root = "/different/machine"
    cfg.logging.wandb_entity = "someone-else"
    cfg.dsrl.pi05_checkpoint = "/different/machine/pi05"
    cfg.model.dinov2_model = "/different/machine/dino"
    cfg.resources.policy_gpu = 7
    cfg.resources.reward_server.gpu = 6
    cfg.resources.reward_server.host = "reward.internal"
    cfg.resources.reward_server.port = 19000
    assert scientific_recipe_hash(cfg) == baseline
    cfg.online_algorithm.actor_optimizer_lr = 2e-5
    assert scientific_recipe_hash(cfg) != baseline


def test_task_extension_inherits_reference_without_hidden_hydra_defaults():
    cfg = load_experiment_config(ROOT / "configs/tasks/libero_template.yaml")
    assert cfg.online_algorithm.batch_size == 128
    assert cfg.env.task_id == 0
    assert cfg.reward_model.success_detection_threshold is None


def test_wandb_allowlist_uses_one_shared_step_axis():
    class FakeRun:
        def __init__(self):
            self.rows = []

        def log(self, row, **kwargs):
            self.rows.append((row, kwargs))

    logger = object.__new__(WandbLogger)
    logger.prefix = None
    logger._metric_allowlist = {"train/loss"}
    logger.logger = FakeRun()

    logger.log_dict(
        {"loss": 1.25, "not_allowlisted": 99.0},
        step=17,
        prefix="train",
    )

    assert logger.logger.rows == [({"train/loss": 1.25, "env_step": 17}, {})]
