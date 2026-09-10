import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from robometer_policy_learning.algorithms.modeling_algorithm import BaseAlgorithm
from robometer_policy_learning.libero_pi05_dsrl.evaluation_worker import (
    LiberoPi05EvaluationWorker,
)
from robometer_policy_learning.runners.serial_runner import SerialRunner


class HorizonEnv(gym.Env):
    observation_space = gym.spaces.Dict({
        "observation/image": gym.spaces.Box(0, 255, (2, 2, 3), dtype=np.uint8),
    })
    action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)

    def __init__(self):
        self.resets = 0
        self.actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        self.steps = 0
        return self.observation(self.resets), {
            "initial_observation_fingerprint": self.resets,
        }

    @staticmethod
    def observation(value):
        return {"observation/image": np.full((2, 2, 3), value, dtype=np.uint8)}

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        return self.observation(250), -1.0, False, self.steps == 240, {
            "sim_success_once": self.resets % 2 == 1 and self.steps >= 7,
        }


def evaluation_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("RUN_ROOT", str(tmp_path))
    monkeypatch.delenv("SAVE_STEP0_TRAJECTORIES", raising=False)
    monkeypatch.delenv("STEP0_ACTOR_DETERMINISTIC", raising=False)
    env = gym.vector.SyncVectorEnv([HorizonEnv])
    worker = LiberoPi05EvaluationWorker(
        pi05_policy=None, action_exec_len=5, eval_env=env, device="cpu",
        num_episodes=2, record_video=False,
    )
    worker._decode = lambda actor, obs, deterministic: np.full(
        (1, 5, 7), 0.125, dtype=np.float32
    )
    return worker


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_evaluations_keep_per_step_evidence_and_full_horizon(monkeypatch, tmp_path):
    worker = evaluation_worker(monkeypatch, tmp_path)
    actor = torch.nn.Linear(1, 1)
    runner = object.__new__(SerialRunner)
    runner.evaluation_worker = worker
    runner.eval_kwargs = {}
    runner.total_env_steps = 0
    first_metrics = runner.evaluate(actor)
    initial_path = tmp_path / "step0_eval_episodes.jsonl"
    initial_bytes = initial_path.read_bytes()
    runner.total_env_steps = 25000
    later_metrics = runner.evaluate(actor)

    assert initial_path.read_bytes() == initial_bytes
    later_path = tmp_path / "step25000_eval_episodes.jsonl"
    assert later_path.is_file()
    first_rows, later_rows = read_rows(initial_path), read_rows(later_path)
    assert [row["evaluation_step"] for row in first_rows] == [0, 0]
    assert [row["evaluation_step"] for row in later_rows] == [25000, 25000]
    assert [row["initial_observation_fingerprint"] for row in first_rows] == [1, 2]
    # Preserve the existing reset at the start of each evaluation call, too.
    assert [row["initial_observation_fingerprint"] for row in later_rows] == [4, 5]
    assert worker.eval_env.envs[0].resets == 6
    assert len(worker.eval_env.envs[0].actions) == 4 * 240
    expected_hash = hashlib.sha256(
        np.full((240, 7), 0.125, dtype=np.float32).tobytes()
    ).hexdigest()
    for row in first_rows + later_rows:
        assert row["action_sha256"] == expected_hash
        assert row["steps"] == 240
        assert row["reward"] == -240.0
    assert first_metrics == later_metrics == {
        "avg_reward": -240.0, "std_reward": 0.0, "min_reward": -240.0,
        "max_reward": -240.0, "avg_steps": 240.0, "success_rate": 0.5,
        "num_eval_episodes": 2, "unique_init_states": 2,
    }
    assert actor.training


class CheckpointAlgorithm(BaseAlgorithm):
    def __init__(self):
        super().__init__(SimpleNamespace(logger=None))
        self.actor = torch.nn.Linear(2, 1)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.log_ent_coef = torch.tensor(0.25, requires_grad=True)
        self.component_names = ["actor", "actor_optimizer", "log_ent_coef"]
        self._n_updates = 17

    def train_step(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("mode", [None, "resume", "auto", "weights_only"])
def test_loading_requires_explicit_warm_start_before_deserialization(tmp_path, mode):
    calls = []
    algorithm = SimpleNamespace(load=lambda path: calls.append(path))
    with pytest.raises(ValueError, match="warm_start"):
        SerialRunner.load_checkpoint(algorithm, tmp_path, load_mode=mode)
    assert calls == []


def test_warm_start_loads_algorithm_state_without_replacing_parameters(tmp_path):
    source = CheckpointAlgorithm()
    source.step_counter = 19
    source.actor(torch.ones(1, 2)).sum().backward()
    source.actor_optimizer.step()
    source.save(tmp_path)
    restored = CheckpointAlgorithm()
    actor_parameter = next(restored.actor.parameters())
    SerialRunner.load_checkpoint(restored, tmp_path, load_mode="warm_start")
    assert next(restored.actor.parameters()) is actor_parameter
    assert restored.actor_optimizer.param_groups[0]["params"][0] is actor_parameter
    torch.testing.assert_close(actor_parameter, next(source.actor.parameters()))
    assert restored.actor_optimizer.state[actor_parameter]["step"].item() == 1
    assert restored._n_updates == 17
    assert restored.step_counter == 19


def test_checkpoint_persists_step_evidence_not_a_resume_claim(tmp_path):
    runner = object.__new__(SerialRunner)
    runner.save_dir = str(tmp_path)
    runner.algorithm = CheckpointAlgorithm()
    runner.logger = None
    runner.total_env_steps = 175000
    runner.total_episodes_completed = 700
    runner._save_checkpoint("final", metadata={"rollouts_completed": 35000})
    evidence = json.loads((tmp_path / "final/checkpoint_manifest.json").read_text())
    assert evidence["env_steps"] == 175000
    assert evidence["episodes"] == 700
    assert evidence["resume_supported"] is False
    assert evidence["checkpoint_type"] == "algorithm_warm_start"
    assert evidence["n_updates"] == 17
    assert evidence["metadata"]["rollouts_completed"] == 35000


def test_repeated_evaluation_at_same_step_preserves_evidence(monkeypatch, tmp_path):
    worker = evaluation_worker(monkeypatch, tmp_path)
    worker.evaluation_step = 25000
    actor = torch.nn.Linear(1, 1)
    worker.run(actor)
    path = tmp_path / "step25000_eval_episodes.jsonl"
    original = path.read_bytes()
    worker.run(actor)
    worker.run(actor)
    assert path.read_bytes() == original
    for repeat in (1, 2):
        rows = read_rows(tmp_path / f"step25000_eval_episodes_repeat{repeat}.jsonl")
        assert len(rows) == 2
        assert all(row["evaluation_step"] == 25000 for row in rows)


@pytest.mark.parametrize("mode", ["warm_start", "evaluate"])
def test_incomplete_checkpoint_is_rejected_before_loading(tmp_path, mode):
    source = CheckpointAlgorithm()
    source.save(tmp_path)
    (tmp_path / "actor_optimizer.pt").unlink()
    restored = CheckpointAlgorithm()
    before = next(restored.actor.parameters()).detach().clone()
    with pytest.raises(FileNotFoundError, match="actor_optimizer.pt"):
        SerialRunner.load_checkpoint(restored, tmp_path, load_mode=mode)
    torch.testing.assert_close(next(restored.actor.parameters()), before)


def checkpoint_config(tmp_path, *, mode=None, steps=0):
    config = OmegaConf.create({
        "training": {"load_dir": str(tmp_path), "num_rollouts": steps},
        "eval": {"eval_on_first_step": True, "eval_num_episodes": 2},
    })
    if mode is not None:
        config.training.load_mode = mode
    return config


@pytest.mark.parametrize("mode,steps", [(None, 10), ("resume", 0), ("evaluate", 10)])
def test_train_rejects_invalid_load_intent_before_setup(monkeypatch, tmp_path, mode, steps):
    from scripts import train

    calls = []
    monkeypatch.setattr(train, "setup_training", lambda cfg: calls.append(cfg))
    with pytest.raises(ValueError, match="warm_start"):
        train.run(checkpoint_config(tmp_path, mode=mode, steps=steps))
    assert calls == []


def test_fresh_config_validation_does_not_change_recipe_hash():
    from scripts.train import _checkpoint_options
    from robometer_policy_learning.reproducibility import (
        load_experiment_config, scientific_recipe_hash,
    )

    config = load_experiment_config(
        Path(__file__).resolve().parents[1] / "configs/reproduction/sf73jk43.yaml"
    )
    before = OmegaConf.to_container(config)
    recipe_hash = scientific_recipe_hash(config)
    assert _checkpoint_options(config) == (None, None)
    assert OmegaConf.to_container(config) == before
    assert scientific_recipe_hash(config) == recipe_hash
    assert "load_mode" not in config.training


def test_evaluation_step_uses_manifest_not_algorithm_counter(tmp_path):
    from scripts.train import _checkpoint_options

    (tmp_path / "training_state.json").write_text(json.dumps({"step": 19}))
    config = checkpoint_config(tmp_path, mode="evaluate")
    with pytest.raises(ValueError, match="no environment-step evidence"):
        _checkpoint_options(config)
    (tmp_path / "checkpoint_manifest.json").write_text(json.dumps({"env_steps": 175000}))
    assert _checkpoint_options(config) == ("evaluate", 175000)
    config.eval.evaluation_step = 42
    with pytest.raises(ValueError, match="conflicts"):
        _checkpoint_options(config)


def test_legacy_checkpoint_step_requires_explicit_value_unless_numeric_tag(tmp_path):
    assert SerialRunner.checkpoint_evaluation_step(tmp_path / "175000") == 175000
    assert SerialRunner.checkpoint_evaluation_step(tmp_path / "final", 175000) == 175000
    with pytest.raises(ValueError, match="explicitly"):
        SerialRunner.checkpoint_evaluation_step(tmp_path / "final")


@pytest.mark.parametrize("step", [-1, True, 1.5, "175000"])
def test_invalid_evaluation_step_is_rejected(tmp_path, step):
    with pytest.raises(ValueError, match="nonnegative integer"):
        SerialRunner.checkpoint_evaluation_step(tmp_path, step)


def test_evaluate_mode_requires_an_actual_evaluation(tmp_path):
    from scripts.train import _checkpoint_options

    config = checkpoint_config(tmp_path, mode="evaluate")
    config.eval.eval_on_first_step = False
    with pytest.raises(ValueError, match="eval_on_first_step"):
        _checkpoint_options(config)
    config.eval.eval_on_first_step = True
    config.eval.eval_num_episodes = 0
    with pytest.raises(ValueError, match="eval_num_episodes"):
        _checkpoint_options(config)


def test_training_cannot_override_evidence_step(tmp_path):
    from scripts.train import _checkpoint_options

    config = checkpoint_config(tmp_path, mode="warm_start", steps=100)
    config.eval.evaluation_step = 175000
    with pytest.raises(ValueError, match="only allowed"):
        _checkpoint_options(config)


def test_evaluation_cli_explicitly_selects_evaluate_mode(monkeypatch, tmp_path):
    from scripts import evaluate

    monkeypatch.setenv("PYTHON_BIN", sys.executable)
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", "--config", "config.yaml", "--checkpoint", str(tmp_path),
        "--episodes", "20", "--evaluation-step", "175000",
    ])
    calls = []
    monkeypatch.setattr(evaluate.subprocess, "run", lambda command, **kwargs: (
        calls.append(command) or SimpleNamespace(returncode=7)
    ))
    with pytest.raises(SystemExit) as result:
        evaluate.main()
    assert result.value.code == 7
    assert "training.load_mode=evaluate" in calls[0]
    assert "training.num_rollouts=0" in calls[0]
    assert "eval.eval_on_first_step=true" in calls[0]
    assert "eval.evaluation_step=175000" in calls[0]
    assert "eval.eval_num_episodes=20" in calls[0]


def test_warm_start_restarts_collection_warmup_and_scheduling(tmp_path):
    source = CheckpointAlgorithm()
    source.save(tmp_path)
    algorithm = CheckpointAlgorithm()
    SerialRunner.load_checkpoint(algorithm, tmp_path, load_mode="warm_start")
    algorithm.config.learning_starts = 5
    training_calls = []
    algorithm.train_step = lambda **kwargs: training_calls.append(kwargs)
    collection_calls = []
    rollout = SimpleNamespace(device="cpu", num_rollouts=1, total_episodes=0)

    def collect(*, can_train):
        collection_calls.append(can_train)
        return {"total_steps": len(collection_calls) * 5}

    rollout.run = collect
    rollout.update_actor = lambda actor: None
    eval_steps = []

    class StepRecorder:
        def __init__(self, **kwargs):
            self.evaluation_step = 0

        def run(self, actor):
            eval_steps.append(self.evaluation_step)
            return {}

    runner = SerialRunner(
        env=object(), eval_env=object(), algorithm=algorithm, actor=algorithm.actor,
        buffer=[], rollout_worker=rollout, num_rollouts=15, eval_freq=5,
        eval_on_first_step=True, evaluation_worker_class=StepRecorder,
        save_dir=str(tmp_path / "new_run"), save_interval=10,
    )
    assert runner.total_env_steps == runner.total_episodes_completed == 0
    assert runner.previous_total_episodes == runner._last_buffer_save_steps == 0
    assert runner.buffer == []
    runner.run()
    assert collection_calls == [False, True, True]
    assert len(training_calls) == 2
    assert eval_steps == [0, 5, 10, 15]
    assert runner.total_env_steps == 15
    assert {path.name for path in (tmp_path / "new_run").iterdir()} == {"10", "final"}
