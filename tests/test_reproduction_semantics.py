import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from robometer_policy_learning.algorithms.modeling_algorithm import BaseAlgorithm
from robometer_policy_learning.algorithms.sac.modeling_sac import (
    gradient_l2_norm,
    should_update_entropy_coefficient,
)
from robometer_policy_learning.envs.libero_pi0_wrapper import LiberoPI0Wrapper
from robometer_policy_learning.libero_pi05_dsrl.detection import (
    combine_environment_and_estimated_reward,
    normalized_terminal_progress_delta,
    success_window_detected,
    terminal_progress_contexts,
)
from robometer_policy_learning.libero_pi05_dsrl.evaluation_worker import (
    LiberoPi05EvaluationWorker,
)
from robometer_policy_learning.libero_pi05_dsrl.pi05_policy import (
    FrozenRLinfPi05Policy,
)
from robometer_policy_learning.libero_pi05_dsrl.policy_init import (
    initialize_residual_gaussian_actor,
)
from robometer_policy_learning.libero_pi05_dsrl.rollout_worker import (
    LiberoPi05RobometerRolloutWorker,
)
from robometer_policy_learning.libero_pi05_dsrl.training_semantics import (
    resolve_macro_gamma,
)


def test_pi05_macro_reward_matches_official_dsrl_last_step_semantics():
    reward = LiberoPi05RobometerRolloutWorker.official_dsrl_macro_reward(
        step_reward=-1.0,
        num_steps=5,
    )

    assert reward == -1.0
    assert LiberoPi05RobometerRolloutWorker.official_dsrl_macro_reward(
        step_reward=-1.0,
        num_steps=0,
    ) == 0.0


def test_macro_discount_is_not_exponentiated_by_action_exec_len():
    assert resolve_macro_gamma(
        configured_gamma=0.999,
        action_exec_len=5,
        discount_unit="macro",
    ) == pytest.approx(0.999)
    assert resolve_macro_gamma(
        configured_gamma=0.99,
        action_exec_len=5,
        discount_unit="low_level",
    ) == pytest.approx(0.99**5)


def test_entropy_coefficient_updates_on_every_actor_step_in_validated_control():
    assert [
        step
        for step in range(8)
        if should_update_entropy_coefficient(step, update_interval=1)
    ] == list(range(8))


def test_gradient_norm_is_observable_when_clipping_is_disabled():
    layer = torch.nn.Linear(3, 2)
    layer(torch.ones(4, 3)).sum().backward()
    before = [parameter.grad.detach().clone() for parameter in layer.parameters()]
    value = gradient_l2_norm(layer.parameters())
    assert value > 0
    for parameter, expected in zip(layer.parameters(), before, strict=True):
        torch.testing.assert_close(parameter.grad, expected)


def test_validated_sf73jk43_control_is_the_default_training_preset():
    repo_root = Path(__file__).resolve().parents[1]
    from robometer_policy_learning.reproducibility import load_experiment_config

    cfg = load_experiment_config(repo_root / "configs/reproduction/sf73jk43.yaml")

    assert cfg.dsrl.training_preset == "sf73jk43_legacy_control"
    assert cfg.dsrl.discount_unit == "low_level"
    assert cfg.online_algorithm.gamma == pytest.approx(0.99)
    assert resolve_macro_gamma(
        configured_gamma=cfg.online_algorithm.gamma,
        action_exec_len=cfg.dsrl.action_exec_len,
        discount_unit=cfg.dsrl.discount_unit,
    ) == pytest.approx(0.99**5)
    assert cfg.online_algorithm.num_updates_per_train_step == 1
    assert cfg.online_algorithm.num_critic_updates_per_actor_update == 1
    assert cfg.online_algorithm.ent_coef_update_interval == 1
    assert cfg.online_algorithm.actor_optimizer_lr == pytest.approx(1e-5)
    assert cfg.online_algorithm.critic_optimizer_lr == pytest.approx(1e-5)
    assert cfg.online_algorithm.actor_clip_grad_norm is None
    assert cfg.online_algorithm.critic_clip_grad_norm is None


def test_replay_reward_is_step_cost_plus_absolute_robometer_progress():
    assert combine_environment_and_estimated_reward(
        -1.0,
        0.6,
        add_estimated_reward=True,
    ) == pytest.approx(-0.4)
    assert combine_environment_and_estimated_reward(
        -1.0,
        0.6,
        add_estimated_reward=False,
    ) == pytest.approx(0.6)


class _CheckpointAlgorithm(BaseAlgorithm):
    def __init__(self):
        super().__init__(SimpleNamespace(logger=None))
        self.actor = torch.nn.Linear(2, 1)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.log_ent_coef = torch.tensor(0.25, requires_grad=True)
        self.component_names = ["actor", "actor_optimizer", "log_ent_coef"]
        self._n_updates = 17

    def train_step(self, batch=None, logging_prefix=None, rollout_step=None):
        del batch, logging_prefix, rollout_step


def test_checkpoint_load_preserves_optimizer_parameter_identity(tmp_path):
    source = _CheckpointAlgorithm()
    source.actor(torch.ones(1, 2)).sum().backward()
    source.actor_optimizer.step()
    source.save(tmp_path)

    restored = _CheckpointAlgorithm()
    actor_parameter = next(restored.actor.parameters())
    restored.load(tmp_path)

    assert next(restored.actor.parameters()) is actor_parameter
    assert restored.actor_optimizer.param_groups[0]["params"][0] is actor_parameter
    torch.testing.assert_close(
        next(restored.actor.parameters()), next(source.actor.parameters())
    )
    assert restored._n_updates == 17


def test_repeat_noise_matches_rlinf_shared_noise_semantics():
    noise = torch.arange(64, dtype=torch.float32).reshape(2, 32)

    repeated = FrozenRLinfPi05Policy.repeat_noise(noise, action_horizon=5)

    assert repeated.shape == (2, 5, 32)
    for horizon_index in range(5):
        torch.testing.assert_close(repeated[:, horizon_index], noise, rtol=0, atol=0)


def test_pi05_precision_null_matches_rlinf_fp32_contract():
    assert FrozenRLinfPi05Policy.normalize_precision(None) is None
    assert FrozenRLinfPi05Policy.normalize_precision("null") is None
    assert FrozenRLinfPi05Policy.normalize_precision("fp32") is None
    assert FrozenRLinfPi05Policy.normalize_precision("bf16") == "bf16"
    with pytest.raises(ValueError, match="precision"):
        FrozenRLinfPi05Policy.normalize_precision("fp16")


def test_pi05_direct_observation_matches_rlinf_embodied_schema():
    policy = object.__new__(FrozenRLinfPi05Policy)
    obs = {
        "observation/state": np.zeros(8, dtype=np.float32),
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    }

    formatted = policy.format_observation(obs, "pick up the bowl")

    assert formatted["states"].shape == (1, 8)
    assert formatted["main_images"].shape == (1, 224, 224, 3)
    assert formatted["wrist_images"].shape == (1, 224, 224, 3)
    assert formatted["extra_view_images"] is None
    assert formatted["task_descriptions"] == ["pick up the bowl"]


def test_residual_actor_output_initialization_matches_rlinf_scale():
    torch.manual_seed(7)
    actor = SimpleNamespace(
        mean_layer=torch.nn.Linear(128, 32),
        log_std_layer=torch.nn.Linear(128, 32),
    )

    initialize_residual_gaussian_actor(actor, output_gain=0.01)

    torch.testing.assert_close(actor.mean_layer.bias, torch.zeros(32))
    torch.testing.assert_close(actor.log_std_layer.bias, torch.zeros(32))
    assert actor.mean_layer.weight.std().item() < 0.002
    assert actor.log_std_layer.weight.std().item() < 0.002
    assert actor.mean_layer.weight.abs().sum().item() > 0


def test_residual_actor_allows_exact_zero_step0_control():
    actor = SimpleNamespace(
        mean_layer=torch.nn.Linear(128, 32),
        log_std_layer=torch.nn.Linear(128, 32),
    )

    initialize_residual_gaussian_actor(actor, output_gain=0.0)

    torch.testing.assert_close(actor.mean_layer.weight, torch.zeros_like(actor.mean_layer.weight))
    torch.testing.assert_close(actor.mean_layer.bias, torch.zeros_like(actor.mean_layer.bias))
    torch.testing.assert_close(
        actor.log_std_layer.weight, torch.zeros_like(actor.log_std_layer.weight)
    )
    torch.testing.assert_close(actor.log_std_layer.bias, torch.zeros_like(actor.log_std_layer.bias))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast requires CUDA")
def test_local_robometer_scoring_uses_configured_bf16_autocast():
    worker = object.__new__(LiberoPi05RobometerRolloutWorker)
    worker.buffer = SimpleNamespace(reward_model=object())
    worker.device = torch.device("cuda")
    worker.reward_autocast_dtype = "bf16"

    with worker._reward_forward_context():
        assert torch.is_autocast_enabled("cuda")
        assert torch.get_autocast_dtype("cuda") == torch.bfloat16


def test_remote_robometer_scoring_does_not_enable_local_autocast():
    worker = object.__new__(LiberoPi05RobometerRolloutWorker)
    worker.buffer = SimpleNamespace(reward_model=None)
    worker.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    worker.reward_autocast_dtype = "bf16"

    with worker._reward_forward_context():
        assert not torch.is_autocast_enabled("cuda")


def test_enhanced_success_detector_matches_validated_rlinf_rule():
    assert success_window_detected(
        [0.95, 0.96, 0.97], threshold=0.94, duration=3, rule="all_consecutive"
    )
    assert not success_window_detected(
        [0.95, 0.93, 0.97], threshold=0.94, duration=3, rule="all_consecutive"
    )
    assert not success_window_detected(
        [0.99, 0.99], threshold=0.94, duration=3, rule="all_consecutive"
    )


def test_terminal_delta_context_and_normalization_match_rlinf():
    frames = np.arange(11, dtype=np.uint8)[:, None, None, None]
    first, final = terminal_progress_contexts(frames, context_frames=4)

    assert first[:, 0, 0, 0].tolist() == [0, 0, 0, 0]
    assert final[:, 0, 0, 0].tolist() == [0, 3, 6, 10]
    assert normalized_terminal_progress_delta(
        0.2, 0.3, frame_count=11
    ) == pytest.approx(0.01)


class _FakeLiberoEnv(gym.Env):
    language_instruction = "pick up the bowl"
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
    observation_space = gym.spaces.Dict({})

    def reset(self):
        return {"raw": 0}, {}

    def step(self, action):
        return {"raw": 1}, 1.0, True, False, {}

    def close(self):
        return None


class _SettlingLiberoEnv(gym.Env):
    language_instruction = "pick up the bowl"
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
    observation_space = gym.spaces.Dict({})

    def __init__(self):
        self.actions = []
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return {"settle_index": 0}, {"reset": True}

    def step(self, action):
        self.current_step += 1
        self.actions.append(np.asarray(action).copy())
        return {"settle_index": self.current_step}, 0.0, False, False, {}


def test_libero_reset_matches_rlinf_fifteen_step_settling(monkeypatch):
    monkeypatch.setattr(
        "robometer_policy_learning.envs.libero_pi0_wrapper.preprocess_obs_for_pi0",
        lambda obs: dict(obs),
    )
    base = _SettlingLiberoEnv()
    wrapper = LiberoPI0Wrapper(base, reset_settle_steps=15)

    obs, info = wrapper.reset()

    assert obs["settle_index"] == 15
    assert info == {"reset": True}
    assert len(base.actions) == 15
    for action in base.actions:
        np.testing.assert_array_equal(action[:-1], np.zeros(6, dtype=np.float32))
        assert action[-1] == -1.0
    assert base.current_step == 0


class _RandomResetLiberoEnv(_SettlingLiberoEnv):
    def reset(self):
        self.current_step = 0
        return {"reset_value": int(np.random.randint(0, 1_000_000))}, {}

    def step(self, action):
        self.current_step += 1
        np.random.randint(0, 1_000_000)
        return {"settle_index": self.current_step}, 0.0, False, False, {}


def test_libero_procedural_reset_rng_is_isolated_per_environment(monkeypatch):
    monkeypatch.setattr(
        "robometer_policy_learning.envs.libero_pi0_wrapper.preprocess_obs_for_pi0",
        lambda obs: dict(obs),
    )
    env0 = LiberoPI0Wrapper(_RandomResetLiberoEnv(), procedural_seed=0)
    env1 = LiberoPI0Wrapper(_RandomResetLiberoEnv(), procedural_seed=1)
    expected0 = np.random.RandomState(0)
    expected1 = np.random.RandomState(1)

    np.random.seed(999)
    caller_state = np.random.get_state()
    values0 = []
    values1 = []
    for _ in range(3):
        values0.append(env0.reset()[0]["reset_value"])
        env0.step(np.zeros(7, dtype=np.float32))
        values1.append(env1.reset()[0]["reset_value"])
        env1.step(np.zeros(7, dtype=np.float32))

    expected_values0 = []
    expected_values1 = []
    for _ in range(3):
        expected_values0.append(int(expected0.randint(0, 1_000_000)))
        expected0.randint(0, 1_000_000)
        expected_values1.append(int(expected1.randint(0, 1_000_000)))
        expected1.randint(0, 1_000_000)
    assert values0 == expected_values0
    assert values1 == expected_values1
    actual_caller_state = np.random.get_state()
    assert actual_caller_state[0] == caller_state[0]
    np.testing.assert_array_equal(actual_caller_state[1], caller_state[1])
    assert actual_caller_state[2:] == caller_state[2:]


def test_training_wrapper_hides_simulator_success(monkeypatch):
    monkeypatch.setattr(
        "robometer_policy_learning.envs.libero_pi0_wrapper.preprocess_obs_for_pi0",
        lambda obs: dict(obs),
    )
    wrapper = LiberoPI0Wrapper(
        _FakeLiberoEnv(),
        ignore_environment_termination=True,
        ignore_environment_reward=True,
    )
    wrapper.reset()
    _, reward, done, truncated, info = wrapper.step(np.zeros(7, dtype=np.float32))

    assert reward == 0.0
    assert not done
    assert not truncated
    assert info["sim_success"]
    assert info["sim_success_once"]
    assert "success" not in info


class _TerminatedVectorEvalEnv:
    def __init__(self):
        self.steps = 0

    def reset(self):
        self.steps = 0
        return {"observation/state": np.zeros((1, 8), dtype=np.float32)}, {}

    def step(self, action):
        self.steps += 1
        obs = {"observation/state": np.zeros((1, 8), dtype=np.float32)}
        # Match Gymnasium auto-reset's easy-to-miss batched-info shape: no
        # top-level success key even though native LIBERO terminated.
        return obs, np.array([0.0]), np.array([True]), np.array([False]), {}


def test_libero_eval_counts_native_termination_as_success():
    worker = object.__new__(LiberoPi05EvaluationWorker)
    worker.eval_env = _TerminatedVectorEvalEnv()
    worker.action_exec_len = 1
    worker._decode = lambda actor, obs, deterministic: np.zeros((1, 1, 7), dtype=np.float32)

    metrics = worker._run_evaluations(actor=None, num_episodes=1)

    assert metrics["success_rate"] == 1.0
    assert metrics["unique_init_states"] == 0


def test_step0_actor_sampling_mode_controls_decode(monkeypatch):
    worker = object.__new__(LiberoPi05EvaluationWorker)
    worker.eval_env = _TerminatedVectorEvalEnv()
    worker.action_exec_len = 1
    observed = []
    worker._decode = lambda actor, obs, deterministic: (
        observed.append(deterministic)
        or np.zeros((1, 1, 7), dtype=np.float32)
    )
    monkeypatch.setenv("STEP0_ACTOR_DETERMINISTIC", "0")

    worker._run_evaluations(actor=None, num_episodes=1)

    assert observed == [False]


class _AutoResetVectorEvalEnv:
    def __init__(self):
        self.reset_count = 0

    def _reset_result(self):
        self.reset_count += 1
        fingerprint = self.reset_count
        obs = {
            "observation/state": np.full((1, 8), fingerprint, dtype=np.float32),
            "observation/image": np.full(
                (1, 4, 4, 3), fingerprint, dtype=np.uint8
            ),
        }
        infos = {
            "initial_observation_fingerprint": np.array([fingerprint]),
            "_initial_observation_fingerprint": np.array([True]),
        }
        return obs, infos

    def reset(self):
        return self._reset_result()

    def step(self, action):
        final_obs = {
            "observation/state": np.zeros((8,), dtype=np.float32),
            "observation/image": np.zeros((4, 4, 3), dtype=np.uint8),
        }
        obs, infos = self._reset_result()
        infos.update(
            {
                "final_observation": np.array([final_obs], dtype=object),
                "_final_observation": np.array([True]),
                "final_info": np.array([{}], dtype=object),
                "_final_info": np.array([True]),
            }
        )
        return obs, np.array([0.0]), np.array([False]), np.array([True]), infos


def test_libero_eval_reuses_vector_autoreset_without_skipping_initial_states(
    monkeypatch, tmp_path
):
    worker = object.__new__(LiberoPi05EvaluationWorker)
    worker.eval_env = _AutoResetVectorEvalEnv()
    worker.action_exec_len = 1
    worker._decode = lambda actor, obs, deterministic: np.zeros(
        (1, 1, 7), dtype=np.float32
    )
    monkeypatch.setenv("RUN_ROOT", str(tmp_path))

    metrics = worker._run_evaluations(actor=None, num_episodes=3)

    rows = [
        json.loads(line)
        for line in (tmp_path / "step0_eval_episodes.jsonl").read_text().splitlines()
    ]
    assert [row["initial_observation_fingerprint"] for row in rows] == [1, 2, 3]
    assert metrics["unique_init_states"] == 3


def test_step0_trajectory_capture_uses_terminal_observation(monkeypatch, tmp_path):
    worker = object.__new__(LiberoPi05EvaluationWorker)
    worker.eval_env = _AutoResetVectorEvalEnv()
    worker.action_exec_len = 1
    worker._decode = lambda actor, obs, deterministic: np.zeros(
        (1, 1, 7), dtype=np.float32
    )
    worker._instruction = lambda: "test instruction"
    monkeypatch.setenv("RUN_ROOT", str(tmp_path))
    monkeypatch.setenv("SAVE_STEP0_TRAJECTORIES", "1")

    worker._run_evaluations(actor=None, num_episodes=1)

    paths = list((tmp_path / "step0_trajectories").glob("*.pkl"))
    assert len(paths) == 1
    with paths[0].open("rb") as handle:
        episode = pickle.load(handle)
    assert len(episode["observations"]) == len(episode["actions"]) + 1 == 2
    assert len(episode["infos"]) == len(episode["observations"])
    assert episode["observations"][0]["main_images"].mean() == 1
    # The returned vector observation is already the next reset (value 2), but
    # the saved final frame must be Gymnasium's pre-autoreset observation (0).
    assert episode["observations"][-1]["main_images"].mean() == 0
