import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from robometer_policy_learning import runtime_metadata
from scripts import run_experiment


@pytest.fixture
def fake_torch(monkeypatch):
    torch = SimpleNamespace(
        __version__="2.9.0+cu128",
        version=SimpleNamespace(cuda="12.8", hip=None, git_version="torch-commit"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 91002)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda index: "Test GPU",
            get_device_capability=lambda index: (8, 9),
            get_device_properties=lambda index: SimpleNamespace(total_memory=24_000_000_000),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_probe_records_current_interpreter_versions_and_visible_gpu(monkeypatch, fake_torch):
    monkeypatch.setattr(sys, "executable", "/reward-env/bin/python")
    monkeypatch.setattr(sys, "prefix", "/reward-env")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("HF_TOKEN", "secret-hf-value")
    monkeypatch.setenv("WANDB_API_KEY", "secret-wandb-value")
    monkeypatch.setenv("PIP_INDEX_URL", "https://user:secret-pip-value@example.com")
    distributions = [
        SimpleNamespace(metadata={"Name": "torch"}, version="2.9.0"),
        SimpleNamespace(metadata={"Name": "transformers"}, version="4.57.1"),
    ]
    monkeypatch.setattr(runtime_metadata.importlib.metadata, "distributions", lambda: distributions)

    manifest = runtime_metadata.collect_runtime_metadata()

    assert manifest["python"]["executable"] == "/reward-env/bin/python"
    assert manifest["python"]["prefix"] == "/reward-env"
    assert manifest["packages"] == {"torch": "2.9.0", "transformers": "4.57.1"}
    assert manifest["torch"]["version"] == "2.9.0+cu128"
    assert manifest["torch"]["cuda_version"] == "12.8"
    assert manifest["torch"]["cudnn_version"] == 91002
    assert manifest["torch"]["devices"][0]["capability"] == [8, 9]
    assert manifest["environment"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert "secret-" not in json.dumps(manifest)
    assert "PIP_INDEX_URL" not in manifest["environment"]


def test_probe_reports_cuda_failure_without_exception_secrets(monkeypatch, fake_torch):
    def fail():
        raise RuntimeError("secret exception content")

    monkeypatch.setattr(fake_torch.cuda, "is_available", fail)
    manifest = runtime_metadata.collect_runtime_metadata()
    assert manifest["torch"]["status"] == "partial"
    assert manifest["torch"]["error_type"] == "RuntimeError"
    assert "secret exception content" not in json.dumps(manifest)


def test_probe_reports_missing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    manifest = runtime_metadata.collect_runtime_metadata()
    assert manifest["torch"]["status"] == "unavailable"
    assert manifest["torch"]["error_type"] == "ModuleNotFoundError"


def test_snapshot_is_create_only_and_redacts_nested_credentials(tmp_path):
    path = tmp_path / "model_info.json"
    payload = {
        "experiment_config": {
            "hub_token": "secret-hf-value",
            "use_per_frame_progress_token": True,
            "max_new_tokens": 128,
            "nested": [{"api_key": "secret-api-value", "password": "secret-password"}],
        },
        "model_architecture": {"model_class": "Qwen3VL"},
    }
    runtime_metadata.write_manifest(path, payload)
    original = path.read_bytes()

    assert b"secret-" not in original
    archived = json.loads(original)
    assert archived["experiment_config"]["hub_token"] == "[REDACTED]"
    assert archived["experiment_config"]["use_per_frame_progress_token"] is True
    assert archived["experiment_config"]["max_new_tokens"] == 128
    assert payload["experiment_config"]["hub_token"] == "secret-hf-value"
    with pytest.raises(FileExistsError):
        runtime_metadata.write_manifest(path, {"replacement": True})
    assert path.read_bytes() == original


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    args = SimpleNamespace(
        config=tmp_path / "recipe.yaml",
        set=["training.seed=17"],
        no_start_server=False,
        preflight_only=False,
    )
    cfg = OmegaConf.create({
        "runtime": {"output_root": str(tmp_path / "outputs")},
        "resources": {
            "policy_gpu": 0,
            "reward_server": {
                "host": "127.0.0.1", "port": 8000, "gpu": 3,
                "max_workers": 1, "batch_size": 4,
                "autocast_dtype": "bf16", "use_unsloth": False,
            },
        },
    })
    state = SimpleNamespace(
        args=args, cfg=cfg, calls=[], policy_envs=[], servers=[], stopped=[],
        info={"model_architecture": {"model_class": "Qwen3VL"},
              "experiment_config": {"hub_token": "do-not-archive"}},
        probe_failure=False,
    )
    policy_python = tmp_path / "policy-python"
    reward_python = tmp_path / "reward-python"
    policy_python.touch()
    reward_python.touch()
    monkeypatch.setenv("PYTHON_BIN", str(policy_python))
    monkeypatch.setenv("REWARD_PYTHON_BIN", str(reward_python))
    monkeypatch.setenv("ROBOMETER_BASE_MODEL", "/models/Qwen3-VL")
    monkeypatch.setenv("ROBOMETER_MODEL", "/models/reward")
    monkeypatch.setenv("REWARD_RUNTIME_MANIFEST", "/stale/runtime.json")
    monkeypatch.setenv("ROBOMETER_MODEL_INFO_PATH", "/stale/model_info.json")
    monkeypatch.setenv("ROBOMETER_SERVER_LOG_PATH", "/stale/server.log")
    monkeypatch.setattr(run_experiment, "parse_args", lambda: args)
    monkeypatch.setattr(run_experiment, "load_experiment_config", lambda path: cfg)
    monkeypatch.setattr(run_experiment, "stop_process", state.stopped.append)
    monkeypatch.setattr(run_experiment, "wait_for_server", lambda *args: state.info)
    monkeypatch.setattr(run_experiment.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: state.info,
    ))

    def run(command, **kwargs):
        state.calls.append((command, kwargs))
        if command[1].endswith("runtime_metadata.py"):
            if state.probe_failure:
                raise subprocess.CalledProcessError(1, command)
            runtime_metadata.write_manifest(Path(command[-1]), {
                "schema_version": 1, "source": "reward_interpreter_probe", "status": "captured",
            })
        if command[1].endswith("train.py"):
            env = kwargs["env"]
            assert Path(env["REWARD_RUNTIME_MANIFEST"]).is_file()
            assert Path(env["ROBOMETER_MODEL_INFO_PATH"]).is_file()
            state.policy_envs.append(env)
            return SimpleNamespace(returncode=7)
        return SimpleNamespace(returncode=0)

    def popen(command, **kwargs):
        server = SimpleNamespace(command=command, kwargs=kwargs)
        state.servers.append(server)
        return server

    monkeypatch.setattr(run_experiment.subprocess, "run", run)
    monkeypatch.setattr(run_experiment.subprocess, "Popen", popen)
    return state


def launch():
    with pytest.raises(SystemExit) as exc:
        run_experiment.main()
    assert exc.value.code == 7


def test_launcher_hands_off_unique_reward_evidence_and_preserves_commands(launcher):
    launch()
    first_env = launcher.policy_envs[0]
    first_info = Path(first_env["ROBOMETER_MODEL_INFO_PATH"])
    original = first_info.read_bytes()
    launcher.info["launch"] = 2
    launch()
    second_env = launcher.policy_envs[1]

    assert first_info.read_bytes() == original
    assert b"do-not-archive" not in original
    for name in ("REWARD_RUNTIME_MANIFEST", "ROBOMETER_MODEL_INFO_PATH", "ROBOMETER_SERVER_LOG_PATH"):
        assert first_env[name] != second_env[name]
        assert Path(first_env[name]).is_absolute()
        assert Path(first_env[name]).is_file()
        assert Path(first_env[name]).parent == first_info.parent
    assert first_env["CUDA_VISIBLE_DEVICES"] == "0"
    probe_command, probe_kwargs = launcher.calls[1]
    server = launcher.servers[0]
    assert probe_command[0] == server.command[0]
    assert probe_command[0].endswith("reward-python")
    assert probe_kwargs["env"] == server.kwargs["env"]
    assert probe_kwargs["cwd"] == server.kwargs["cwd"]
    assert probe_kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert probe_kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert probe_kwargs["check"] is True
    assert server.command[1:] == [
        "-m", "robometer.evals.eval_server", "model_path=/models/reward", "num_gpus=1",
        "max_workers=1", "batch_size=4", "server_url=127.0.0.1", "server_port=8000",
        "autocast_dtype=bf16", "use_unsloth=false",
    ]
    train_command, _ = launcher.calls[2]
    assert train_command[0].endswith("policy-python")
    assert train_command[2:] == ["--config", str(launcher.args.config), "--set", "training.seed=17"]
    assert launcher.stopped == launcher.servers
    assert all(server.kwargs["stdout"].closed for server in launcher.servers)


def test_external_server_does_not_claim_local_runtime_or_inherit_stale_log(launcher, monkeypatch):
    launcher.args.no_start_server = True
    monkeypatch.setenv("REWARD_PYTHON_BIN", "/missing/reward-python")
    launch()
    env = launcher.policy_envs[0]
    manifest = json.loads(Path(env["REWARD_RUNTIME_MANIFEST"]).read_text())
    assert manifest["status"] == "unavailable"
    assert manifest["source"] == "external_server"
    assert "python" not in manifest
    assert "ROBOMETER_SERVER_LOG_PATH" not in env
    assert not launcher.servers
    assert len(launcher.calls) == 2  # preflight and training, never a local reward probe


def test_probe_failure_prevents_server_and_policy_start(launcher):
    launcher.probe_failure = True
    with pytest.raises(subprocess.CalledProcessError):
        run_experiment.main()
    assert not launcher.servers
    assert not launcher.policy_envs
    assert launcher.calls[1][1]["stdout"].closed


def test_preflight_only_creates_no_service_artifacts(launcher):
    launcher.args.preflight_only = True
    run_experiment.main()
    assert len(launcher.calls) == 1
    assert not Path(launcher.cfg.runtime.output_root).exists()
