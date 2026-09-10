"""Exercise real checkpoint loader functions without importing the GPU training stack."""

import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file


@pytest.fixture
def loaders():
    source = Path(__file__).resolve().parents[1] / "vendor/robometer/robometer/utils/setup_utils.py"
    tree = ast.parse(source.read_text())
    names = {
        "_get_checkpoint_safetensors_files",
        "_load_checkpoint_weights_from_safetensors",
        "_load_custom_heads_from_safetensors",
        "_require_checkpoint_heads",
    }
    # Compile the production function bodies unchanged; only optional training imports are omitted.
    selected = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        + [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
        type_ignores=[],
    )
    namespace = {
        "Path": Path,
        "torch": torch,
        "load_file": load_file,
        "logger": logging.getLogger(__name__),
    }
    namespace["json"] = json
    exec(compile(ast.fix_missing_locations(selected), str(source), "exec"), namespace)  # noqa: S102 - trusted local source
    return namespace


def head_model():
    model = torch.nn.Module()
    model.progress_head = torch.nn.Sequential(torch.nn.Linear(2, 1))
    model.success_head = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    return model


@pytest.mark.parametrize("sidecar", [False, True])
@pytest.mark.parametrize("missing", ["all", "success_head.1.bias"])
def test_missing_success_head_is_fatal_before_loading(loaders, tmp_path, sidecar, missing):
    model = head_model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = {key: value + 1 for key, value in before.items()}
    state = {
        key: value
        for key, value in state.items()
        if not (key.startswith("success_head.") if missing == "all" else key == missing)
    }
    save_file(state, tmp_path / ("custom_heads.safetensors" if sidecar else "model.safetensors"))
    with pytest.raises(ValueError, match="success_head"):
        if sidecar:
            loaders["_load_custom_heads_from_safetensors"](model, str(tmp_path))
        else:
            loaders["_load_checkpoint_weights_from_safetensors"](
                model, str(tmp_path), SimpleNamespace(use_peft=False)
            )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


@pytest.mark.parametrize("sidecar", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
def test_complete_heads_load_even_when_values_are_unchanged(loaders, tmp_path, sidecar, offset):
    model = head_model()
    state = {key: value + offset for key, value in model.state_dict().items()}
    save_file(state, tmp_path / ("custom_heads.safetensors" if sidecar else "model.safetensors"))
    if sidecar:
        assert loaders["_load_custom_heads_from_safetensors"](model, str(tmp_path))
    else:
        loaders["_load_checkpoint_weights_from_safetensors"](
            model, str(tmp_path), SimpleNamespace(use_peft=False)
        )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key])
