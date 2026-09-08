"""Frozen RLinf Pi0.5 action decoder used by LIBERO MLP DSRL.

The SAC actor owns a 32-dimensional squashed-Gaussian action. Following the
RLinf DSRL implementation, that single noise vector is repeated over the
Pi0.5 action horizon before the frozen flow policy decodes environment actions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf

from robometer_policy_learning.libero_pi05_dsrl.policy_init import (
    initialize_residual_gaussian_actor,
)

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


class FrozenRLinfPi05Policy:
    """Small adapter around RLinf's PyTorch Pi0.5 evaluation model."""

    @staticmethod
    def normalize_precision(precision: str | None) -> str | None:
        normalized = (
            None
            if precision is None or str(precision).lower() in {"none", "null", "fp32"}
            else str(precision).lower()
        )
        if normalized not in {None, "bf16"}:
            raise ValueError(
                "Pi0.5 precision must be null/fp32 or bf16, got "
                f"{precision!r}"
            )
        return normalized

    def __init__(
        self,
        *,
        checkpoint: str,
        device: torch.device,
        action_chunk: int = 5,
        action_dim: int = 7,
        noise_dim: int = 32,
        num_steps: int = 3,
        precision: str | None = "bf16",
        config_name: str = "pi05_libero",
        backend: str = "validated_openpi",
        noise_source: str = "external",
        internal_actor_seed: int = 42,
        force_zero_actor: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = torch.device(device)
        self.action_chunk = int(action_chunk)
        self.action_dim = int(action_dim)
        self.noise_dim = int(noise_dim)
        self.num_steps = int(num_steps)
        self.backend = str(backend)
        self.noise_source = str(noise_source)
        self.internal_actor_seed = int(internal_actor_seed)
        self.force_zero_actor = bool(force_zero_actor)
        if self.backend != "validated_openpi":
            raise ValueError(
                "Formal LIBERO DSRL only supports backend=validated_openpi; "
                "the openpi_rlinf compatibility port has not reproduced the "
                "validated task-4 baseline"
            )
        if self.noise_source not in {"external", "internal_rlinf"}:
            raise ValueError(
                "noise_source must be external or internal_rlinf, got "
                f"{self.noise_source!r}"
            )

        if not (self.checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"Pi0.5 checkpoint is incomplete: {self.checkpoint}")
        if self.action_chunk <= 0 or self.num_steps <= 0:
            raise ValueError("action_chunk and num_steps must be positive")
        # These packages are installed from the repository's vendored source by
        # scripts/setup.sh. Never splice another virtualenv into sys.path: doing
        # so can mix incompatible torch, torchvision, and transformers builds.
        import torchvision  # noqa: F401
        import transformers  # noqa: F401
        from transformers import AutoProcessor  # noqa: F401

        from openpi.models import model as openpi_model
        from rlinf.models.embodiment.openpi import get_model

        model_cfg = OmegaConf.create(
            {
                # Match examples/embodiment/config/model/pi0_5.yaml and the
                # previously validated RLinf task-4 DSRL run. The SAC actor is
                # external here, so use_dsrl remains false and infer() injects
                # its noise directly into the same OpenPI sampler.
                "model_type": "openpi",
                "model_path": str(self.checkpoint),
                "precision": precision,
                "is_lora": False,
                "lora_rank": 32,
                "num_action_chunks": self.action_chunk,
                "action_dim": self.action_dim,
                "use_proprio": True,
                "num_steps": self.num_steps,
                "add_value_head": False,
                "add_q_head": self.noise_source == "internal_rlinf",
                "q_head_type": "default",
                "openpi_data": {
                    "norm_stats_path": str(
                        self.checkpoint / "physical-intelligence" / "libero" / "norm_stats.json"
                    )
                },
                "openpi": {
                    "config_name": config_name,
                    "num_images_in_input": 2,
                    "noise_level": 0.5,
                    "action_chunk": self.action_chunk,
                    "num_steps": self.num_steps,
                    "action_env_dim": self.action_dim,
                    "noise_method": "reinflow",
                    "train_expert_only": True,
                    "add_value_head": False,
                    "value_after_vlm": False,
                    "value_vlm_mode": "mean_token",
                    "detach_critic_input": self.noise_source == "internal_rlinf",
                    "use_dsrl": self.noise_source == "internal_rlinf",
                    "dsrl_state_dim": 8,
                    "dsrl_action_noise_dim": self.noise_dim,
                    "dsrl_num_q_heads": 10,
                    "dsrl_image_latent_dim": 64,
                    "dsrl_state_latent_dim": 64,
                    "dsrl_hidden_dims": [128, 128, 128],
                },
            }
        )
        normalized_precision = self.normalize_precision(precision)
        model_cfg.precision = normalized_precision
        dtype = torch.bfloat16 if normalized_precision == "bf16" else None
        if self.noise_source == "internal_rlinf":
            # Match the validated RLinf ActorGroup construction seed. This path
            # is an eval-only differential; formal policy-learning trains the
            # external SAC actor instead.
            torch.manual_seed(self.internal_actor_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.internal_actor_seed)
        self.model = get_model(model_cfg, torch_dtype=dtype).to(self.device).eval()
        if self.noise_source == "internal_rlinf" and self.force_zero_actor:
            initialize_residual_gaussian_actor(
                self.model.dsrl_action_noise_net, output_gain=0.0
            )
        self.model.requires_grad_(False)
        self.model_action_horizon = int(self.model.config.action_horizon)
        if int(self.model.config.action_chunk) != self.action_chunk:
            raise RuntimeError(
                "RLinf Pi0.5 action-chunk override was not applied: "
                f"requested {self.action_chunk}, model has {self.model.config.action_chunk}"
            )
        if self.model_action_horizon < self.action_chunk:
            raise RuntimeError(
                "Pi0.5 model action horizon cannot be shorter than the returned chunk: "
                f"model={self.model_action_horizon}, chunk={self.action_chunk}"
            )
        self._observation_class = openpi_model.Observation

    @staticmethod
    def repeat_noise(noise: torch.Tensor, action_horizon: int) -> torch.Tensor:
        """Apply RLinf GaussianPolicy's shared-noise horizon expansion."""
        if noise.ndim == 3:
            if noise.shape[1] != action_horizon:
                raise ValueError(
                    f"Noise horizon {noise.shape[1]} does not match Pi0.5 horizon {action_horizon}"
                )
            return noise
        if noise.ndim != 2:
            raise ValueError(f"Expected noise [B,D] or [B,H,D], got {tuple(noise.shape)}")
        return noise.unsqueeze(1).repeat(1, action_horizon, 1)

    @staticmethod
    def _ensure_batched(value: Any, *, image: bool = False) -> Any:
        if torch.is_tensor(value):
            if (image and value.ndim == 3) or (not image and value.ndim == 1):
                return value.unsqueeze(0)
            return value
        array = np.asarray(value)
        if (image and array.ndim == 3) or (not image and array.ndim == 1):
            array = array[None]
        return array

    @staticmethod
    def _first_present(obs: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in obs:
                return obs[key]
        raise KeyError(f"None of the required observation keys are present: {keys}")

    def format_observation(self, obs: Mapping[str, Any], instruction: str | list[str]) -> dict[str, Any]:
        state = self._first_present(obs, ("observation/state", "state", "states"))
        main_image = self._first_present(
            obs, ("observation/image", "image", "main_images")
        )
        wrist_image = self._first_present(
            obs, ("observation/wrist_image", "wrist_image", "wrist_images")
        )

        state = self._ensure_batched(state)
        main_image = self._ensure_batched(main_image, image=True)
        wrist_image = self._ensure_batched(wrist_image, image=True)
        batch_size = int(state.shape[0])
        if isinstance(instruction, str):
            instructions = [instruction] * batch_size
        else:
            instructions = list(instruction)
        if len(instructions) != batch_size:
            raise ValueError(
                f"Instruction batch ({len(instructions)}) does not match observation batch ({batch_size})"
            )

        return {
            "states": state,
            "main_images": main_image,
            "wrist_images": wrist_image,
            # RLinf's distributed embodied-data schema always materializes this
            # optional key. Direct policy invocation bypasses that schema, but
            # the validated OpenPI obs_processor still expects the key.
            "extra_view_images": None,
            "task_descriptions": instructions,
        }

    @torch.inference_mode()
    def infer(
        self,
        observations: Mapping[str, Any],
        noise: np.ndarray | torch.Tensor,
        instruction: str | list[str],
    ) -> np.ndarray:
        env_obs = self.format_observation(observations, instruction)
        if self.noise_source == "internal_rlinf":
            env_obs = {
                key: (
                    torch.as_tensor(value)
                    if value is not None and key != "task_descriptions"
                    else value
                )
                for key, value in env_obs.items()
            }
            actions, _ = self.model.predict_action_batch(
                env_obs,
                mode="eval",
                compute_values=False,
            )
            return actions.detach().float().cpu().numpy()

        noise_tensor = torch.as_tensor(noise, device=self.device, dtype=torch.float32)
        if noise_tensor.ndim == 1:
            noise_tensor = noise_tensor.unsqueeze(0)
        if noise_tensor.shape[-1] != self.noise_dim:
            raise ValueError(
                f"SAC noise dim {noise_tensor.shape[-1]} does not match configured {self.noise_dim}"
            )
        # The pi05_libero checkpoint denoises 10 latent action slots even when
        # RLinf returns/executes only the first 5. Its native DSRL policy repeats
        # the same 32-D SAC sample across model.config.action_horizon.
        repeated = self.repeat_noise(noise_tensor, self.model_action_horizon)
        to_process_obs = self.model.obs_processor(env_obs)
        processed_obs = self.model.input_transform(to_process_obs, transpose=False)
        processed_obs = self.model.precision_processor(processed_obs)
        observation = self._observation_class.from_dict(processed_obs)
        outputs = self.model.sample_actions(
            observation,
            noise=repeated,
            mode="eval",
            compute_values=False,
        )
        actions = self.model.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"]
        actions_np = actions.detach().float().cpu().numpy()
        if actions_np.shape != (noise_tensor.shape[0], self.action_chunk, self.action_dim):
            raise RuntimeError(
                "Unexpected Pi0.5 action shape: "
                f"got {actions_np.shape}, expected "
                f"({noise_tensor.shape[0]}, {self.action_chunk}, {self.action_dim})"
            )
        return actions_np

    def close(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
