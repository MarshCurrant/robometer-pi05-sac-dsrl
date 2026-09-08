# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rlinf.config import SupportedModel
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay import (
    PreloadReplayBufferDataset,
    ReplayBufferDataset,
    TrajectoryReplayBuffer,
    replay_buffer_collate_fn,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.scheduler import Channel, Worker
from rlinf.utils import drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory, collect_param_names_need_sync
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


def _resize_dsrl_replay_images(
    images: torch.Tensor, device: torch.device | str
) -> torch.Tensor:
    """Apply the exact DSRL image resize once before replay persistence.

    DSRL consumes only the main camera at 64x64.  Persisting the original 256x256
    main/wrist images (and duplicate forward inputs) made each 8-env trajectory
    roughly 436 MiB.  The returned NCHW float32 tensor is already in [0, 1]; the
    model's existing preprocessing then performs an identity-size interpolate
    before mapping it to [-1, 1].
    """
    if images.dim() != 5:
        raise ValueError(
            f"Expected replay images with [T, B, H, W, C] or [T, B, C, H, W], got {tuple(images.shape)}"
        )
    time_steps, batch_size = images.shape[:2]
    if images.shape[-1] == 3:
        flat = images.reshape(-1, *images.shape[2:]).permute(0, 3, 1, 2)
    elif images.shape[2] == 3:
        flat = images.reshape(-1, *images.shape[2:])
    else:
        raise ValueError(f"Cannot identify RGB channel in shape {tuple(images.shape)}")

    flat = flat.to(device=device, non_blocking=True)
    if flat.dtype == torch.uint8:
        flat = flat.float().div_(255.0)
    else:
        flat = flat.float()
        if flat.min() < 0:
            flat = (flat + 1.0) / 2.0
    flat = F.interpolate(
        flat.clamp_(0.0, 1.0),
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    )
    return flat.reshape(time_steps, batch_size, 3, 64, 64).cpu().contiguous()


def _compact_dsrl_replay_trajectory(
    trajectory: Trajectory, device: torch.device | str
) -> None:
    """Retain exactly the fields consumed by the official DSRL SAC learner."""
    for obs_name in ("curr_obs", "next_obs"):
        obs = getattr(trajectory, obs_name)
        if not obs or obs.get("main_images") is None or obs.get("states") is None:
            raise ValueError(f"DSRL replay trajectory is missing {obs_name} main image/state")
        setattr(
            trajectory,
            obs_name,
            {
                "main_images": _resize_dsrl_replay_images(
                    obs["main_images"], device=device
                ),
                "states": obs["states"].cpu().contiguous(),
            },
        )

    # These fields are rollout/debug payloads and are never read by
    # EmbodiedSACFSDPPolicy.forward_{actor,critic,alpha}.
    trajectory.intervene_flags = None
    trajectory.truncations = None
    trajectory.dones = None
    trajectory.prev_logprobs = None
    trajectory.prev_values = None
    trajectory.versions = None
    trajectory.forward_inputs = {}


def _select_dsrl_transition_reward(
    rewards: torch.Tensor, *, aggregation: str
) -> torch.Tensor:
    """Map low-level chunk rewards to the official DSRL transition reward."""
    if rewards.ndim != 2 or rewards.shape[1] == 0:
        raise ValueError(
            f"Expected DSRL rewards shaped [batch, action_horizon], got {rewards.shape}"
        )
    if aggregation == "last":
        # robometer-policy-learning/rollouts/dsrl_rollout_worker.py stores the
        # reward from the last executed low-level action as the chunk reward.
        return rewards[:, -1:]
    if aggregation == "sum":
        return rewards.sum(dim=-1, keepdim=True)
    raise ValueError(f"Unsupported DSRL reward aggregation: {aggregation!r}")


class EmbodiedSACFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # SAC-specific initialization
        self.replay_buffer = None
        self.target_model = None
        self.entropy_temp = None
        self.demo_buffer = None
        self.alpha_optimizer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))

    def init_worker(self):
        self.setup_model_and_optimizer(initialize_target=True)
        self.setup_sac_components()
        self.soft_update_target_model(tau=1.0)
        if self.use_dsrl:
            self._init_target_shadow()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(
                self.model, mode="default"
            )  # max-autotune-no-cudagraphs
            self.target_model = torch.compile(self.target_model, mode="default")

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        """Setup model, lr_scheduler, optimizer and grad_scaler."""
        """Add initializing target model logic."""
        module = self.model_provider_func()
        if initialize_target:
            target_module = self.model_provider_func()

        # Enable gradient checkpointing if configured
        if self.cfg.actor.model.get("gradient_checkpointing", False):
            self.logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
            if initialize_target:
                target_module.gradient_checkpointing_enable()
        else:
            self.logger.info("[FSDP] Gradient checkpointing is disabled")

        # Record the original trainable parameter names before FSDP wrapping.
        # Persistent buffer names are also recorded for selective weight syncing.
        self.param_names_need_sync = collect_param_names_need_sync(module)

        # build model, optimizer, lr_scheduler, grad_scaler
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        # When precision is null (e.g. Pi0), detect actual dtype from wrapped model
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype
        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                model=target_module, device_mesh=self._device_mesh
            )
            self.target_model.requires_grad_(False)
            self.target_model_initialized = True

        self.use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        use_dsrl = self.use_dsrl
        if use_dsrl:
            # DSRL: separate actor/critic encoders into different optimizer groups
            param_filters = {
                "critic": ["critic_image_encoder", "critic_state_encoder", "q_head"]
            }
        else:
            param_filters = {"critic": ["encoders", "encoder", "q_head", "state_proj"]}
        filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=param_filters,
            filtered_optim_config=filtered_optim_config,
        )
        self.optimizer = optimizers[0]
        self.qf_optimizer = optimizers[1]

        # SAC alpha
        # Initialize temperature parameter for automatic entropy tuning
        alpha_type = self.cfg.algorithm.entropy_tuning.get(
            "alpha_type", "softplus"
        )  # supported type: ["softplus","exp","fixed_alpha"]
        self.entropy_temp = EntropyTemperature(
            initial_alpha=self.cfg.algorithm.entropy_tuning.get("initial_alpha", 0.01),
            alpha_type=alpha_type,
            device=self.device,
            dtype=self.torch_dtype,
        )
        if alpha_type != "fixed_alpha":
            self.target_entropy = self.cfg.algorithm.entropy_tuning.get(
                "target_entropy",
                -self.cfg.actor.model.action_dim,
            )

            self.alpha_optimizer = torch.optim.Adam(
                self.entropy_temp.parameters(),
                lr=self.cfg.algorithm.entropy_tuning.optim.lr,
            )

        self.build_lr_schedulers()

        self.grad_scaler = self.build_grad_scaler(
            self.cfg.actor.fsdp_config.grad_scaler
        )

    def build_lr_schedulers(self):
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        if self.alpha_optimizer is not None:
            self.alpha_lr_scheduler = self.build_lr_scheduler(
                self.alpha_optimizer, self.cfg.algorithm.entropy_tuning.optim
            )

    def setup_sac_components(self):
        """Initialize SAC-specific components"""
        # Initialize replay buffer
        seed = self.cfg.actor.get("seed", 1234)
        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
            cache_size=self.cfg.algorithm.replay_buffer.cache_size,
            sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
            auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=self.cfg.algorithm.replay_buffer.get(
                "trajectory_format", "pt"
            ),
        )
        replay_load_path = self.cfg.algorithm.replay_buffer.get("load_path", None)
        if replay_load_path is not None:
            self.replay_buffer.load_checkpoint(
                replay_load_path,
                is_distributed=True,
                local_rank=self._rank,
                world_size=self._world_size,
            )

        min_demo_buffer_size = 0
        if self.cfg.algorithm.get("demo_buffer", None) is not None:
            auto_save_path = self.cfg.algorithm.demo_buffer.get("auto_save_path", None)
            if auto_save_path is None:
                auto_save_path = os.path.join(
                    self.cfg.runner.logger.log_path, f"demo_buffer/rank_{self._rank}"
                )
            else:
                auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
            self.demo_buffer = TrajectoryReplayBuffer(
                seed=seed,
                enable_cache=self.cfg.algorithm.demo_buffer.enable_cache,
                cache_size=self.cfg.algorithm.demo_buffer.cache_size,
                sample_window_size=self.cfg.algorithm.demo_buffer.sample_window_size,
                auto_save=self.cfg.algorithm.demo_buffer.get("auto_save", False),
                auto_save_path=auto_save_path,
                trajectory_format="pt",
            )
            min_demo_buffer_size = self.cfg.algorithm.demo_buffer.min_buffer_size
            if self.cfg.algorithm.demo_buffer.get("load_path", None) is not None:
                self.demo_buffer.load_checkpoint(
                    self.cfg.algorithm.demo_buffer.load_path,
                    is_distributed=True,
                    local_rank=self._rank,
                    world_size=self._world_size,
                )

        if self.cfg.algorithm.replay_buffer.get("enable_preload", False):
            buffer_dataset_cls = PreloadReplayBufferDataset
        else:
            buffer_dataset_cls = ReplayBufferDataset
        self.buffer_dataset = buffer_dataset_cls(
            replay_buffer=self.replay_buffer,
            demo_buffer=self.demo_buffer,
            batch_size=self.cfg.actor.global_batch_size // self._world_size,
            min_replay_buffer_size=self.cfg.algorithm.replay_buffer.min_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            prefetch_size=self.cfg.algorithm.replay_buffer.get("prefetch_size", 10),
        )
        self.buffer_dataloader = DataLoader(
            self.buffer_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            collate_fn=replay_buffer_collate_fn,
        )
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)

        self.critic_actor_ratio = self.cfg.algorithm.get("critic_actor_ratio", 1)
        self.critic_subsample_size = self.cfg.algorithm.get("critic_subsample_size", -1)
        self.critic_sample_generator = torch.Generator(self.device)
        self.critic_sample_generator.manual_seed(seed)

        self.target_update_type = self.cfg.algorithm.get("target_update_type", "all")
        assert self.target_update_type in ["all", "q_head_only"], (
            f"{self.target_update_type=} is not suppported!"
        )

    def _init_target_shadow(self):
        """Create persistent float32 shadow of target model parameters.

        bfloat16 has only 7 mantissa bits (ULP ~0.002 at magnitude 0.3).
        With tau=0.005, per-step EMA delta can be smaller than ULP/2, so
        storing back to bf16 each step rounds away the update. The shadow
        keeps the accumulated EMA state in float32 (ULP ~3.6e-8) across
        steps, preventing precision loss.
        """
        self._target_shadow_f32 = {}
        online_parameters = dict(self.model.named_parameters())
        for name, param in self.target_model.named_parameters():
            online_param = online_parameters.get(name)
            if online_param is not None and online_param.requires_grad:
                self._target_shadow_f32[name] = param.data.float().clone()

    def soft_update_target_model(self, tau: Optional[float] = None):
        """Soft update target model parameters.

        For DSRL (bfloat16 models), uses a persistent float32 shadow buffer
        to prevent EMA precision loss. For non-DSRL SAC, uses direct EMA
        on model parameters.
        """
        if tau is None:
            tau = self.cfg.algorithm.tau

        assert self.target_model_initialized

        with torch.no_grad():
            if not hasattr(self, "_target_shadow_f32"):
                # Non-DSRL path (or before shadow init): direct EMA update
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert name1 == name2
                    if "q_head" not in name1:
                        if self.target_update_type == "all":
                            target_param.data.mul_(1.0 - tau)
                            target_param.data.add_(online_param.data * tau)
                        else:
                            target_param.data.mul_(0.0)
                            target_param.data.add_(online_param.data)
                    else:
                        target_param.data.mul_(1.0 - tau)
                        target_param.data.add_(online_param.data * tau)
            else:
                # DSRL path: float32 shadow buffer for bf16 precision
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert name1 == name2
                    if "q_head" not in name1 and self.target_update_type != "all":
                        shadow = self._target_shadow_f32.get(name1)
                        if shadow is None:
                            continue
                        shadow.copy_(online_param.data.float())
                        target_param.data.copy_(shadow.to(target_param.data.dtype))
                    else:
                        shadow = self._target_shadow_f32.get(name1)
                        if shadow is None:
                            continue
                        shadow.mul_(1.0 - tau).add_(
                            online_param.data.float(), alpha=tau
                        )
                        target_param.data.copy_(shadow.to(target_param.data.dtype))

    def _supports_selective_target_checkpoint(self) -> bool:
        return (
            self.cfg.actor.fsdp_config.get("sharding_strategy") == "no_shard"
            and self.cfg.actor.fsdp_config.get("use_orig_params", False)
        )

    def _build_selective_target_checkpoint(self) -> dict:
        """Save only target parameters that can differ from the frozen base."""
        online_parameters = dict(self.model.named_parameters())
        target_parameters = dict(self.target_model.named_parameters())
        trainable_names = sorted(
            name for name, param in online_parameters.items() if param.requires_grad
        )
        missing = [name for name in trainable_names if name not in target_parameters]
        if missing:
            raise RuntimeError(
                f"Target model is missing trainable parameters: {missing[:8]}"
            )
        return {
            "format": "trainable_target_v1",
            "parameters": {
                name: target_parameters[name].detach().cpu().clone()
                for name in trainable_names
            },
            "shadow_f32": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self._target_shadow_f32.items()
            },
        }

    def _load_selective_target_checkpoint(self, checkpoint: dict) -> None:
        if checkpoint.get("format") != "trainable_target_v1":
            raise ValueError(
                f"Unsupported target checkpoint format: {checkpoint.get('format')}"
            )
        target_parameters = dict(self.target_model.named_parameters())
        for name, value in checkpoint["parameters"].items():
            if name not in target_parameters:
                raise RuntimeError(f"Target checkpoint parameter {name!r} is missing")
            target_parameters[name].data.copy_(
                value.to(
                    device=target_parameters[name].device,
                    dtype=target_parameters[name].dtype,
                )
            )

        self._init_target_shadow()
        for name, value in checkpoint.get("shadow_f32", {}).items():
            if name not in self._target_shadow_f32:
                raise RuntimeError(f"Target shadow parameter {name!r} is missing")
            self._target_shadow_f32[name].copy_(
                value.to(device=self._target_shadow_f32[name].device, dtype=torch.float32)
            )

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []

        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        intervene_traj_list = []
        if self.demo_buffer is not None:
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                intervene_trajs = traj.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)

        if self.use_dsrl and self.cfg.algorithm.replay_buffer.get(
            "compact_dsrl", False
        ):
            for trajectory in recv_list:
                _compact_dsrl_replay_trajectory(trajectory, self.device)

        self.replay_buffer.add_trajectories(recv_list)

        if self.demo_buffer is not None:
            if len(intervene_traj_list) > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        if use_dsrl:
            num_action_chunks = self.cfg.actor.model.get("num_action_chunks", 1)
            discount = self.cfg.algorithm.gamma**num_action_chunks
            rewards_for_bootstrap = _select_dsrl_transition_reward(
                batch["rewards"],
                aggregation=str(
                    self.cfg.algorithm.get("dsrl_reward_aggregation", "last")
                ),
            ).to(self.torch_dtype)
        else:
            discount = self.cfg.algorithm.gamma
            rewards_for_bootstrap = (
                batch["rewards"].sum(dim=-1, keepdim=True).to(self.torch_dtype)
            )
        terminations = batch["terminations"].to(self.torch_dtype)

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]

        with torch.no_grad():
            kwargs = {}
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                kwargs["temperature"] = (
                    self.cfg.rollout.sampling_params.temperature_train
                )
            if use_dsrl:
                kwargs["train"] = True
            next_state_actions, next_state_log_pi, shared_feature = self.model(
                forward_type=ForwardType.SAC, obs=next_obs, **kwargs
            )
            if next_state_log_pi.ndim == 1:
                next_state_log_pi = next_state_log_pi.unsqueeze(-1)
            next_state_log_pi = next_state_log_pi.sum(dim=-1, keepdim=True)
            if not use_crossq:
                dsrl_kwargs = {"train": True} if use_dsrl else {}
                all_qf_next_target = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_state_actions,
                    shared_feature=None,
                    **dsrl_kwargs,
                )
                if self.critic_subsample_size > 0:
                    sample_idx = torch.randint(
                        0,
                        all_qf_next_target.shape[-1],
                        (self.critic_subsample_size,),
                        generator=self.critic_sample_generator,
                        device=self.device,
                    )
                    all_qf_next_target = all_qf_next_target.index_select(
                        dim=-1, index=sample_idx
                    )

                if agg_q == "min":
                    qf_next_target, _ = torch.min(
                        all_qf_next_target, dim=1, keepdim=True
                    )
                elif agg_q == "mean":
                    qf_next_target = torch.mean(all_qf_next_target, dim=1, keepdim=True)

                if self.cfg.algorithm.get("backup_entropy", True):
                    qf_next_target = (
                        qf_next_target - self.entropy_temp.alpha * next_state_log_pi
                    )
                    qf_next_target = qf_next_target.to(dtype=self.torch_dtype)
                if bootstrap_type == "always":
                    target_q_values = (
                        rewards_for_bootstrap + discount * qf_next_target
                    )  # [bsz, 1]
                elif bootstrap_type == "standard":
                    target_q_values = (
                        rewards_for_bootstrap
                        + (~(terminations.any(dim=-1, keepdim=True)))
                        * discount
                        * qf_next_target
                    )  # [bsz, 1]
                else:
                    raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        if not use_crossq:
            dsrl_kwargs = {"train": True} if use_dsrl else {}
            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
                **dsrl_kwargs,
            )
        else:
            all_data_q_values, all_qf_next = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_state_actions,
            )

            all_qf_next = all_qf_next.detach()
            if agg_q == "min":
                qf_next, _ = torch.min(all_qf_next, dim=1, keepdim=True)
            elif agg_q == "mean":
                qf_next = torch.mean(all_qf_next, dim=1, keepdim=True)
            if self.cfg.algorithm.get("backup_entropy", True):
                qf_next = qf_next - self.entropy_temp.alpha * next_state_log_pi
                qf_next = qf_next.to(dtype=self.torch_dtype)

            if bootstrap_type == "always":
                target_q_values = rewards_for_bootstrap + discount * qf_next  # [bsz, 1]
            elif bootstrap_type == "standard":
                target_q_values = (
                    rewards_for_bootstrap
                    + (~(terminations.any(dim=-1, keepdim=True))) * discount * qf_next
                )  # [bsz, 1]
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        # Align dtype: bool ops with Python floats promote to float32,
        # which can mismatch with bfloat16 model outputs.
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss, {"q_data": all_data_q_values.mean().item()}

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        if "actor_agg_q" in self.cfg.algorithm:
            agg_q = self.cfg.algorithm["actor_agg_q"]
        else:
            agg_q = self.cfg.algorithm.get("agg_q", "min")

        curr_obs = batch["curr_obs"]
        kwargs = {}
        if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
            kwargs["temperature"] = self.cfg.rollout.sampling_params.temperature_train
        if self.use_dsrl:
            kwargs["train"] = True
        pi, log_pi, shared_feature = self.model(
            forward_type=ForwardType.SAC, obs=curr_obs, **kwargs
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)  # sum over the chunk dimension
        if not use_crossq:
            dsrl_kwargs = {"train": True} if self.use_dsrl else {}
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi,
                shared_feature=None,
                detach_encoder=True,
                **dsrl_kwargs,
            )
        else:
            all_qf_pi, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi,
                next_obs=None,
                next_actions=None,
                shared_feature=None,
                detach_encoder=True,
            )
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(self.cfg.actor.model.get("num_q_heads", 2))
        }
        if agg_q == "min":
            qf_pi, _ = torch.min(all_qf_pi, dim=1, keepdim=True)
        elif agg_q == "mean":
            qf_pi = torch.mean(all_qf_pi, dim=1, keepdim=True)
        metrics["q_pi"] = qf_pi.mean().item()
        actor_loss = ((self.entropy_temp.alpha * log_pi) - qf_pi).mean()

        entropy = -log_pi.mean()
        return actor_loss, entropy, metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        curr_obs = batch["curr_obs"]
        with torch.no_grad():
            kwargs = {}
            if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
                kwargs["temperature"] = (
                    self.cfg.rollout.sampling_params.temperature_train
                )
            if self.use_dsrl:
                kwargs["train"] = True
            _, log_pi, _ = self.model(
                forward_type=ForwardType.SAC, obs=curr_obs, **kwargs
            )
            if log_pi.ndim == 1:
                log_pi = log_pi.unsqueeze(-1)
            log_pi = log_pi.sum(dim=-1, keepdim=True)

        alpha = self.entropy_temp.compute_alpha()
        alpha_loss = -alpha * (log_pi.mean() + self.target_entropy)
        return alpha_loss

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self, train_actor: bool = True):
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )

        with self.worker_timer("sample"):
            global_batch = next(self.buffer_dataloader_iter)

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )

        # move train_micro_batch_list to device and apply DRQ for critic/actor/alpha passes
        for i, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            train_micro_batch_list[i] = batch

        self.qf_optimizer.zero_grad()
        gbs_critic_loss = []
        all_critic_metrics = {}
        for batch in train_micro_batch_list:
            critic_loss, critic_metrics = self.forward_critic(batch)
            critic_loss = critic_loss / self.gradient_accumulation
            critic_loss.backward()
            gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
            append_to_dict(all_critic_metrics, critic_metrics)
        all_critic_metrics = {
            f"critic/{key}": np.mean(value) for key, value in all_critic_metrics.items()
        }
        qf_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.critic_optim.clip_grad
        )

        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()

        metrics_data = {
            "sac/critic_loss": np.mean(gbs_critic_loss),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": qf_grad_norm,
            **all_critic_metrics,
        }

        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            self.optimizer.zero_grad()
            gbs_actor_loss = []
            gbs_entropy = []
            all_actor_metrics = {}
            for batch in train_micro_batch_list:
                actor_loss, entropy, q_metrics = self.forward_actor(batch)
                actor_loss = actor_loss / self.gradient_accumulation
                actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
                gbs_entropy.append(entropy.item())
                append_to_dict(all_actor_metrics, q_metrics)
            all_actor_metrics = {
                f"actor/{key}": np.mean(value)
                for key, value in all_actor_metrics.items()
            }
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()

            # Update temperature parameter if using automatic entropy tuning
            gbs_alpha_loss = [0]
            alpha_grad_norm = 0
            if self.alpha_optimizer is not None:
                self.alpha_optimizer.zero_grad()
                gbs_alpha_loss = []
                for batch in train_micro_batch_list:
                    alpha_loss = self.forward_alpha(batch) / self.gradient_accumulation
                    alpha_loss.backward()
                    gbs_alpha_loss.append(
                        alpha_loss.item() * self.gradient_accumulation
                    )
                torch.distributed.all_reduce(
                    self.entropy_temp.base_alpha.grad, op=torch.distributed.ReduceOp.AVG
                )
                alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.entropy_temp.base_alpha,
                    self.cfg.algorithm.entropy_tuning.optim.clip_grad,
                )
                self.alpha_optimizer.step()
                self.alpha_lr_scheduler.step()

            # Collect metrics
            metrics_data.update(
                {
                    "sac/actor_loss": np.mean(gbs_actor_loss),
                    "sac/alpha_loss": np.mean(gbs_alpha_loss),
                    "sac/alpha": self.entropy_temp.alpha,
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    "actor/entropy": np.mean(gbs_entropy),
                    "alpha/grad_norm": alpha_grad_norm,
                    **all_actor_metrics,
                }
            )
        # Soft update target network
        if (
            self.target_model_initialized
            and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0
        ):
            self.soft_update_target_model()

        return metrics_data

    def process_train_metrics(self, metrics):
        replay_buffer_stats = self.replay_buffer.get_stats()
        replay_buffer_stats = {
            f"replay_buffer/{key}": value for key, value in replay_buffer_stats.items()
        }
        append_to_dict(metrics, replay_buffer_stats)

        if self.demo_buffer is not None:
            demo_buffer_stats = self.demo_buffer.get_stats()
            demo_buffer_stats = {
                f"demo_buffer/{key}": value for key, value in demo_buffer_stats.items()
            }
            append_to_dict(metrics, demo_buffer_stats)
        # Average metrics across updates
        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0:
                # Convert tensor values to CPU and detach before computing mean
                cpu_values = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        cpu_values.append(v.detach().cpu().item())
                    else:
                        cpu_values.append(v)
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                # Handle single values
                if isinstance(value, torch.Tensor):
                    mean_metric_dict[key] = value.detach().cpu().item()
                else:
                    mean_metric_dict[key] = value

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    @Worker.timer("run_training")
    def run_training(self):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        # Check if replay buffer has enough samples
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        if not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
            )
            return {}

        # Delay actor training until buffer has enough samples
        train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0)
        train_actor_steps = max(min_buffer_size, train_actor_steps)
        train_actor = self.replay_buffer.is_ready(train_actor_steps)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}

        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch(train_actor=train_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        Worker.torch_platform.synchronize()
        torch.distributed.barrier()
        Worker.torch_platform.empty_cache()
        return mean_metric_dict

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self):
        """
        SAC doesn't compute advantages/returns like PPO.
        This method is kept for compatibility but returns empty metrics.
        """
        return {}

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        # Save model
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
            save_full_model_weights=self.cfg.actor.fsdp_config.get(
                "save_full_model_weights", True
            ),
        )

        # Save sac components
        # save alpha
        if self.alpha_optimizer is not None:
            alpha_save_path = os.path.join(save_base_path, "sac_components/alpha")
            self._strategy.save_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                save_path=alpha_save_path,
                save_full_model_weights=False,
            )

        # save target model
        target_model_save_path = os.path.join(
            save_base_path, "sac_components/target_model"
        )
        os.makedirs(target_model_save_path, exist_ok=True)
        if self._supports_selective_target_checkpoint():
            if self._rank == 0:
                torch.save(
                    self._build_selective_target_checkpoint(),
                    os.path.join(target_model_save_path, "target_state.pt"),
                )
        else:
            # Full-state collection is collective, so every rank participates;
            # CPU offload prevents the checkpoint from cloning a second full
            # target policy on each accelerator.
            target_model_state_dict = self._strategy.get_model_state_dict(
                self.target_model, cpu_offload=True, full_state_dict=True
            )
            if self._rank == 0:
                torch.save(
                    target_model_state_dict,
                    os.path.join(target_model_save_path, "target_state.pt"),
                )
        torch.distributed.barrier()

        trainer_state_path = os.path.join(
            save_base_path, "sac_components", "trainer_state"
        )
        os.makedirs(trainer_state_path, exist_ok=True)
        trainer_state = {
            "update_step": self.update_step,
            "critic_sample_generator_state": self.critic_sample_generator.get_state().cpu(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            trainer_state["device_rng_state"] = torch.cuda.get_rng_state(self.device)
        torch.save(
            trainer_state,
            os.path.join(trainer_state_path, f"rank_{self._rank}.pt"),
        )

        # save replay buffer
        buffer_save_path = os.path.join(
            save_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.save_checkpoint(buffer_save_path)

    def load_checkpoint(self, load_base_path):
        # load model
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # load alpha
        if self.alpha_optimizer is not None:
            alpha_load_path = os.path.join(load_base_path, "sac_components/alpha")
            self._strategy.load_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                load_path=alpha_load_path,
            )

        # load target model
        target_model_load_path = os.path.join(
            load_base_path, "sac_components/target_model"
        )
        target_state_path = os.path.join(target_model_load_path, "target_state.pt")
        if os.path.isfile(target_state_path):
            target_checkpoint = torch.load(target_state_path, map_location="cpu")
            if target_checkpoint.get("format") == "trainable_target_v1":
                self._load_selective_target_checkpoint(target_checkpoint)
            else:
                self._strategy.load_model_with_state_dict(
                    self.target_model,
                    target_checkpoint,
                    cpu_offload=True,
                    full_state_dict=True,
                )
                if self.use_dsrl:
                    self._init_target_shadow()
        else:
            # Backward compatibility with checkpoints produced before the
            # lightweight target format.
            legacy_target_path = os.path.join(
                target_model_load_path, f"checkpoint_rank_{self._rank}.pt"
            )
            target_model_state_dict = torch.load(legacy_target_path, map_location="cpu")
            self._strategy.load_model_with_state_dict(
                self.target_model,
                target_model_state_dict,
                cpu_offload=True,
                full_state_dict=True,
            )
            if self.use_dsrl:
                self._init_target_shadow()

        trainer_state_path = os.path.join(
            load_base_path,
            "sac_components",
            "trainer_state",
            f"rank_{self._rank}.pt",
        )
        if os.path.isfile(trainer_state_path):
            trainer_state = torch.load(trainer_state_path, map_location="cpu")
            self.update_step = int(trainer_state["update_step"])
            self.critic_sample_generator.set_state(
                trainer_state["critic_sample_generator_state"]
            )
            torch.set_rng_state(trainer_state["torch_rng_state"])
            if "device_rng_state" in trainer_state and torch.cuda.is_available():
                torch.cuda.set_rng_state(trainer_state["device_rng_state"], self.device)

        # load replay buffer
        buffer_load_path = os.path.join(
            load_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)
