import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time
import os

# Perf knobs: allow cudnn to pick best kernels and faster matmul
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from robometer_policy_learning.algorithms.modeling_algorithm import BaseAlgorithm
from robometer_policy_learning.algorithms.sac.configuration_sac import SACConfig
from robometer_policy_learning.modules.base.distributions import kl_divergence
from loguru import logger
from contextlib import nullcontext
from robometer_policy_learning.utils.network_utils import polyak_update
from robometer_policy_learning.utils.gpu_utils import is_fp16_supported


# AMP context (FP16 on CUDA if supported)
amp_dtype = None
if torch.cuda.is_available():
    if is_fp16_supported():
        amp_dtype = torch.float16
        print(f"Using FP16 for SAC")
    else:
        print(f"Using FP32 for SAC")
        amp_dtype = None


def should_update_entropy_coefficient(
    update_step: int, update_interval: int
) -> bool:
    """Return whether alpha should update on this actor optimizer step."""

    interval = int(update_interval)
    if interval <= 0:
        raise ValueError(
            f"ent_coef_update_interval must be positive, got {update_interval}"
        )
    return int(update_step) % interval == 0


def gradient_l2_norm(parameters) -> float:
    """Measure the pre-update gradient norm without modifying gradients."""

    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(2))


class SAC(BaseAlgorithm):
    """
    A SAC algorithm.
    """

    def __init__(self, config: SACConfig):
        super().__init__(config)
        self.config = config  # Store config reference
        self.actor = config.actor

        self.device = next(self.actor.parameters()).device
        if self.config.train_actor_with_kl_divergence:
            self.update_old_actor(self.actor)  # initialize old_actor

        self.component_names = [
            "actor",
            "critic",
            "critic_target",
            "actor_optimizer",
            "critic_optimizer",
            "ent_coef_optimizer",
            "log_ent_coef",
        ]
        self.pooled_critic_features = config.pooled_critic_features

        # Create critic ensemble from single critic
        self.critic = self._create_critic_ensemble(config.critic, config.num_critics)
        self.buffer = config.buffer

        # Store configuration
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.compute_chunked_gamma = config.compute_chunked_gamma
        self.tau = config.tau
        self.target_update_interval = config.target_update_interval
        self.learning_starts = config.learning_starts
        self.num_critics = config.num_critics
        self.critic_reduction = config.critic_reduction
        self.ent_coef_update_interval = int(config.ent_coef_update_interval)
        should_update_entropy_coefficient(0, self.ent_coef_update_interval)
        self.actor_clip_grad_norm = (
            config.actor_clip_grad_norm
            if config.actor_clip_grad_norm is not None
            else config.clip_grad_norm
        )
        self.critic_clip_grad_norm = (
            config.critic_clip_grad_norm
            if config.critic_clip_grad_norm is not None
            else config.clip_grad_norm
        )

        # Entropy coefficient setup
        if config.ent_coef == "auto":
            # Auto-tune entropy coefficient
            self.log_ent_coef = torch.tensor(
                0.0, device=self.device, requires_grad=True
            )  # log(1.0) = 0.0, so ent_coef starts at 1.0

            # Allow manual override of target entropy (e.g. for DSRL noise space)
            if config.target_entropy == "auto" or config.target_entropy is None:
                if hasattr(config, "env") and config.env is not None:
                    if hasattr(config.env, "single_action_space"):
                        self.target_entropy = -np.prod(config.env.single_action_space.shape).astype(np.float32)
                    else:
                        self.target_entropy = -np.prod(config.env.action_space.shape).astype(np.float32)
                else:
                    self.target_entropy = -1.0
            else:
                self.target_entropy = float(config.target_entropy)

            logger.info("SAC: Auto-tuning entropy coefficient enabled")
            logger.info(f"SAC: Target entropy = {self.target_entropy}")
        else:
            self.log_ent_coef = None
            self.target_entropy = None
            self.ent_coef_tensor = torch.tensor(float(config.ent_coef), device=self.device)
            logger.info(f"SAC: Fixed entropy coefficient = {self.ent_coef_tensor.item()}")

        # Create optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_optimizer_lr,
            eps=config.actor_optimizer_eps,
            weight_decay=config.actor_optimizer_weight_decay,
        )

        # Get deduplicated critic parameters
        critic_params = list(self.critic.parameters())
        logger.info(f"Critic ensemble has {len(critic_params)} params (deduplicated)")

        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=config.critic_optimizer_lr,
            betas=config.critic_optimizer_betas,
            eps=config.critic_optimizer_eps,
            weight_decay=config.critic_optimizer_weight_decay,
        )

        # Entropy coefficient optimizer
        if self.log_ent_coef is not None:
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=config.ent_coef_lr)
        else:
            self.ent_coef_optimizer = None

        # Create target critic ensemble (with separate parameters, no sharing with main critic)
        self.critic_target = self._create_critic_ensemble(copy.deepcopy(config.critic), config.num_critics)
        for param in self.critic_target.parameters():
            param.requires_grad = False
        # Initialize target networks with same weights as main networks
        polyak_update(
            self.critic.parameters(),
            self.critic_target.parameters(),
            tau=1.0,  # Complete copy for initialization
        )
        self.critic_target.eval()

        # Training counters
        self._n_updates = 0

    def train_step(self, logging_prefix: str = "online", rollout_step: int = None) -> None:
        if rollout_step is not None and rollout_step < self.learning_starts:
            return {}

        t_total_start = time.perf_counter()
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        actor_log_pis = []
        actor_raw_stds = []
        actor_saturation_fracs = []
        actor_grad_norms = []
        critic_grad_norms = []
        min_qf_pis = []
        q_values_list = []
        q_next_values_list = []
        reward_values = []
        progress_rewards = []  # Collect progress rewards from info dict
        success_probs = []  # Collect success probabilities from info dict

        gradient_steps = self.config.num_updates_per_train_step

        # Timing accumulators
        time_sample_s = 0.0
        time_old_actor_fwd_s = 0.0
        time_actor_params_fwd_s = 0.0
        time_actor_logprob_s = 0.0
        time_batch_slice_s = 0.0
        time_actor_next_action_fwd_s = 0.0
        time_critic_target_fwd_s = 0.0
        time_critic_current_fwd_s = 0.0
        time_critic_backward_s = 0.0
        time_target_update_s = 0.0
        time_entcoef_opt_s = 0.0
        time_actor_value_fwd_s = 0.0
        time_actor_backward_s = 0.0

        # Optional controls
        kl_update_interval = int(getattr(self.config, "kl_update_interval", 1))

        # Use a stable global update counter for schedules (target updates, optional KL cadence)
        base_update_step = int(self._n_updates)

        for gradient_step in range(gradient_steps):
            actor_update_step = base_update_step + gradient_step
            if self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            # Sample replay buffer
            if len(self.buffer) == 0:
                logger.info("Buffer is empty. Skipping this training step")
                return {}
            t0 = time.perf_counter()

            batch = self.buffer.sample(
                self.batch_size * self.config.num_critic_updates_per_actor_update,
                convert_to_tensors=True,
                device=self.device,
            )
            time_sample_s += time.perf_counter() - t0

            if not batch:
                logger.info("Buffer is still empty. Skipping this training step")
                return {}

            obs = batch["obs"]
            actions = batch["action"]
            rewards = batch["reward"]
            next_obs = batch["next_obs"]
            dones = batch["done"]
            truncateds = batch["truncated"]

            if len(obs) == 0:
                logger.info("Buffer is still empty. Skipping this training step")
                return {}

            # Prepare indices for chunking
            shuffled_indices = np.random.permutation(rewards.shape[0])
            num_chunks = self.config.num_critic_updates_per_actor_update
            per_chunk = max(1, int(rewards.shape[0] // max(1, num_chunks)))

            # Compute actor outputs ONCE for the full batch (outside critic loop to avoid memory leak)
            # Use first chunk's worth of data for actor update
            actor_idx = shuffled_indices[:per_chunk]
            actor_observations = self._slice_batch_data(obs, actor_idx)

            with torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else nullcontext():
                t0 = time.perf_counter()
                mean_actions, log_std, kwargs = self.actor.get_action_dist_params(actor_observations)
                time_actor_params_fwd_s += time.perf_counter() - t0
                t0 = time.perf_counter()
                actions_pi, log_prob = self.actor.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
                time_actor_logprob_s += time.perf_counter() - t0

            for critic_update in range(self.config.num_critic_updates_per_actor_update):
                update_step = (
                    base_update_step + gradient_step * self.config.num_critic_updates_per_actor_update + critic_update
                )
                # Take a contiguous slice from the pre-sampled batch
                t0 = time.perf_counter()
                start = critic_update * per_chunk
                end = min(start + per_chunk, shuffled_indices.shape[0])
                if start >= end:
                    start = max(0, shuffled_indices.shape[0] - per_chunk)
                    end = shuffled_indices.shape[0]
                idx_np = shuffled_indices[start:end]

                # Slice all fields
                critic_observations = self._slice_batch_data(obs, idx_np)
                critic_actions = self._slice_batch_data(actions, idx_np)
                critic_rewards = rewards[idx_np]
                critic_next_obs = self._slice_batch_data(next_obs, idx_np)
                critic_dones = dones[idx_np]
                time_batch_slice_s += time.perf_counter() - t0

                # Precompute next actions/logprobs (no grad needed - only used for target Q)
                t0 = time.perf_counter()
                with torch.no_grad():
                    critic_next_actions, critic_next_log_prob = self.actor.action_log_prob(critic_next_obs)
                time_actor_next_action_fwd_s += time.perf_counter() - t0

                # Precompute pooled features for next state (target network, no grad needed)
                t0 = time.perf_counter()
                if self.pooled_critic_features:
                    with torch.no_grad():
                        pooled_next = self.critic_target.critics[0].compute_pooled(critic_next_obs, critic_next_actions)
                time_critic_target_fwd_s += time.perf_counter() - t0

                with torch.no_grad():
                    if critic_next_actions.ndim == 3:
                        critic_next_log_prob = critic_next_log_prob.mean(dim=1, keepdim=True)

                    # Compute the next Q values
                    critic_indices = torch.randperm(self.num_critics)[: self.config.n_critics_to_sample]

                    if self.pooled_critic_features:
                        next_q_values = torch.stack(
                            [self.critic_target.critics[int(i)].value_head(pooled_next) for i in critic_indices],
                            dim=1,
                        ).squeeze(-1)
                    else:
                        next_q_values = torch.stack(
                            self.critic_target(critic_next_obs, critic_next_actions, critic_indices=critic_indices),
                            dim=1,
                        ).squeeze(-1)
                    time_critic_target_fwd_s += time.perf_counter() - t0
                    next_q_values = torch.mean(next_q_values, dim=1, keepdim=True)

                    # Add entropy term
                    if self.config.train_critic_with_entropy:
                        next_q_values = next_q_values - ent_coef * critic_next_log_prob.reshape(-1, 1)

                    # Discount handling for chunked actions
                    if critic_next_actions.ndim == 3 and self.compute_chunked_gamma:
                        k_steps = critic_next_actions.size(1)
                        gamma_power = self.gamma**k_steps
                    else:
                        gamma_power = self.gamma

                    target_q_values = (
                        critic_rewards.reshape(-1, 1)
                        + (1 - critic_dones.reshape(-1, 1).float()) * gamma_power * next_q_values
                    )

                # Get current Q-values
                t0 = time.perf_counter()
                if self.pooled_critic_features:
                    pooled = self.critic.critics[0].compute_pooled(critic_observations, critic_actions)

                    current_q_values = torch.stack(
                        [c.value_head(pooled) for c in self.critic.critics],
                        dim=1,
                    ).squeeze(-1)
                else:
                    current_q_values = torch.stack(
                        self.critic(critic_observations, critic_actions),
                        dim=1,
                    ).squeeze(-1)
                time_critic_current_fwd_s += time.perf_counter() - t0

                # Compute critic loss
                current_q_values_f32 = current_q_values.float()
                target_q_values_f32 = target_q_values.float().expand_as(current_q_values_f32)
                critic_loss = F.mse_loss(current_q_values_f32, target_q_values_f32)
                critic_losses.append(critic_loss.item())

                # Optimize the critic
                t0 = time.perf_counter()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.critic_clip_grad_norm and self.critic_clip_grad_norm > 0:
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), max_norm=self.critic_clip_grad_norm
                    )
                else:
                    critic_grad_norm = gradient_l2_norm(self.critic.parameters())
                critic_grad_norms.append(float(critic_grad_norm))
                self.critic_optimizer.step()
                time_critic_backward_s += time.perf_counter() - t0

                # Target network update (once per gradient_step, using global schedule)
                if self.target_update_interval is not None and int(self.target_update_interval) > 0:
                    if update_step % int(self.target_update_interval) == 0:
                        t0 = time.perf_counter()
                        polyak_update(
                            self.critic.parameters(),
                            self.critic_target.parameters(),
                            self.tau,
                        )
                        time_target_update_s += time.perf_counter() - t0

            if actions_pi.ndim == 3:
                log_prob = log_prob.mean(dim=1, keepdim=True)
            log_prob = log_prob.reshape(-1, 1).float()

            ent_coef_loss = None
            # Only tune entropy coefficient if entropy is actually used somewhere.
            # (Actor entropy term and/or critic target entropy term.)
            should_tune_entropy = bool(
                getattr(self.config, "train_actor_with_entropy", True)
                or getattr(self.config, "train_critic_with_entropy", False)
            )
            update_entropy_coefficient = should_update_entropy_coefficient(
                actor_update_step, self.ent_coef_update_interval
            )
            if (
                should_tune_entropy
                and update_entropy_coefficient
                and self.ent_coef_optimizer is not None
                and self.log_ent_coef is not None
            ):
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + float(self.target_entropy)).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = (
                    torch.exp(self.log_ent_coef.detach())
                    if self.log_ent_coef is not None
                    else self.ent_coef_tensor
                )

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                t0 = time.perf_counter()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                time_entcoef_opt_s += time.perf_counter() - t0

            reward_values.append(rewards.mean().item())
            q_values_list.append(np.mean([q.mean().item() for q in current_q_values]))
            # Historical W&B contract used by the validated sf73jk43 run:
            # ``target_q_mean`` names the sampled target critics' next-Q, not
            # the entropy-adjusted Bellman target. Keep the metric stable;
            # this value is diagnostic only and does not affect the loss.
            q_next_values_list.append(next_q_values.mean().item())

            # Extract progress rewards and success probs from info if available
            batch_info = batch.get("info")
            if batch_info is not None:
                # Handle different formats: list of dicts, tensor, or array
                if isinstance(batch_info, (list, tuple)):
                    for info in batch_info:
                        if isinstance(info, dict):
                            if "relabeled_reward" in info and info["relabeled_reward"] is not None:
                                progress_rewards.append(float(info["relabeled_reward"]))
                            if "success_prob" in info and info["success_prob"] is not None:
                                success_probs.append(float(info["success_prob"]))

            # Compute actor loss (use actor_observations which matches actions_pi)
            with torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else nullcontext():
                t0 = time.perf_counter()
                self.critic.eval()
                if self.pooled_critic_features:
                    pooled_pi = self.critic.critics[0].compute_pooled(actor_observations, actions_pi)
                    q_values_pi = torch.stack(
                        [c.value_head(pooled_pi) for c in self.critic.critics],
                        dim=1,
                    ).squeeze(-1)
                else:
                    q_values_pi = torch.stack(self.critic(actor_observations, actions_pi), dim=1).squeeze(-1)
                self.critic.train()
                time_actor_value_fwd_s += time.perf_counter() - t0

            if self.critic_reduction == "min":
                min_qf_pi = torch.min(q_values_pi, dim=1, keepdim=True)[0].float()
            elif self.critic_reduction == "mean":
                min_qf_pi = q_values_pi.mean(dim=1, keepdim=True).float()
            else:
                raise ValueError(f"Invalid critic reduction: {self.critic_reduction}")

            current_actor_distribution = self.actor.action_dist.proba_distribution(mean_actions, log_std, **kwargs)

            kl_div = None
            # Calculate KL divergence between old and current actor (optional)
            compute_kl_this_step = self.config.train_actor_with_kl_divergence and (
                kl_update_interval <= 1 or (update_step % kl_update_interval == 0)
            )
            old_actor_distribution = None
            if compute_kl_this_step:
                with torch.no_grad():
                    t0 = time.perf_counter()
                    old_mean, old_log_std, _ = self.old_actor.get_action_dist_params(actor_observations)
                    old_actor_distribution = self.old_actor.action_dist.proba_distribution(old_mean, old_log_std)
                    time_old_actor_fwd_s += time.perf_counter() - t0

            if compute_kl_this_step and old_actor_distribution is not None:
                kl_div = kl_divergence(
                    current_actor_distribution,
                    old_actor_distribution,
                ).float()
                kl_div = kl_div.sum(dim=-1, keepdim=True)

            if self.config.train_actor_with_kl_divergence and (kl_div is not None):
                actor_loss = (ent_coef.float() * kl_div - min_qf_pi).mean()
            else:
                # Gate actor entropy term if configured.
                if getattr(self.config, "train_actor_with_entropy", True):
                    actor_loss = (ent_coef.float() * log_prob - min_qf_pi).mean()
                else:
                    actor_loss = (-min_qf_pi).mean()
            min_qf_pis.append(min_qf_pi.mean().item())
            actor_losses.append(actor_loss.item())

            t0 = time.perf_counter()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.actor_clip_grad_norm and self.actor_clip_grad_norm > 0:
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=self.actor_clip_grad_norm
                )
            else:
                actor_grad_norm = gradient_l2_norm(self.actor.parameters())
            actor_grad_norms.append(float(actor_grad_norm))
            self.actor_optimizer.step()
            time_actor_backward_s += time.perf_counter() - t0

            actor_log_pis.append(log_prob.mean().item())
            actor_raw_stds.append(log_std.float().exp().mean().item())
            actor_saturation_fracs.append(
                (actions_pi.detach().float().abs() > 0.95).float().mean().item()
            )

        self._n_updates += gradient_steps

        metrics_dict = {
            "ent_coef": np.mean(ent_coefs),
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "q_values_mean": np.mean(q_values_list),
            "target_q_mean": np.mean(q_next_values_list),
            "reward_mean": np.mean(reward_values),
            "reward_min": np.min(reward_values),
            "reward_max": np.max(reward_values),
            "actor_log_pis_mean": np.mean(actor_log_pis),
            "actor_chosen_q_mean": np.mean(min_qf_pis),
            "actor_entropy": -np.mean(actor_log_pis),
            "actor_raw_std_mean": np.mean(actor_raw_stds),
            "actor_saturation_frac": np.mean(actor_saturation_fracs),
            "ent_coef_update_fraction": len(ent_coef_losses) / max(1, gradient_steps),
        }

        if actor_grad_norms:
            metrics_dict["actor_grad_norm"] = np.mean(actor_grad_norms)
        if critic_grad_norms:
            metrics_dict["critic_grad_norm"] = np.mean(critic_grad_norms)

        if len(ent_coef_losses) > 0:
            metrics_dict["ent_coef_loss"] = np.mean(ent_coef_losses)

        # Add progress rewards and success probs if available
        if len(progress_rewards) > 0:
            metrics_dict["progress_reward_mean"] = np.mean(progress_rewards)
            metrics_dict["progress_reward_min"] = np.min(progress_rewards)
            metrics_dict["progress_reward_max"] = np.max(progress_rewards)

        if len(success_probs) > 0:
            metrics_dict["success_prob_mean"] = np.mean(success_probs)

        total_time_s = time.perf_counter() - t_total_start
        metrics_dict["train_step_total_time_s"] = float(total_time_s)
        metrics_dict.update(
            {
                "time_sample_s": float(time_sample_s),
                "time_old_actor_fwd_s": float(time_old_actor_fwd_s),
                "time_actor_params_fwd_s": float(time_actor_params_fwd_s),
                "time_actor_logprob_s": float(time_actor_logprob_s),
                "time_batch_slice_s": float(time_batch_slice_s),
                "time_actor_next_action_fwd_s": float(time_actor_next_action_fwd_s),
                "time_critic_target_fwd_s": float(time_critic_target_fwd_s),
                "time_critic_current_fwd_s": float(time_critic_current_fwd_s),
                "time_critic_backward_s": float(time_critic_backward_s),
                "time_target_update_s": float(time_target_update_s),
                "time_entcoef_opt_s": float(time_entcoef_opt_s),
                "time_actor_value_fwd_s": float(time_actor_value_fwd_s),
                "time_actor_backward_s": float(time_actor_backward_s),
            }
        )

        if self.logger is not None:
            self.logger.log(metrics_dict, step=self._n_updates, prefix=logging_prefix)
        return metrics_dict

    def _slice_batch_data(self, data, indices):
        """Helper method to slice batch data (handles both dict and tensor observations)."""
        if isinstance(data, dict):
            sliced_data = {}
            for key, value in data.items():
                if isinstance(indices, np.ndarray):
                    indices_tensor = torch.from_numpy(indices).long().to(value.device)
                else:
                    indices_tensor = indices.to(value.device).long()
                sliced_data[key] = value[indices_tensor]
            return sliced_data
        else:
            if isinstance(indices, np.ndarray):
                indices_tensor = torch.from_numpy(indices).long().to(data.device)
            else:
                indices_tensor = indices.to(data.device).long()
            return data[indices_tensor]

    def copy_components(self, other: BaseAlgorithm):
        super().copy_components(other)
        self.update_old_actor(other.actor)

    def update_old_actor(self, other_actor):
        """Create a copy of the actor for KL divergence computation."""
        self.old_actor = type(other_actor)(other_actor.config)

        if hasattr(other_actor, "feature_extractor") and hasattr(self.old_actor, "feature_extractor"):
            self.old_actor.feature_extractor = copy.deepcopy(other_actor.feature_extractor)

        self.old_actor.load_state_dict(other_actor.state_dict())
        self.old_actor.to(self.device)
        self.old_actor.eval()
