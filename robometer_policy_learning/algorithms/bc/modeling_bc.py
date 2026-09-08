import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from robometer_policy_learning.algorithms.modeling_algorithm import BaseAlgorithm
from robometer_policy_learning.algorithms.bc.configuration_bc import BCConfig


class BC(BaseAlgorithm):
    """
    Behavior Cloning (BC) algorithm.
    A simple imitation learning approach that trains a policy using supervised learning
    on expert demonstrations.
    """

    def __init__(self, config: BCConfig):
        super().__init__(config)
        self.config = config
        self.actor = config.actor

        if self.actor is None:
            raise ValueError("Actor is required for BC")

        self.device = next(self.actor.parameters()).device

        self.component_names = [
            "actor",
            "actor_optimizer",
        ]

        self.buffer = config.buffer

        self.use_weighted_bc = config.use_weighted_bc

        # Store configuration
        self.batch_size = config.batch_size
        self.learning_starts = config.learning_starts
        self.loss_type = config.loss_type
        self.l2_regularization = config.l2_regularization

        # Create optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_optimizer_lr,
            eps=config.actor_optimizer_eps,
            weight_decay=config.actor_optimizer_weight_decay,
        )

        print(f"BC: loss_type = {self.loss_type}")
        print(f"BC: l2_regularization = {self.l2_regularization}")

    def train_step(self, logging_prefix: str = "bc") -> dict:
        """
        Perform one training step of behavior cloning.
        """
        actor_losses = []
        actor_log_pis = []
        expert_action_means = []
        predicted_action_means = []
        mse_errors = []
        unnormalized_mse_errors = []
        unnormalized_max_predicted_actions = []
        unnormalized_min_predicted_actions = []
        gradient_steps = self.config.num_updates_per_train_step

        if gradient_steps != 1:
            print(f"Going to take {gradient_steps} training steps")
            print(self.buffer.size())

        for gradient_step in range(gradient_steps):
            # Sample replay buffer with automatic tensor conversion
            batch = self.buffer.sample(self.batch_size, device=self.device)

            if not batch:
                print("Buffer is still empty. Skipping this training step")
                return {}

            obs = batch["obs"]
            expert_actions = batch["action"]
            if self.use_weighted_bc:
                reward_weights = batch["reward"]

            if len(obs) == 0:
                print("Buffer is still empty. Skipping this training step")
                return {}

            # Data augmentation: Add noise to observations for robustness
            if hasattr(self.config, "obs_noise_std") and self.config.obs_noise_std > 0:
                if isinstance(obs, dict):
                    obs = {k: v + torch.randn_like(v) * self.config.obs_noise_std for k, v in obs.items()}
                else:
                    obs = obs + torch.randn_like(obs) * self.config.obs_noise_std

            # Data augmentation: Add noise to expert actions for robustness
            if hasattr(self.config, "action_noise_std") and self.config.action_noise_std > 0:
                expert_actions = expert_actions + torch.randn_like(expert_actions) * self.config.action_noise_std

            # Get action distribution parameters from the actor
            mean_actions, log_std, kwargs = self.actor.get_action_dist_params(obs)

            if self.loss_type == "mse":
                # For deterministic policies or when we want MSE loss
                # Use the mean actions directly
                predicted_actions = mean_actions
                if self.use_weighted_bc:
                    # Per-sample weighted MSE
                    per_elem = F.mse_loss(predicted_actions, expert_actions, reduction="none")
                    # Reduce over non-batch dims to get per-sample loss
                    if per_elem.dim() > 1:
                        per_sample = per_elem.view(per_elem.size(0), -1).mean(dim=1)
                    else:
                        per_sample = per_elem
                    w = reward_weights.to(per_sample.dtype).view(-1)
                    actor_loss = (w * per_sample).sum() / (w.sum() + 1e-8)
                else:
                    actor_loss = F.mse_loss(predicted_actions, expert_actions)

                # Log probability is not meaningful for MSE, set to zero
                log_prob_actions = torch.zeros(len(expert_actions), 1, device=self.device)

            elif self.loss_type == "nll":
                # For stochastic policies using negative log-likelihood
                if log_std is not None:
                    distribution = self.actor.action_dist.proba_distribution(mean_actions, log_std)
                else:
                    distribution = self.actor.action_dist.proba_distribution(mean_actions)

                # Compute negative log-likelihood of expert actions
                # For chunked actions, we need to handle the shape properly
                if expert_actions.dim() == 3:  # (batch, chunk_size, action_dim)
                    # Reshape to (batch * chunk_size, action_dim) for log_prob computation
                    batch_size, chunk_size, action_dim = expert_actions.shape
                    expert_actions_flat = expert_actions.view(-1, action_dim)
                    mean_actions_flat = mean_actions.view(-1, action_dim)

                    # Create distribution for flattened actions
                    if log_std is not None:
                        log_std_flat = log_std.view(-1, action_dim)
                        distribution_flat = self.actor.action_dist.proba_distribution(mean_actions_flat, log_std_flat)
                    else:
                        distribution_flat = self.actor.action_dist.proba_distribution(mean_actions_flat)

                    # Compute log probabilities for flattened actions
                    log_prob_actions = distribution_flat.log_prob(expert_actions_flat)

                    # Reshape back to (batch, chunk_size) and take mean over chunks
                    log_prob_actions = log_prob_actions.view(batch_size, chunk_size)
                    log_prob_actions = log_prob_actions.mean(dim=1, keepdim=True)
                else:
                    # Non-chunked case
                    log_prob_actions = distribution.log_prob(expert_actions)
                    if log_prob_actions.dim() > 2:
                        log_prob_actions = log_prob_actions.mean(dim=-1, keepdim=True)
                    else:
                        log_prob_actions = log_prob_actions.reshape(-1, 1)

                # Negative log-likelihood loss
                if self.use_weighted_bc:
                    w = reward_weights.to(log_prob_actions.dtype).view(-1, 1)
                    actor_loss = -(w * log_prob_actions).sum() / (w.sum() + 1e-8)
                else:
                    actor_loss = -log_prob_actions.mean()

            elif self.loss_type == "huber":
                # Huber loss for robustness to outliers
                predicted_actions = mean_actions
                if self.use_weighted_bc:
                    per_elem = F.huber_loss(predicted_actions, expert_actions, delta=1.0, reduction="none")
                    if per_elem.dim() > 1:
                        per_sample = per_elem.view(per_elem.size(0), -1).mean(dim=1)
                    else:
                        per_sample = per_elem
                    w = reward_weights.to(per_sample.dtype).view(-1)
                    actor_loss = (w * per_sample).sum() / (w.sum() + 1e-8)
                else:
                    actor_loss = F.huber_loss(predicted_actions, expert_actions, delta=1.0)
                log_prob_actions = torch.zeros(len(expert_actions), 1, device=self.device)

            elif self.loss_type == "smooth_l1":
                # Smooth L1 loss (similar to Huber)
                predicted_actions = mean_actions
                if self.use_weighted_bc:
                    per_elem = F.smooth_l1_loss(predicted_actions, expert_actions, reduction="none")
                    if per_elem.dim() > 1:
                        per_sample = per_elem.view(per_elem.size(0), -1).mean(dim=1)
                    else:
                        per_sample = per_elem
                    w = reward_weights.to(per_sample.dtype).view(-1)
                    actor_loss = (w * per_sample).sum() / (w.sum() + 1e-8)
                else:
                    actor_loss = F.smooth_l1_loss(predicted_actions, expert_actions)
                log_prob_actions = torch.zeros(len(expert_actions), 1, device=self.device)

            else:
                raise ValueError(f"Invalid loss_type: {self.loss_type}. Must be 'mse', 'nll', 'huber', or 'smooth_l1'")

            # Add L2 regularization if specified
            if self.l2_regularization > 0:
                l2_reg = 0
                for param in self.actor.parameters():
                    l2_reg += torch.norm(param) ** 2
                actor_loss += self.l2_regularization * l2_reg

            # Add gradient penalty for better generalization
            if hasattr(self.config, "gradient_penalty_weight") and self.config.gradient_penalty_weight > 0:
                # Compute gradient penalty
                obs.requires_grad_(True)
                mean_actions_grad, _, _ = self.actor.get_action_dist_params(obs)
                gradients = torch.autograd.grad(
                    outputs=mean_actions_grad.sum(),
                    inputs=obs,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]

                if isinstance(gradients, dict):
                    gradient_norm = torch.norm(torch.cat([g.flatten() for g in gradients.values()]))
                else:
                    gradient_norm = torch.norm(gradients)

                gradient_penalty = self.config.gradient_penalty_weight * (gradient_norm - 1.0) ** 2
                actor_loss += gradient_penalty

            # Add consistency regularization (if we have multiple samples)
            if hasattr(self.config, "consistency_weight") and self.config.consistency_weight > 0:
                # Add noise to observations and check consistency
                if isinstance(obs, dict):
                    obs_noisy = {k: v + torch.randn_like(v) * 0.01 for k, v in obs.items()}
                else:
                    obs_noisy = obs + torch.randn_like(obs) * 0.01

                mean_actions_noisy, _, _ = self.actor.get_action_dist_params(obs_noisy)
                consistency_loss = F.mse_loss(mean_actions, mean_actions_noisy)
                actor_loss += self.config.consistency_weight * consistency_loss

            # Log metrics
            actor_losses.append(actor_loss.item())

            mse_error = F.mse_loss(mean_actions, expert_actions)
            mse_errors.append(mse_error.item())

            # unnormalized mse
            unnormalized_mse_error = F.mse_loss(
                self.actor.unnormalize_action(mean_actions),
                self.actor.unnormalize_action(expert_actions),
            )
            unnormalized_mse_errors.append(unnormalized_mse_error.item())

            # unnormalized max predicted actions
            unnormalized_max_predicted_actions.append(self.actor.unnormalize_action(mean_actions).max().item())
            # unnormalized min predicted actions
            unnormalized_min_predicted_actions.append(self.actor.unnormalize_action(mean_actions).min().item())

            if self.loss_type == "nll":
                # Only log probabilities for NLL loss
                actor_log_pis.append(log_prob_actions.mean().item())
            else:
                # For MSE loss, log the actual MSE value instead
                actor_log_pis.append(actor_loss.item())  # Use MSE value directly

            # For chunked actions, compute mean over each chunk separately
            if len(expert_actions.shape) == 3:  # (batch, chunk_size, action_dim)
                # Take mean over each chunk, then mean over batch
                chunk_means = expert_actions.mean(dim=1)  # (batch, action_dim)
                expert_action_means.append(chunk_means.mean().item())
            else:
                expert_action_means.append(expert_actions.mean().item())

            if self.loss_type == "mse":
                predicted_action_means.append(predicted_actions.mean().item())
            else:
                predicted_action_means.append(mean_actions.mean().item())

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            # Gradient clipping for stability
            if hasattr(self.config, "clip_grad_norm") and self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.config.clip_grad_norm)

            self.actor_optimizer.step()

        self.step_counter += gradient_steps

        # Create metrics dictionary based on loss type
        metrics_dict = {
            "actor_loss": np.mean(actor_losses),
            "expert_action_mean": np.mean(expert_action_means),
            "predicted_action_mean": np.mean(predicted_action_means),
            "mse_error": np.mean(mse_errors),
            "unnormalized_mse_error": np.mean(unnormalized_mse_errors),
            "unnormalized_max_predicted_actions": np.mean(unnormalized_max_predicted_actions),
            "unnormalized_min_predicted_actions": np.mean(unnormalized_min_predicted_actions),
        }

        if self.loss_type == "nll":
            metrics_dict["actor_log_pis_mean"] = np.mean(actor_log_pis)
        else:
            metrics_dict["mse_value"] = np.mean(actor_log_pis)  # This is actually the MSE value

        self.logger.log(metrics_dict, step=self.step_counter, prefix=logging_prefix)

        return metrics_dict

    def get_action(self, obs, deterministic: bool = True):
        """
        Get action from the policy.

        Args:
            obs: Observation
            deterministic: Whether to use deterministic action (mean) or sample
        """
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float().to(self.device)
            elif isinstance(obs, dict):
                obs = {k: torch.from_numpy(v).float().to(self.device) for k, v in obs.items()}

            if deterministic:
                # Return mean action
                mean_actions, _, _ = self.actor.get_action_dist_params(obs.unsqueeze(0))
                return mean_actions.squeeze(0).cpu().numpy()
            else:
                # Sample from the distribution
                actions, _ = self.actor.action_log_prob(obs.unsqueeze(0))
                return actions.squeeze(0).cpu().numpy()

    def evaluate_policy(self, eval_buffer, num_eval_batches=10):
        """
        Evaluate the policy on a separate evaluation buffer.
        This helps detect overfitting.
        """
        self.actor.eval()
        eval_losses = []
        eval_mse_errors = []

        with torch.no_grad():
            for _ in range(num_eval_batches):
                batch = eval_buffer.sample(self.batch_size, device=self.device)
                if not batch:
                    continue

                obs = batch["obs"]
                expert_actions = batch["action"]

                mean_actions, log_std, kwargs = self.actor.get_action_dist_params(obs)

                if self.loss_type == "mse":
                    eval_loss = F.mse_loss(mean_actions, expert_actions)
                elif self.loss_type == "nll":
                    if log_std is not None:
                        distribution = self.actor.action_dist.proba_distribution(mean_actions, log_std)
                    else:
                        distribution = self.actor.action_dist.proba_distribution(mean_actions)

                    if expert_actions.dim() == 3:
                        batch_size, chunk_size, action_dim = expert_actions.shape
                        expert_actions_flat = expert_actions.view(-1, action_dim)
                        mean_actions_flat = mean_actions.view(-1, action_dim)

                        if log_std is not None:
                            log_std_flat = log_std.view(-1, action_dim)
                            distribution_flat = self.actor.action_dist.proba_distribution(
                                mean_actions_flat, log_std_flat
                            )
                        else:
                            distribution_flat = self.actor.action_dist.proba_distribution(mean_actions_flat)

                        log_prob_actions = distribution_flat.log_prob(expert_actions_flat)
                        log_prob_actions = log_prob_actions.view(batch_size, chunk_size)
                        log_prob_actions = log_prob_actions.mean(dim=1, keepdim=True)
                    else:
                        log_prob_actions = distribution.log_prob(expert_actions)
                        if log_prob_actions.dim() > 2:
                            log_prob_actions = log_prob_actions.mean(dim=-1, keepdim=True)
                        else:
                            log_prob_actions = log_prob_actions.reshape(-1, 1)

                    eval_loss = -log_prob_actions.mean()
                else:
                    eval_loss = F.mse_loss(mean_actions, expert_actions)

                eval_losses.append(eval_loss.item())
                eval_mse_errors.append(F.mse_loss(mean_actions, expert_actions).item())

        self.actor.train()

        return {
            "eval_loss": np.mean(eval_losses),
            "eval_mse_error": np.mean(eval_mse_errors),
        }

    def compute_uncertainty(self, obs, num_samples=10):
        """
        Compute uncertainty estimates for actions.
        Useful for detecting OOD states.
        """
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float().to(self.device)
            elif isinstance(obs, dict):
                obs = {k: torch.from_numpy(v).float().to(self.device) for k, v in obs.items()}

            actions_list = []
            for _ in range(num_samples):
                actions, _ = self.actor.action_log_prob(obs.unsqueeze(0))
                actions_list.append(actions.squeeze(0))

            actions_stack = torch.stack(actions_list, dim=0)
            uncertainty = torch.std(actions_stack, dim=0)

            return uncertainty.cpu().numpy()

    def apply_early_stopping(self, eval_metrics, patience=5, min_delta=1e-4):
        """
        Simple early stopping based on evaluation metrics.
        """
        if not hasattr(self, "_best_eval_loss"):
            self._best_eval_loss = float("inf")
            self._patience_counter = 0

        current_eval_loss = eval_metrics.get("eval_loss", float("inf"))

        if current_eval_loss < self._best_eval_loss - min_delta:
            self._best_eval_loss = current_eval_loss
            self._patience_counter = 0
            return False  # Don't stop
        else:
            self._patience_counter += 1
            return self._patience_counter >= patience  # Stop if patience exceeded
