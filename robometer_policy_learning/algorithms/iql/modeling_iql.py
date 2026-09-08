import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from loguru import logger

from robometer_policy_learning.algorithms.modeling_algorithm import BaseAlgorithm
from robometer_policy_learning.algorithms.iql.configuration_iql import IQLConfig
from robometer_policy_learning.utils.network_utils import CriticEnsemble, polyak_update
from robometer_policy_learning.utils.gpu_utils import is_fp16_supported


class IQL(BaseAlgorithm):
    """
    Implicit Q-Learning (IQL) algorithm.
    """

    def __init__(self, config: IQLConfig):
        super().__init__(config)
        self.config = config
        self.actor = config.actor
        self.v_net = config.v_net

        if self.v_net is None:
            # Let's create a default v_net
            raise ValueError("V-Net is required for IQL")

        self.device = next(self.actor.parameters()).device

        self.component_names = [
            "actor",
            "critic",
            "critic_target",
            "v_net",
            "actor_optimizer",
            "critic_optimizer",
            "v_net_optimizer",
        ]

        self.pooled_critic_features = config.pooled_critic_features
        self.critic = self._create_critic_ensemble(config.critic, config.num_critics)
        self.buffer = config.buffer

        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.compute_chunked_gamma = config.compute_chunked_gamma
        self.tau = config.tau
        self.target_update_interval = config.target_update_interval
        self.num_critics = config.num_critics

        # IQL-specific parameters
        self.advantage_temp = config.advantage_temp
        self.expectile = config.expectile
        self.clip_score = config.clip_score
        assert config.policy_extraction in ["awr", "ddpg"], "Policy extraction must be 'awr' or 'ddpg'"
        self.policy_extraction = config.policy_extraction
        self.ddpg_bc_weight = config.ddpg_bc_weight
        self.n_critics_to_sample = config.n_critics_to_sample
        self.current_critic_update_ratio = config.offline_critic_update_ratio

        # Create optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_optimizer_lr,
            eps=config.actor_optimizer_eps,
            weight_decay=config.actor_optimizer_weight_decay,
        )

        # Debug: Check if parameters are properly deduplicated
        critic_params = list(self.critic.parameters())
        logger.info(f"Critic ensemble has {len(critic_params)} param networks after deduplication")

        # Check if transformer encoder is shared (if it exists)
        if hasattr(self.critic.critics[0], "transformer_encoder"):
            encoder_0 = self.critic.critics[0].transformer_encoder
            encoder_1 = self.critic.critics[1].transformer_encoder if len(self.critic.critics) > 1 else None
            logger.info(f"Transformer encoder is same object: {encoder_0 is encoder_1}")

            # Count how many times the transformer encoder's first parameter appears
            first_param = next(encoder_0.parameters())
            count = sum(1 for p in critic_params if p is first_param)
            logger.info(f"Transformer encoder first param appears {count} time(s) in optimizer param list")
            logger.info(f"Transformer encoder first param requires_grad: {first_param.requires_grad}")
            logger.info(f"Transformer encoder first param is leaf: {first_param.is_leaf}")

        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=config.critic_optimizer_lr,
            betas=config.critic_optimizer_betas,
            eps=config.critic_optimizer_eps,
            weight_decay=config.critic_optimizer_weight_decay,
        )

        self.v_net_optimizer = torch.optim.Adam(
            self.v_net.parameters(),
            lr=config.v_net_optimizer_lr,
            betas=config.v_net_optimizer_betas,
            eps=config.v_net_optimizer_eps,
            weight_decay=config.v_net_optimizer_weight_decay,
        )

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

    def train_step(self, logging_prefix: str = "offline", rollout_step: int = None) -> dict:
        if rollout_step is not None and rollout_step < self.learning_starts:
            return {}

        actor_losses, q_losses, v_losses = [], [], []
        actor_log_pis = []
        q_values_list = []
        v_values_list = []
        v_next_values_list = []
        target_q_values_list = []
        reward_values = []

        gradient_steps = self.config.num_updates_per_train_step

        if gradient_steps != 1:
            print(f"Going to take {gradient_steps} training steps")
            print(self.buffer.size())

        # Use global update counter for target network updates (not local gradient_step)
        base_update_step = int(self._n_updates)

        for gradient_step in range(gradient_steps):
            update_step = base_update_step + gradient_step

            # Sample replay buffer with automatic tensor conversion
            batch = self.buffer.sample(self.batch_size, convert_to_tensors=True, device=self.device)
            if not batch:
                print("Buffer is still empty. Skipping this training step")
                return {}

            obs = batch["obs"]
            actions = batch["action"]
            rewards = batch["reward"]
            next_obs = batch["next_obs"]
            dones = batch["done"]

            # if dones is a boolean tensor, convert it to a float tensor
            if dones.dtype == torch.bool:
                dones = dones.float()

            truncateds = batch["truncated"]

            # Use truncated as done signal for IQL
            # dones = truncateds

            if len(obs) == 0:
                print("Buffer is still empty. Skipping this training step")
                return {}

            # # Prepare indices once and split into K chunks to avoid repeated buffer.sample() calls
            # shuffled_indices = np.random.permutation(rewards.shape[0])
            # num_chunks = self.current_critic_update_ratio
            # per_chunk = max(1, int(rewards.shape[0] // max(1, num_chunks)))

            # Critic and value function updates
            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            # IMPORTANT: Compute pooled features INSIDE the loop so gradients flow correctly
            # after each optimizer step updates the transformer weights

            if self.pooled_critic_features:
                pooled = self.critic.critics[0].compute_pooled(obs, actions)
                current_q_values = torch.stack(
                    [c.value_head(pooled) for c in self.critic.critics],
                    dim=1,
                ).squeeze(-1)
            else:
                current_q_values = torch.stack(
                    self.critic(obs, actions),
                    dim=1,
                ).squeeze(-1)
            # Upcast to float32 for stable loss under AMP
            current_q_values_f32 = current_q_values.float()

            with torch.no_grad():
                # # Shared pooled features for next state-actions (precomputed)
                critic_indices = torch.randperm(self.num_critics)[: self.n_critics_to_sample]
                if self.pooled_critic_features:
                    pooled_target = self.critic_target.critics[0].compute_pooled(obs, actions)
                    target_q_value_preds = torch.stack(
                        [self.critic_target.critics[int(i)].value_head(pooled_target) for i in critic_indices],
                        dim=1,
                    ).squeeze(-1)
                else:
                    target_q_value_preds = torch.stack(
                        self.critic_target(obs, actions, critic_indices=critic_indices),
                        dim=1,
                    ).squeeze(-1)
                target_q_pred, _ = torch.min(target_q_value_preds, dim=1)
                target_q_pred = target_q_pred.reshape(-1, 1)

                # Next value function prediction
                self.v_net.eval()
                next_vf_pred = self.v_net(next_obs).reshape(-1, 1)
                self.v_net.train()

            # Current value function prediction
            vf_pred = self.v_net(obs).reshape(-1, 1)

            # If actions are chunked (B, K, A), the next state is K steps ahead.
            # Discount target Q by gamma^K; rewards from sampler are already discounted over K.
            if actions.ndim == 3 and self.compute_chunked_gamma:
                k_steps = actions.size(1)
                gamma_power = self.gamma**k_steps
            else:
                gamma_power = self.gamma

            target_q_values = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * gamma_power * next_vf_pred

            target_q_values_f32 = target_q_values.float().expand_as(current_q_values_f32)
            # Q-value loss (Bellman backup)
            q_loss = F.mse_loss(current_q_values_f32, target_q_values_f32)

            # Value function expectile loss
            vf_err = vf_pred - target_q_pred
            vf_sign = (vf_err > 0).float()
            vf_weight = (1 - vf_sign) * self.expectile + vf_sign * (1 - self.expectile)
            vf_loss = (vf_weight * (vf_err**2)).mean()

            # Log metrics
            q_values_list.append(current_q_values_f32.mean().item())
            v_values_list.append(vf_pred.mean().item())
            v_next_values_list.append(next_vf_pred.mean().item())
            target_q_values_list.append(target_q_values_f32.mean().item())
            q_losses.append(q_loss.item())
            v_losses.append(vf_loss.item())

            # Optimize critic
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()

            # Optimize value function
            self.v_net_optimizer.zero_grad()
            vf_loss.backward()
            self.v_net_optimizer.step()

            # Target network update (using global update counter)
            if self.target_update_interval is not None and int(self.target_update_interval) > 0:
                if update_step % int(self.target_update_interval) == 0:
                    polyak_update(
                        self.critic.parameters(),
                        self.critic_target.parameters(),
                        self.tau,
                    )

            # Policy update
            if self.policy_extraction == "awr":
                # Advantage-weighted regression
                advantage = target_q_pred - vf_pred.detach()
                weights = torch.clamp(torch.exp(advantage * self.advantage_temp), 0, self.clip_score)

                mean_actions, log_std, kwargs = self.actor.get_action_dist_params(obs)
                actions_pi, log_prob = self.actor.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

                # Compute log probability of actual actions
                if log_std is not None:
                    distribution = self.actor.action_dist.proba_distribution(mean_actions, log_std)
                else:
                    distribution = self.actor.action_dist.proba_distribution(mean_actions)

                log_prob_actions = distribution.log_prob(actions)

                # if we have a sequence of actions, we need to average over the sequence
                if log_prob_actions.dim() >= 2:
                    log_prob_actions = log_prob_actions.mean(dim=-1)

                # reshape to [batch, 1]
                log_prob_actions = log_prob_actions.reshape(-1, 1)
                policy_loss = -(weights * log_prob_actions).mean()

            elif self.policy_extraction == "ddpg":
                # DDPG-style policy extraction with behavior cloning
                with torch.no_grad():
                    average_q_value = torch.abs(torch.min(current_q_values, dim=1)[0]).mean()
                    scaled_ddpg_bc_weight = self.ddpg_bc_weight / (average_q_value + 1e-8)

                mean_actions, log_std, kwargs = self.actor.get_action_dist_params(obs)
                actions_pi, _ = self.actor.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

                # Compute log probability of actual actions for BC term
                if log_std is not None:
                    distribution = self.actor.action_dist.proba_distribution(mean_actions, log_std)
                else:
                    distribution = self.actor.action_dist.proba_distribution(mean_actions)
                log_prob_actions = distribution.log_prob(actions)

                # if we have a sequence of actions, we need to average over the sequence
                if log_prob_actions.dim() >= 2:
                    log_prob_actions = log_prob_actions.mean(dim=-1)

                # reshape to [batch, 1]
                log_prob_actions = log_prob_actions.reshape(-1, 1)

                # Q-values for sampled actions
                critic_indices = torch.randperm(self.num_critics)[: self.n_critics_to_sample].to(self.device)
                q_values_pi = self.critic(obs, actions_pi, critic_indices=critic_indices)
                min_qf_pi = torch.min(torch.cat(q_values_pi, dim=1), dim=1)[0]

                policy_loss = -(min_qf_pi + scaled_ddpg_bc_weight * log_prob_actions).mean()

            # Log actor metrics
            reward_values.append(rewards.mean().item())
            actor_losses.append(policy_loss.item())
            actor_log_pis.append(log_prob_actions.mean().item())

            # Optimize actor
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

        self._n_updates += gradient_steps

        metrics_dict = {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(q_losses),
            "v_loss": np.mean(v_losses),
            "q_values_mean": np.mean(q_values_list),
            "v_values_mean": np.mean(v_values_list),
            "v_next_values_mean": np.mean(v_next_values_list),
            "target_q_mean": np.mean(target_q_values_list),
            "reward_mean": np.mean(reward_values),
            "actor_log_pis_mean": np.mean(actor_log_pis),
        }

        if self.logger is not None:
            self.logger.log(metrics_dict, step=self._n_updates, prefix=logging_prefix)

        return metrics_dict

    def _slice_batch_data(self, data, indices):
        """Helper method to slice batch data (handles both dict and tensor observations)."""
        if isinstance(data, dict):
            sliced_data = {}
            for key, value in data.items():
                # Convert numpy indices to tensor indices if needed
                if isinstance(indices, np.ndarray):
                    indices_tensor = torch.from_numpy(indices).long().to(value.device)
                else:
                    indices_tensor = indices.to(value.device).long()

                # Optionally: check for out-of-bounds
                assert indices_tensor.max().item() < value.shape[0], (
                    f"indices out of bounds for {key} with shape {value.shape}"
                )
                sliced_data[key] = value[indices_tensor]
            return sliced_data
        else:
            # Convert numpy indices to tensor indices if needed
            if isinstance(indices, np.ndarray):
                indices_tensor = torch.from_numpy(indices).long().to(data.device)
            else:
                indices_tensor = indices.to(data.device).long()

            assert indices_tensor.max().item() < data.shape[0], "indices out of bounds"
            return data[indices_tensor]
