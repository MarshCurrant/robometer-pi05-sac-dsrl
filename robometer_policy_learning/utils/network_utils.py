import torch
import torch.nn as nn


def rename_module_key(key: str) -> str:
    """Rename a module key by replacing .  and / with _ as pytorch will complain if there are . in the key"""
    return key.replace(".", "_").replace("/", "_")


def polyak_update(params, target_params, tau):
    """Polyak averaging for target network updates.

    Handles the case where params and target_params may share storage (e.g., when
    some submodules are intentionally shared). In that case, avoid in-place ops
    that zero-out shared tensors; instead, copy directly.
    """
    with torch.no_grad():
        # Fast path: direct copy when tau == 1.0
        if tau == 1.0:
            for p, tp in zip(params, target_params):
                if tp.data.data_ptr() == p.data.data_ptr():
                    # Same tensor; nothing to do
                    continue
                tp.data.copy_(p.data)
            return

        # General Polyak averaging
        for p, tp in zip(params, target_params):
            if tp.data.data_ptr() == p.data.data_ptr():
                # Shared storage; fallback to direct copy for safety
                tp.data.copy_(p.data)
            else:
                tp.data.mul_(1.0 - tau)
                tp.data.add_(p.data, alpha=tau)


class CriticEnsemble(nn.Module):
    """Ensemble of critic networks for SAC."""

    def __init__(self, critics):
        super().__init__()
        self.critics = nn.ModuleList(critics)
        self.num_critics = len(critics)

    def forward(self, obs, action, critic_indices=None):
        q_values = []
        if critic_indices is not None:
            for idx in critic_indices:
                q_values.append(self.critics[idx](obs, action))
        else:
            for critic in self.critics:
                q_values.append(critic(obs, action))
        return q_values

    def parameters(self, recurse=True):
        """Return parameters from all critics, deduplicated for shared params.

        This is critical when using pooled_critic_features=True, where the
        transformer encoder and feature extractors are shared across critics.
        Without deduplication, shared parameters would receive gradient updates
        multiple times per optimizer step, effectively multiplying the learning
        rate for those parameters.
        """
        seen_ids = set()
        for critic in self.critics:
            for param in critic.parameters(recurse=recurse):
                param_id = id(param)
                if param_id not in seen_ids:
                    seen_ids.add(param_id)
                    yield param


# def polyak_update(params, target_params, tau):
#    """Polyak averaging for target network updates (foreach-optimized when available)."""
#    with torch.no_grad():
#        # Convert to lists if needed
#        params = list(params) if not isinstance(params, list) else params
#        target_params = list(target_params) if not isinstance(target_params, list) else target_params
#
#        try:
#            torch._foreach_mul_(target_params, 1.0 - tau)
#            torch._foreach_add_(target_params, params, alpha=tau)
#        except Exception:
#            for param, target_param in zip(params, target_params):
#                target_param.data.mul_(1 - tau)
#                target_param.data.add_(tau * param.data)


# class CriticEnsemble(nn.Module):
#    """Ensemble of critic networks for SAC."""
#
#    def __init__(self, critics):
#        super().__init__()
#        self.critics = nn.ModuleList(critics)
#        self.num_critics = len(critics)
#
#    def forward(self, obs, action, critic_indices=None):
#        q_values = []
#        if critic_indices is not None:
#            for idx in critic_indices:
#                q_values.append(self.critics[idx](obs, action))
#        else:
#            for critic in self.critics:
#                q_values.append(critic(obs, action))
#        return q_values
#
#    def parameters(self, recurse=True):
#        """Return parameters from all critics (deduplicated for shared params)."""
#        # EXTREMELY IMPORTANT: Deduplicate by id AND ensure consistent ordering
#        # by using named_parameters to establish a canonical order
#        seen = set()
#        for name, param in self.named_parameters(recurse=recurse):
#            param_id = id(param)
#            if param_id not in seen:
#                seen.add(param_id)
#                yield param
#
#    def named_parameters(self, prefix='', recurse=True):
#        """Return named parameters from all critics (deduplicated for shared params)."""
#        seen = set()
#        for i, critic in enumerate(self.critics):
#            for name, param in critic.named_parameters(prefix=f'{prefix}critics.{i}.', recurse=recurse):
#                param_id = id(param)
#                if param_id not in seen:
#                    seen.add(param_id)
#                    yield (name, param)
