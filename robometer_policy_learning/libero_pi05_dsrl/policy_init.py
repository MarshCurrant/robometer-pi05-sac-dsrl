"""Initialization helpers for Pi0.5 residual-noise SAC policies."""

from __future__ import annotations

import torch


def initialize_residual_gaussian_actor(actor, *, output_gain: float = 0.01) -> None:
    """Match RLinf/DSRL's near-zero initial residual-noise policy.

    The frozen Pi0.5 checkpoint is already a capable behavior policy. SAC acts
    in its flow-noise space, so a newly initialized actor must start close to
    zero residual noise instead of replacing the base policy with an arbitrary
    random perturbation.
    """

    if output_gain < 0:
        raise ValueError("output_gain must be non-negative")
    for name in ("mean_layer", "log_std_layer"):
        layer = getattr(actor, name, None)
        if layer is None or not isinstance(layer, torch.nn.Linear):
            raise TypeError(f"Pi0.5 DSRL actor requires a linear {name}")
        torch.nn.init.xavier_uniform_(layer.weight, gain=float(output_gain))
        torch.nn.init.zeros_(layer.bias)
