"""Auditable unit conversions for LIBERO Pi0.5 DSRL training."""

from __future__ import annotations


def resolve_macro_gamma(
    *,
    configured_gamma: float,
    action_exec_len: int,
    discount_unit: str,
) -> float:
    """Resolve a configured discount into the replay transition unit.

    A replay row is one Pi0.5 macro action. ``macro`` therefore uses gamma
    directly, while ``low_level`` preserves the legacy conversion from a
    per-environment-step discount.
    """

    gamma = float(configured_gamma)
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"configured_gamma must be in (0, 1], got {gamma}")
    if int(action_exec_len) <= 0:
        raise ValueError(f"action_exec_len must be positive, got {action_exec_len}")

    unit = str(discount_unit).strip().lower()
    if unit == "macro":
        return gamma
    if unit == "low_level":
        return gamma ** int(action_exec_len)
    raise ValueError(
        f"discount_unit must be 'macro' or 'low_level', got {discount_unit!r}"
    )
