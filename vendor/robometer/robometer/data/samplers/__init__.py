"""Lazy sampler exports.

The evaluation server only needs the core samplers. Eagerly importing every
offline evaluator also imports optional embedding stacks.
"""

from importlib import import_module

_EXPORTS = {
    "RBMBaseSampler": ("robometer.data.samplers.base", "RBMBaseSampler"),
    "PrefSampler": ("robometer.data.samplers.pref", "PrefSampler"),
    "ProgressSampler": ("robometer.data.samplers.progress", "ProgressSampler"),
    "ConfusionMatrixSampler": (
        "robometer.data.samplers.eval.confusion_matrix",
        "ConfusionMatrixSampler",
    ),
    "ProgressPolicyRankingSampler": (
        "robometer.data.samplers.eval.progress_policy_ranking",
        "ProgressPolicyRankingSampler",
    ),
    "RewardAlignmentSampler": (
        "robometer.data.samplers.eval.reward_alignment",
        "RewardAlignmentSampler",
    ),
    "QualityPreferenceSampler": (
        "robometer.data.samplers.eval.quality_preference",
        "QualityPreferenceSampler",
    ),
    "RoboArenaQualityPreferenceSampler": (
        "robometer.data.samplers.eval.roboarena_quality_preference",
        "RoboArenaQualityPreferenceSampler",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
