__all__ = [
    "RBMDataset",
    "StrategyFirstDataset",
    "BaseDataset",
    "CustomEvalDataset",
    "RepeatedDataset",
]

_LAZY_IMPORTS = {
    "RBMDataset": ("robometer.data.datasets.rbm_data", "RBMDataset"),
    "StrategyFirstDataset": (
        "robometer.data.datasets.strategy_first_dataset",
        "StrategyFirstDataset",
    ),
    "BaseDataset": ("robometer.data.datasets.base", "BaseDataset"),
    "CustomEvalDataset": ("robometer.data.datasets.custom_eval", "CustomEvalDataset"),
    "RepeatedDataset": ("robometer.data.datasets.repeated_dataset", "RepeatedDataset"),
}


def __getattr__(name):
    """Avoid importing training-only Hugging Face datasets in scorer clients."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
