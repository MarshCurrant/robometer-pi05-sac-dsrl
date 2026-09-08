"""
Utility functions for Hydra configuration.

Hydra handles configuration loading natively, so this module provides
helper functions for converting OmegaConf DictConfig to dataclasses.
"""

from omegaconf import OmegaConf, DictConfig
from typing import Type, TypeVar

T = TypeVar("T")


def dictconfig_to_dataclass(cfg: DictConfig, config_class: Type[T]) -> T:
    """
    Convert an OmegaConf DictConfig to a dataclass instance.

    Args:
        cfg: OmegaConf DictConfig to convert
        config_class: The dataclass class to instantiate

    Returns:
        An instance of config_class with values from cfg
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return config_class(**cfg_dict)
