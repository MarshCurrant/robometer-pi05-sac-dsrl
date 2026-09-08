import abc
from typing import Optional, List, Dict
import torch

__all__ = ["Logger"]


class Logger:
    """A generic logger class."""

    def __init__(self, exp_name: str, log_dir: str, prefix: str = None) -> None:
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.prefix = prefix

    def set_prefix(self, prefix: str) -> None:
        self.prefix = prefix

    @abc.abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def log_scalar(self, name: str, value: float, step: int = None, prefix: str = None, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def log_video(
        self,
        name: str,
        video: torch.Tensor,
        step: int = None,
        prefix: str = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def log_hparams(self, cfg: dict) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def log_dict(self, dictionary: dict, step: int = None, prefix: str = None, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def log_artifact(
        self,
        path: str,
        name: Optional[str] = None,
        type: str = "checkpoint",
        metadata: Optional[Dict] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Log a file or directory as an artifact to the underlying logger.

        Args:
            path: Path to a file or directory to log.
            name: Optional artifact name. If None, a name will be derived.
            type: Artifact type/category, e.g., "checkpoint".
            metadata: Optional metadata dictionary to attach.
            aliases: Optional list of aliases such as ["latest"].
        """
        raise NotImplementedError

    # A generic log method that sends the logs to the right functions for lazy people
    def log(
        self,
        name: str,
        value: float | torch.Tensor,
        step: int = None,
        prefix: str = None,
        **kwargs,
    ) -> None:
        if isinstance(value, torch.Tensor):
            self.log_video(name, value, step, prefix, **kwargs)
        else:
            self.log_scalar(name, value, step, prefix, **kwargs)

    def log(self, dictionary: dict, step: int = None, prefix: str = None, **kwargs) -> None:
        self.log_dict(dictionary, step, prefix, **kwargs)

    @abc.abstractmethod
    def finish(self) -> None:
        raise NotImplementedError
