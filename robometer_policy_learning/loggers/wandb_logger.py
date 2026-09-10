# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from typing import Optional, List, Dict

from torch import Tensor

from .logger import Logger


try:
    import wandb

    _has_wandb = True
except ImportError:
    _has_wandb = False


class WandbLogger(Logger):
    """Wrapper for the wandb logger.

    Args:
        exp_name (str): The name of the experiment.

    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._prev_video_step = -1
        return super().__new__(cls)

    def __init__(
        self,
        exp_name: str,
        offline: bool = False,
        id: str = None,
        project: str = None,
        log_dir: str = None,
        prefix: str = None,
        entity: str = None,
        group: str = None,
        job_type: str = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not _has_wandb:
            raise ImportError("WandB is not installed")

        self.offline = offline
        self.id = id
        self.project = project
        self.entity = entity
        self.group = group
        self.job_type = job_type
        self.tags = tags
        self.notes = notes
        allowlist = os.environ.get("WANDB_METRIC_ALLOWLIST", "").strip()
        self._metric_allowlist = {item.strip() for item in allowlist.split(",") if item.strip()}
        self._disable_videos = os.environ.get("WANDB_DISABLE_VIDEOS", "0").lower() in {
            "1",
            "true",
            "yes",
        }

        # If an id is provided but no explicit resume policy, allow attaching to existing runs
        if id is not None and "resume" not in kwargs and not offline:
            kwargs["resume"] = "allow"

        # Build init kwargs. If resuming an existing run by id, avoid overriding the run name.
        base_kwargs = {
            "dir": log_dir,
            "id": id,
            "project": project,
            **({"entity": entity} if entity is not None else {}),
            **kwargs,
        }
        if group is not None:
            base_kwargs["group"] = group
        if job_type is not None:
            base_kwargs["job_type"] = job_type
        if tags is not None:
            base_kwargs["tags"] = tags
        if notes is not None:
            base_kwargs["notes"] = notes
        if id is None:
            base_kwargs["name"] = exp_name
        self._wandb_kwargs = base_kwargs
        super().__init__(exp_name=exp_name, log_dir=log_dir, prefix=prefix)
        if self.offline:
            os.environ["WANDB_MODE"] = "dryrun"

        self._has_imported_moviepy = False

        self.video_log_counter = 0

        self.initialize()

    def _metric_name(self, name: str, prefix: Optional[str]) -> str:
        """Build one stable W&B key without duplicated namespace components."""
        effective_prefix = prefix if prefix is not None else self.prefix
        if not effective_prefix:
            return name
        if name.startswith(f"{effective_prefix}/"):
            return name
        return f"{effective_prefix}/{name}"

    def _metric_is_allowed(self, name: str) -> bool:
        return not self._metric_allowlist or name in self._metric_allowlist

    @property
    def run_id(self) -> Optional[str]:
        try:
            return getattr(self.logger, "id", None)
        except Exception:
            return None

    @property
    def run_url(self) -> Optional[str]:
        try:
            return getattr(self.logger, "url", None)
        except Exception:
            return None

    def initialize(self) -> None:
        """
        Initialize the wandb logger.
        """
        if self.offline:
            os.environ["WANDB_MODE"] = "dryrun"

        if not _has_wandb:
            raise ImportError("Wandb is not installed")
        self.logger = wandb.init(**self._wandb_kwargs)

    def log_scalar(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        prefix: str = None,
        **kwargs,
    ) -> None:
        """Logs a scalar value to wandb.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged.
                Defaults to None.
        """
        metric_name = self._metric_name(name, prefix)
        if not self._metric_is_allowed(metric_name):
            return
        if step is not None:
            axis = metric_name.rsplit("/", 1)[0] + "/step" if "/" in metric_name else "step"
            self.logger.log({metric_name: value, axis: step})
        else:
            self.logger.log({metric_name: value})

    def log_video(
        self,
        name: str,
        video: Tensor,
        step: Optional[int] = None,
        prefix: str = None,
        **kwargs,
    ) -> None:
        """Log videos inputs to wandb.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to be logged.
            **kwargs: Other keyword arguments. By construction, log_video
                supports 'step' (integer indicating the step index), 'format'
                (default is 'mp4') and 'fps' (default: 6). Other kwargs are
                passed as-is to the :obj:`logger.log` method.
        """
        if self._disable_videos:
            return
        # check for correct format of the video tensor ((N), T, C, H, W)
        # check that the color channel (C) is either 1 or 3
        if video.dim() != 5 or video.size(dim=2) not in {1, 3}:
            raise Exception("Wrong format of the video tensor. Should be ((N), T, C, H, W)")
        if not self._has_imported_moviepy:
            try:
                import moviepy  # noqa

                self._has_imported_moviepy = True
            except ImportError:
                raise Exception("moviepy not found, videos cannot be logged with TensorboardLogger")
        self.video_log_counter += 1
        fps = kwargs.pop("fps", 6)
        fmt = kwargs.pop("format", "mp4")
        # Consume and ignore unsupported kwargs for wandb.log (e.g., prefix already handled)
        _ = kwargs.pop("commit", None)  # optionally could pass commit if needed
        # Handle prefix
        if prefix is None:
            prefix = self.prefix
        # Update internal step tracker (not strictly needed, but kept for continuity)
        if step not in (None, self._prev_video_step, self._prev_video_step + 1):
            warnings.warn(
                "when using step with wandb_logger.log_video, it is expected "
                "that the step is equal to the previous step or that value incremented "
                f"by one. Got step={step} but previous value was {self._prev_video_step}. "
                f"The step value will be set to {self._prev_video_step + 1}. This warning will "
                f"be silenced from now on but the values will keep being incremented."
            )
            step = self._prev_video_step + 1
        self._prev_video_step = step if step is not None else self._prev_video_step + 1

        data = {f"{prefix}/{name}": wandb.Video(video, fps=fps, format=fmt)}
        if step is not None:
            data[f"{prefix}/step"] = step
        self.logger.log(data)

    def log_dict(self, dictionary: dict, step: Optional[int] = None, prefix: str = None, **kwargs) -> None:
        dictionary = {
            metric_name: value
            for key, value in dictionary.items()
            if self._metric_is_allowed(metric_name := self._metric_name(key, prefix))
        }
        if not dictionary:
            return
        # Avoid using wandb's global step kwarg to prevent out-of-order warnings.
        # If a step is provided, log it as a separate metric that users can map via define_metric.
        if step is not None:
            effective_prefix = prefix if prefix is not None else self.prefix
            axis = f"{effective_prefix}/step" if effective_prefix else "step"
            dictionary[axis] = step
        self.logger.log(dictionary, **kwargs)

    def log_hparams(self, cfg: dict) -> None:
        """Logs the hyperparameters of the experiment.

        Args:
            cfg (DictConfig): The configuration of the experiment.

        """
        self.logger.config.update(cfg, allow_val_change=True)

    def log_artifact(
        self,
        path: str,
        name: Optional[str] = None,
        type: str = "checkpoint",
        metadata: Optional[Dict] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Log a file or directory as a W&B artifact.

        Args:
            path: File or directory to log.
            name: Optional artifact name; defaults to exp_name + basename.
            type: Artifact type, defaults to "checkpoint".
            metadata: Optional metadata dict.
            aliases: Optional list of aliases (e.g., ["latest", "final"]).
        """
        if not os.path.exists(path):
            warnings.warn(f"Artifact path does not exist: {path}. Skipping.")
            return

        base_name = os.path.basename(path.rstrip("/"))
        artifact_name = f"{self.exp_name}-{base_name}-{name}"
        artifact = wandb.Artifact(name=artifact_name, type=type, metadata=metadata or {})

        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)

        if aliases is None:
            aliases = ["latest"]

        # Log the artifact to the current run
        self.logger.log_artifact(artifact, aliases=aliases)

    def finish(self) -> None:
        """Finalize/close the current W&B run."""
        try:
            if hasattr(self, "logger") and hasattr(self.logger, "finish"):
                self.logger.finish()
            else:
                wandb.finish()
        except Exception:
            try:
                wandb.finish()
            except Exception:
                pass
