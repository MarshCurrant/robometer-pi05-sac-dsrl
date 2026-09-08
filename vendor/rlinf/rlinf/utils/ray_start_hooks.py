"""Local Ray startup hooks for memory-constrained training hosts."""

from __future__ import annotations

import functools
import os

from ray._private import ray_constants, services


class LimitPrestartedPythonWorkers:
    """Override only Ray's idle Python-worker prestart count.

    Ray 2.58 prestarts one idle Python worker per advertised CPU but does not
    expose that count through ``ray start``. RLinf's actors are unaffected by
    this setting: workers are still started on demand and the node continues
    to advertise its full CPU/GPU resources.
    """

    ENV_NAME = "RLINF_RAY_PRESTART_PYTHON_WORKERS"
    ARG_PREFIX = "--num_prestart_python_workers="

    def __init__(self, ray_params, head: bool):  # noqa: ARG002
        prestart_count = int(os.environ.get(self.ENV_NAME, "0"))
        if prestart_count < 0:
            raise ValueError(f"{self.ENV_NAME} must be non-negative")

        original_start = services.start_ray_process
        if getattr(original_start, "_rlinf_limits_prestarted_workers", False):
            return

        @functools.wraps(original_start)
        def start_ray_process(command, process_type, *args, **kwargs):
            if process_type == ray_constants.PROCESS_TYPE_RAYLET:
                command = list(command)
                replacement = f"{self.ARG_PREFIX}{prestart_count}"
                for index, argument in enumerate(command):
                    if argument.startswith(self.ARG_PREFIX):
                        command[index] = replacement
                        break
                else:
                    command.append(replacement)
            return original_start(command, process_type, *args, **kwargs)

        start_ray_process._rlinf_limits_prestarted_workers = True
        services.start_ray_process = start_ray_process
