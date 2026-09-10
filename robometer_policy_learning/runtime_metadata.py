"""Dependency-light provenance probe, executable by the isolated reward Python.

Only package names/versions and an allowlist of runtime settings are collected;
pip freeze, installation URLs, command lines, and arbitrary environment variables
can contain credentials and must not be included.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUNTIME_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "ROBOMETER_ATTN_IMPLEMENTATION",
)


def redact_credentials(value: Any) -> Any:
    """Remove credential fields from the server's nested experiment config."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            sensitive = normalized in {"token", "authorization", "cookie", "credentials"}
            sensitive |= normalized.endswith(
                ("_token", "api_key", "password", "secret", "access_key", "private_key")
            )
            # Model flags such as use_per_frame_progress_token are not credentials.
            result[key] = "[REDACTED]" if sensitive and isinstance(item, (str, dict, list)) else (
                redact_credentials(item)
            )
        return result
    if isinstance(value, list):
        return [redact_credentials(item) for item in value]
    return value


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Create a snapshot without ever replacing an earlier launch's evidence."""
    serialized = json.dumps(redact_credentials(payload), indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def collect_runtime_metadata(*, role: str = "reward") -> dict[str, Any]:
    """Inspect this interpreter, not the policy environment invoking the launcher."""
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "captured",
        "source": f"{role}_interpreter_probe",
        "python": {
            # Do not resolve the executable symlink: that loses the venv identity.
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
        "environment": {key: os.environ.get(key) for key in RUNTIME_ENV_KEYS},
    }
    try:
        import torch
    except (ImportError, OSError, RuntimeError) as exc:
        manifest["torch"] = {"status": "unavailable", "error_type": type(exc).__name__}
        return manifest

    torch_info: dict[str, Any] = {
        "status": "captured",
        "version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "hip_version": getattr(torch.version, "hip", None),
        "git_version": torch.version.git_version,
    }
    try:
        torch_info["cudnn_version"] = torch.backends.cudnn.version()
        torch_info["cuda_available"] = torch.cuda.is_available()
        torch_info["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    except (OSError, RuntimeError, AssertionError) as exc:
        # Exception messages may contain arbitrary environment or URL values.
        torch_info["status"] = "partial"
        torch_info["error_type"] = type(exc).__name__
    manifest["torch"] = torch_info
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_manifest(args.output, collect_runtime_metadata())


if __name__ == "__main__":
    main()
