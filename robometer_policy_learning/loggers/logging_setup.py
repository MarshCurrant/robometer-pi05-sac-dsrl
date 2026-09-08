from __future__ import annotations

import os
import sys
from typing import Optional

from loguru import logger


def setup_logging(log_dir: Optional[str] = None, level: str = "INFO") -> None:
    """
    Configure Loguru with a colorful, minimal console sink and an optional rotating file sink.

    Console: whole line colored by bound process (SERVER/OFFLINE/TRAIN/REWARD/ROLLOUT/EVAL)
    File: rotated at ~10MB, retain 7 days, one-line entries
    """
    # Reset handlers to avoid duplicates when reconfiguring per-process
    logger.remove()

    # Pretty, minimal, colorful console sink (whole line per-process color)
    def _console_format(record):
        proc = record["extra"].get("process_name")
        proc_color = {
            "SERVER": "green",
            "OFFLINE": "magenta",
            "TRAIN": "blue",
            "REWARD": "yellow",
            "ROLLOUT": "cyan",
            "EVAL": "magenta",
        }.get(proc, "white")
        proc_tag = f"[{proc}] " if proc else ""
        # Best-effort: fetch the source code line for this record
        code_line = ""
        try:
            file_path = record["file"].path
            line_no = int(record["line"]) if record["line"] is not None else None
            if file_path and line_no is not None and line_no > 0:
                with open(file_path, "r") as _f:
                    for idx, line in enumerate(_f, start=1):
                        if idx == line_no:
                            code_line = line.strip()
                            break
        except Exception:
            pass
        # Include location and (optionally) the source line (escape braces to avoid .format_map issues)
        location = f"{record['file'].name}:{record['function']}:{record['line']}"
        code_line_safe = code_line.replace("{", "{{").replace("}", "}}") if code_line else ""
        source = f" | {code_line_safe}" if code_line_safe else ""
        return f"<{proc_color}>{{level:<8}} | {proc_tag}{location} | {{message}}{source}</{proc_color}>\n"

    logger.add(
        sys.stderr,
        level=level.upper(),
        colorize=True,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format=_console_format,
    )

    # Optional rotating file sink using a callable formatter (safe for braces in messages)
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, "run.log")

            def _file_format(record):
                proc = record["extra"].get("process_name")
                proc_tag = f"[{proc}] " if proc else ""
                # Best-effort: fetch the source code line
                code_line = ""
                try:
                    file_path = record["file"].path
                    line_no = int(record["line"]) if record["line"] is not None else None
                    if file_path and line_no is not None and line_no > 0:
                        with open(file_path, "r") as _f:
                            for idx, line in enumerate(_f, start=1):
                                if idx == line_no:
                                    code_line = line.strip()
                                    break
                except Exception:
                    pass
                location = f"{record['file'].name}:{record['function']}:{record['line']}"
                code_line_safe = code_line.replace("{", "{{").replace("}", "}}") if code_line else ""
                source = f" | {code_line_safe}" if code_line_safe else ""
                # Use Loguru time formatting and include location + source line
                return (
                    f"{{time:YYYY-MM-DDTHH:mm:ss.SSSZZ}} | {{level:<8}} | {proc_tag}{location} | {{message}}{source}\n"
                )

            logger.add(
                file_path,
                level=level.upper(),
                enqueue=True,
                rotation="10 MB",
                retention="7 days",
                compression="zip",
                format=_file_format,
            )
        except Exception:
            # Silent fallback if filesystem not writable
            pass
