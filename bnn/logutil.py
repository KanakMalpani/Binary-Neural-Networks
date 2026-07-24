"""Minimal structured-ish logging for CLI/scripts (stdout, flush)."""

from __future__ import annotations

import sys
from typing import Any


def log(level: str, msg: str, **fields: Any) -> None:
    """Print ``LEVEL msg key=val ...`` to stdout (or stderr for WARN/ERROR)."""
    level_u = level.upper()
    extra = " ".join(f"{k}={v!r}" for k, v in fields.items())
    line = f"{level_u} {msg}" + (f" {extra}" if extra else "")
    stream = sys.stderr if level_u in {"WARN", "WARNING", "ERROR"} else sys.stdout
    print(line, flush=True, file=stream)


def info(msg: str, **fields: Any) -> None:
    log("INFO", msg, **fields)


def warn(msg: str, **fields: Any) -> None:
    log("WARN", msg, **fields)


def error(msg: str, **fields: Any) -> None:
    log("ERROR", msg, **fields)
