"""Safe path helpers — prevent traversal outside allowed roots."""

from __future__ import annotations

from pathlib import Path

from .logutil import warn

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories we treat as "lab-local" for pack / checkpoint loads (W10.T06).
_TRUSTED_PACK_ROOTS = ("results", "checkpoints", "data")


class PathSecurityError(ValueError):
    """Raised when a user path escapes an allowed root."""


def resolve_under(root: Path | str, user_path: Path | str, *, must_exist: bool = False) -> Path:
    """Resolve ``user_path`` and require it stays under ``root``.

    Relative paths are joined to ``root``. Absolute paths must still resolve
    inside ``root`` after ``Path.resolve()``.
    """
    root_r = Path(root).resolve()
    p = Path(user_path)
    candidate = (root_r / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        candidate.relative_to(root_r)
    except ValueError as exc:
        raise PathSecurityError(
            f"Path {user_path!s} escapes allowed root {root_r}"
        ) from exc
    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def data_path(*parts: str, create: bool = False) -> Path:
    """Return a path under ``<repo>/data`` (created optionally)."""
    base = REPO_ROOT / "data"
    if create:
        base.mkdir(parents=True, exist_ok=True)
    if not parts:
        return base
    return resolve_under(base, Path(*parts))


def results_path(*parts: str) -> Path:
    """Return a path under ``<repo>/results``."""
    base = REPO_ROOT / "results"
    if not parts:
        return base
    return resolve_under(base, Path(*parts))


def repo_relative(path: Path | str) -> str:
    """Path as a POSIX string relative to the repo root, for committed JSON.

    Committed results are read on other people's machines and diffed across
    them, so an absolute path is both non-portable and a needless disclosure of
    the author's home directory. Falls back to the bare filename when the path
    lies outside the repo.
    """
    p = Path(path)
    try:
        resolved = p.resolve()
    except OSError:
        return p.name
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.name


def is_under_repo_trusted_pack_root(path: Path | str) -> bool:
    """True when ``path`` resolves under repo ``results/``, ``checkpoints/``, or ``data/``."""
    try:
        resolved = Path(path).resolve()
        rel = resolved.relative_to(REPO_ROOT.resolve())
    except (OSError, ValueError):
        return False
    parts = rel.parts
    return bool(parts) and parts[0] in _TRUSTED_PACK_ROOTS


def warn_untrusted_pack(path: Path | str, *, kind: str = ".bnnpack") -> bool:
    """Emit a soft warning when loading a pack/checkpoint from outside lab roots.

    Returns True if a warning was emitted. Does **not** block the load —
    ``load_bnnpack`` still enforces ``weights_only=True`` (no pickle fallback).
    """
    p = Path(path)
    if is_under_repo_trusted_pack_root(p):
        return False
    warn(
        f"loading {kind} from outside lab results/checkpoints/data — "
        "treat as untrusted; refuse files you did not produce",
        path=str(p),
    )
    return True
