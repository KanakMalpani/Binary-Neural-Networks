"""Safe path helpers — prevent traversal outside allowed roots."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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
