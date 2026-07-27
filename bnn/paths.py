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
