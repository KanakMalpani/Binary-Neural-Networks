"""Documentation integrity: nav targets exist, links resolve, autodoc refs are real.

MkDocs cannot check links that point above ``docs_dir`` (README / ROADMAP /
REPRODUCIBILITY / AGENTS live at the repo root by design), so link validation
lives here where the whole repo layout is visible.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
AUTODOC_RE = re.compile(r"^:::\s+([A-Za-z_][\w.]*)\s*$", re.M)
# A concrete home directory such as C:\Users\alice\ . Angle-bracket placeholders
# (C:\Users\<user>\) are documentation, not a leak.
_LEAKED_PATH_RE = re.compile(r"[A-Za-z]:\\Users\\(?!<)[^\\\s<>]+\\")


def _markdown_files() -> list[Path]:
    skip = {"data", "checkpoints", "site", "build", "dist", ".venv", "bnn.egg-info"}
    return [
        p
        for p in ROOT.rglob("*.md")
        if not any(part in skip or part.startswith(".") for part in p.relative_to(ROOT).parts)
    ]


def test_every_relative_markdown_link_resolves():
    """A broken relative link is a real defect — it ships to GitHub readers."""
    broken: list[str] = []
    for md in _markdown_files():
        for target in LINK_RE.findall(md.read_text(encoding="utf-8", errors="ignore")):
            target = target.split("#")[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:", "<")):
                continue
            if not (md.parent / target).exists():
                broken.append(f"{md.relative_to(ROOT)} -> {target}")
    assert not broken, "broken relative links:\n  " + "\n  ".join(broken)


def test_no_committed_absolute_local_paths():
    """A machine-specific path in docs cannot work for a reader."""
    offenders = [
        str(md.relative_to(ROOT))
        for md in _markdown_files()
        if _LEAKED_PATH_RE.search(md.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not offenders, f"docs contain absolute local paths: {offenders}"


# --------------------------------------------------------------------------
# mkdocs site
# --------------------------------------------------------------------------

def _mkdocs_config() -> dict:
    yaml = pytest.importorskip("yaml")

    # mkdocs.yml uses !!python/name: style tags in some setups; ignore unknowns.
    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor(
        "tag:yaml.org,2002:python/name:", lambda loader, suffix, node: None
    )
    Loader.add_multi_constructor("!", lambda loader, suffix, node: None)
    return yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=Loader)


def _nav_targets(nav) -> list[str]:
    out: list[str] = []
    if isinstance(nav, str):
        out.append(nav)
    elif isinstance(nav, list):
        for item in nav:
            out.extend(_nav_targets(item))
    elif isinstance(nav, dict):
        for value in nav.values():
            out.extend(_nav_targets(value))
    return out


def test_mkdocs_nav_targets_all_exist():
    cfg = _mkdocs_config()
    docs_dir = ROOT / cfg.get("docs_dir", "docs")
    missing = [t for t in _nav_targets(cfg.get("nav")) if not (docs_dir / t).exists()]
    assert not missing, f"mkdocs nav points at missing files: {missing}"


def test_mkdocs_has_autodoc_plugin_configured():
    """The API reference is generated, not hand-maintained — keep it that way."""
    plugins = _mkdocs_config().get("plugins") or []
    names = {p if isinstance(p, str) else next(iter(p)) for p in plugins}
    assert "mkdocstrings" in names


def test_every_autodoc_reference_is_importable():
    """`::: bnn.x.y` must resolve, or the API page renders an empty section."""
    api_dir = ROOT / "docs" / "api"
    refs: list[tuple[str, str]] = []
    for md in sorted(api_dir.glob("*.md")):
        for ref in AUTODOC_RE.findall(md.read_text(encoding="utf-8")):
            refs.append((md.name, ref))
    assert refs, "no ::: autodoc references found — did the API pages regress?"

    unresolved: list[str] = []
    for page, ref in refs:
        parts = ref.split(".")
        obj = None
        # Import the longest importable prefix, then walk attributes.
        for split in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:split]))
            except ImportError:
                continue
            for attr in parts[split:]:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            break
        if obj is None:
            unresolved.append(f"{page}: {ref}")
    assert not unresolved, "autodoc references that do not resolve:\n  " + "\n  ".join(unresolved)


def test_api_pages_cover_the_public_api():
    """Everything exported from `bnn` should appear somewhere in the API docs."""
    import bnn

    text = "\n".join(
        p.read_text(encoding="utf-8") for p in (ROOT / "docs" / "api").glob("*.md")
    )
    exported = [n for n in getattr(bnn, "__all__", []) if not n.startswith("__")]
    undocumented = [name for name in exported if name not in text]
    assert not undocumented, f"public API missing from docs/api/: {undocumented}"
