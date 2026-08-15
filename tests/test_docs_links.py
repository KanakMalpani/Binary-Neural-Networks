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


def test_readme_when_not_callout_is_above_the_fold():
    """Issue #1: When-NOT must sit under the thesis, not only the bottom Is/is not table."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lower = text.lower()
    thesis = lower.find("## the thesis")
    when_not = lower.find("when **not**")
    if when_not < 0:
        when_not = lower.find("when not")
    is_is_not = lower.find("## is / is not")
    assert thesis != -1, "README is missing the thesis heading"
    assert when_not != -1, "README is missing a When-NOT callout"
    assert is_is_not != -1, "README is missing the Is/is not table"
    assert thesis < when_not < is_is_not, "When-NOT callout must sit under the thesis, above Is/is not"
    window = text[thesis:when_not + 2500]
    assert "docs/GUIDE_E2E.md" in window, "When-NOT callout must link GUIDE_E2E"
    assert "docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md" in window, "When-NOT callout must link docs/18"
    assert "bnn recommend" in window, "When-NOT callout must point at bnn recommend"
    callout = text[when_not : when_not + 3000]
    assert "bitnet.cpp" in callout.lower()
    assert "gpu" in callout.lower()
    assert "int4" in callout.lower() or "fp8" in callout.lower()


def test_readme_claims_live_pypi_install():
    """W8.T08: advertise `pip install bnn-lab` now that 1.0.0 is on PyPI."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "pypi.org/project/bnn-lab" in text.lower()
    assert any(
        line.strip().startswith("pip install bnn-lab") and "@ git+" not in line
        for line in text.splitlines()
    ), "README should lead with pip install bnn-lab from PyPI"


def _git_pip_fences(text: str) -> list[str]:
    fences = re.findall(r"```(?:bat|bash|sh)?\n(.*?)```", text, re.S)
    return [b for b in fences if "git+" in b and "bnn-lab @" in b]


def test_git_pip_fences_do_not_run_script_clis():
    """Non-editable git-pip wheels do not ship scripts/; do not run bnn repro after them."""
    banned = ("bnn repro", "bnn optimise", "bnn recommend")
    for path in (ROOT / "README.md", ROOT / "docs" / "GUIDE_E2E.md"):
        fences = _git_pip_fences(path.read_text(encoding="utf-8"))
        assert fences, f"{path.name} should still document git-pip install"
        for block in fences:
            for cmd in banned:
                assert cmd not in block, f"{path.name} runs {cmd!r} after git-pip:\n{block}"


def _first_python_fence_after_git_pip(text: str) -> str:
    needle = "git+https://github.com/KanakMalpani/Binary-Neural-Networks.git@v1.0.0"
    idx = text.find(needle)
    assert idx != -1, "missing git-pip install URL"
    match = re.search(r"```python\n(.*?)```", text[idx:], re.S)
    assert match, "missing Python snippet after git-pip install"
    return match.group(1)


def test_git_pip_python_snippet_replaces_ffn_at_32x():
    """Documented git-pip config must wrap FFN layers at 32×, not auto's 0×/16×."""
    for path in (ROOT / "README.md", ROOT / "docs" / "GUIDE_E2E.md"):
        snippet = _first_python_fence_after_git_pip(path.read_text(encoding="utf-8"))
        assert 'policy="hybrid_ffn"' in snippet, path.name
        assert "min_in_features=64" in snippet, path.name
        assert 'policy="auto"' not in snippet, path.name
        ns: dict = {}
        exec(compile(snippet, str(path), "exec"), ns)
        result = ns["result"]
        assert result.report.replaced, (path.name, result.report.skipped)
        assert float(result.payload["compression_replaced_weights"]) == 32.0, (
            path.name,
            result.payload.get("compression_replaced_weights"),
            result.payload.get("status"),
        )


def test_readme_simd_ladder_uses_entry_node():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'Entry["binary_gemm"]' in text


def test_guide_clone_heading_lists_recommend():
    text = (ROOT / "docs" / "GUIDE_E2E.md").read_text(encoding="utf-8")
    heading = next(line for line in text.splitlines() if line.startswith("### 3.1"))
    assert "bnn recommend" in heading


def test_readme_does_not_claim_windows_arm64_wheel():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "windows-amd64" in text
    assert "no Windows ARM64" in text or "no Windows arm64" in text.lower()
    assert "Windows × x86-64 / arm64" not in text


def test_readme_kernel_wrap_simd_bridge_diagrams():
    """Landing page must ship the four concept diagrams (GitHub-renderable mermaid)."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.count("```mermaid") >= 6
    for needle in (
        "Bit-pack",
        "Layer policy",
        "AVX-512",
        "WASM SIMD128",
        "bitnet.cpp",
        "torchao",
    ):
        assert needle in text, needle


def test_api_pages_cover_the_public_api():
    """Everything exported from `bnn` should appear somewhere in the API docs."""
    import bnn

    text = "\n".join(
        p.read_text(encoding="utf-8") for p in (ROOT / "docs" / "api").glob("*.md")
    )
    exported = [n for n in getattr(bnn, "__all__", []) if not n.startswith("__")]
    undocumented = [name for name in exported if name not in text]
    assert not undocumented, f"public API missing from docs/api/: {undocumented}"
