"""In-process script dispatch for the ``bnn`` CLI.

Handlers live in ``_commands``; this module is the seam tests patch
(``ROOT``, ``_run_script``, ``_load_script_module``, ``_run_bridge``).
Lookups of ``ROOT`` / ``_load_script_module`` go through ``bnn.cli`` so
``monkeypatch.setattr(cli, ...)`` keeps working after the package split.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from bnn.paths import REPO_ROOT

# Tests monkeypatch ``bnn.cli.ROOT``; ``_script_path`` reads it via the package.
ROOT = REPO_ROOT

# Top-level verbs printed in ``bnn --help`` epilog (issue #2). Keep in sync
# with ``build_parser()`` subparsers; ``tests/test_cli_surface.py`` fails if a
# registered verb is missing here (unless listed in that test's INTERNAL_COMMANDS).
PUBLIC_CLI_VERBS: tuple[str, ...] = (
    "bench",
    "bridge",
    "compile-native",
    "decode",
    "encode",
    "energy-bound",
    "eval-suite",
    "export-check",
    "kg",
    "memory",
    "optimise",
    "pareto",
    "profile",
    "recommend",
    "repro",
    "train",
    "train-audio",
    "train-cifar",
    "train-image",
    "train-seq2seq",
    "validate-native",
    "version",
    "wrap",
    "wrap-transformer",
)

EPILOG = """
Thesis: packed binary/ternary kernels for CPU/edge inference.
Training (STE) is simulation — not a GPU 32× claim.
Reproduce:  bnn repro
Agents:     see AGENTS.md
Docs:       REPRODUCIBILITY.md

Commands: {verbs}
""".strip().format(verbs=", ".join(PUBLIC_CLI_VERBS))

# First-class production bridges (W12 / docs/23–24). Keys are CLI subcommands.
BRIDGE_RECIPES: dict[str, dict[str, str]] = {
    "gpu": {
        "script": "torchao_int4_recipe.py",
        "doc": "docs/24_GPU_INT4_FP8_LANE.md",
        "lane": "gpu-server",
        "summary": "Commodity NVIDIA → INT4/FP8 (torchao / AWQ / vLLM)",
    },
    "cpu-llm": {
        "script": "llamacpp_bitnet_recipe.py",
        "doc": "docs/23_BITNET_CPP_BRIDGE.md",
        "lane": "cpu-llm",
        "summary": "CPU chat LLMs → GGUF Q4 (llama.cpp) or BitNet → bitnet.cpp",
    },
}
BRIDGE_ALIASES: dict[str, str] = {
    "torchao": "gpu",
    "bitnet": "cpu-llm",
    "llamacpp": "cpu-llm",
}


def _pkg() -> ModuleType:
    """Late-bind the public CLI package (preserves test monkeypatches)."""
    import bnn.cli as cli

    return cli


def _script_path(script: str) -> Path:
    """Resolve a bare ``*.py`` name under ``scripts/`` (no path separators)."""
    if Path(script).name != script or not script.endswith(".py"):
        raise FileNotFoundError(f"refusing non-basename script: {script!r}")
    scripts_root = (_pkg().ROOT / "scripts").resolve()
    path = (scripts_root / script).resolve()
    if not path.is_relative_to(scripts_root) or not path.is_file():
        raise FileNotFoundError(path)
    return path


def _bridge_script_path(script: str) -> Path:
    """Resolve a bare ``*.py`` under ``scripts/bridges/`` (no path separators)."""
    if Path(script).name != script or not script.endswith(".py"):
        raise FileNotFoundError(f"refusing non-basename bridge script: {script!r}")
    bridges_root = (_pkg().ROOT / "scripts" / "bridges").resolve()
    path = (bridges_root / script).resolve()
    if not path.is_relative_to(bridges_root) or not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_script_module(path: Path) -> ModuleType:
    """Import ``path`` once as ``bnn._scripts.<stem>``; undo cache on failure."""
    mod_name = f"bnn._scripts.{path.stem}"
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load script {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise
    return module


def _exit_code(result: object) -> int:
    """Map ``main()`` returns / ``SystemExit.code`` to a process exit status."""
    if result is None or result is True:
        return 0
    if result is False:
        return 1
    if isinstance(result, int):
        return result
    return 1 if result else 0


@contextmanager
def _temporary_argv(argv: list[str]) -> Iterator[None]:
    previous = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = previous


def _call_script_main(main: Callable[..., object], path: Path, argv: list[str]) -> int:
    """Invoke ``main``, supporting either ``main(argv)`` or ``parse_args()``-style mains."""
    if "argv" in inspect.signature(main).parameters:
        return _exit_code(main(argv))
    with _temporary_argv([str(path), *argv]):
        return _exit_code(main())


def _run_bridge(script: str, extra: list[str] | None = None) -> int:
    """Run ``scripts/bridges/<script>`` in-process (same exit contract as ``_run_script``)."""
    argv = list(extra or [])
    path = _bridge_script_path(script)
    print(f"> scripts/bridges/{path.name}", *argv, flush=True)
    try:
        main = getattr(_pkg()._load_script_module(path), "main", None)
        if not callable(main):
            raise RuntimeError(f"{path.name} has no callable main()")
        return _call_script_main(main, path, argv)
    except SystemExit as exc:
        return _exit_code(exc.code)
    except Exception as exc:
        print(f"ERROR {path.name}: {exc}", file=sys.stderr, flush=True)
        return 1


def _run_script(script: str, extra: list[str] | None = None) -> int:
    """Run ``scripts/<script>`` in-process (testable; same exit-code contract as subprocess).

    ``main(argv)`` gets ``extra`` directly. Otherwise ``sys.argv`` is rewritten for
    scripts that call ``argparse.parse_args()`` with no arguments.
    """
    argv = list(extra or [])
    path = _script_path(script)
    print(f"> scripts/{path.name}", *argv, flush=True)
    try:
        main = getattr(_pkg()._load_script_module(path), "main", None)
        if not callable(main):
            raise RuntimeError(f"{path.name} has no callable main()")
        return _call_script_main(main, path, argv)
    except SystemExit as exc:
        return _exit_code(exc.code)
    except Exception as exc:
        print(f"ERROR {path.name}: {exc}", file=sys.stderr, flush=True)
        return 1
