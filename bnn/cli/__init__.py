"""Console entry point: ``bnn <command> ...``.

Implementation is split for locality (docs/45 P4): dispatch, handlers, parser.
Public imports (``from bnn.cli import main, build_parser, ...``) stay stable.
"""

from __future__ import annotations

import sys

from ._commands import (  # noqa: F401
    _ultra_wrap_extra,
    cmd_bench,
    cmd_bridge,
    cmd_compile_native,
    cmd_decode,
    cmd_encode,
    cmd_energy_bound,
    cmd_eval_suite,
    cmd_export_check,
    cmd_kg,
    cmd_memory,
    cmd_optimise,
    cmd_pareto,
    cmd_profile,
    cmd_recommend,
    cmd_repro,
    cmd_train,
    cmd_train_audio,
    cmd_train_cifar,
    cmd_train_image,
    cmd_train_seq2seq,
    cmd_validate_native,
    cmd_version,
    cmd_wrap,
    cmd_wrap_transformer,
)
from ._dispatch import (  # noqa: F401
    BRIDGE_ALIASES,
    BRIDGE_RECIPES,
    EPILOG,
    PUBLIC_CLI_VERBS,
    ROOT,
    _bridge_script_path,
    _call_script_main,
    _exit_code,
    _load_script_module,
    _run_bridge,
    _run_script,
    _script_path,
    _temporary_argv,
)
from ._parser import build_parser

__all__ = [
    "BRIDGE_ALIASES",
    "BRIDGE_RECIPES",
    "EPILOG",
    "PUBLIC_CLI_VERBS",
    "ROOT",
    "build_parser",
    "main",
]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse already printed help/errors; normalize to int exit
        return _exit_code(exc.code)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"ERROR missing file: {exc}", file=sys.stderr, flush=True)
        return 2
    except RuntimeError as exc:
        print(f"ERROR {exc}", file=sys.stderr, flush=True)
        return 1
