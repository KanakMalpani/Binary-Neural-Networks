"""Lane I: pinned llama.cpp / bitnet.cpp bridge recipe (no network, no build)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bridges" / "llamacpp_bitnet_recipe.py"
PINS = ROOT / "scripts" / "bridges" / "llamacpp_bitnet_pins.json"

# Import helpers without installing as a package.
sys.path.insert(0, str(SCRIPT.parent))
import llamacpp_bitnet_recipe as bridge  # noqa: E402


def test_pins_schema_and_policy():
    pins = bridge.load_pins(PINS)
    assert pins["schema"] == bridge.SCHEMA
    assert pins["policy"]["vendor_submodule"] is False
    assert pins["bitnet_cpp"]["ref"]
    assert pins["llama_cpp_gguf"]["ref"]
    assert "BitNet" in pins["models"]["bitnet_b158_2b_4t_gguf"]["hf_id"]


def test_build_recipe_embeds_pins():
    pins = bridge.load_pins(PINS)
    recipe = bridge.build_recipe(pins)
    assert recipe["schema"] == "bnn.bridge_cpu_llamacpp_bitnet.v1"
    assert recipe["pins"]["bitnet_cpp_ref"] == pins["bitnet_cpp"]["ref"]
    assert recipe["pins"]["llama_cpp_gguf_ref"] == pins["llama_cpp_gguf"]["ref"]
    assert recipe["pins"]["vendor_submodule"] is False
    joined = "\n".join(recipe["bitnet_cpp"])
    assert pins["bitnet_cpp"]["ref"] in joined
    assert "setup_env.py" in joined
    assert any("Q4_K_M" in s for s in recipe["llama_cpp_gguf"])


def test_cli_check_ok():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert payload["pins"]["vendor_submodule"] is False


def test_cli_writes_results_json(tmp_path: Path):
    out = tmp_path / "bridge.json"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["lane"] == "cpu-llm"
    assert data["pins"]["bitnet_cpp_ref"]


def test_probe_missing_checkout(tmp_path: Path):
    missing = tmp_path / "no-bitnet"
    probe = bridge.probe_local_bitnet(missing)
    assert probe["found"] is False
    assert "hint" in probe


def test_probe_finds_setup_env(tmp_path: Path):
    fake = tmp_path / "BitNet"
    fake.mkdir()
    (fake / "setup_env.py").write_text("# stub\n", encoding="utf-8")
    (fake / "3rdparty" / "llama.cpp").mkdir(parents=True)
    probe = bridge.probe_local_bitnet(fake)
    assert probe["found"] is True
    assert probe["has_setup_env"] is True
    assert probe["has_3rdparty_llama"] is True


def test_load_pins_rejects_bad_schema(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "nope"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        bridge.load_pins(bad)


def test_docs_23_and_third_party_note_exist():
    assert (ROOT / "docs" / "23_BITNET_CPP_BRIDGE.md").is_file()
    assert (ROOT / "third_party" / "BITNET_PIN.md").is_file()
    assert (ROOT / "docs" / "lanes" / "i.md").is_file()
