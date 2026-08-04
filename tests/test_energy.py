"""Lane H — energy proxy + RAPL detect (M5)."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from bnn.energy import (
    RAPLUnavailable,
    build_energy_bound,
    detect_rapl,
    energy_proxy_relative,
    estimate_energy,
    proxy_status_windows,
    write_energy_bound,
)
from bnn.energy.rapl import RAPLMeter


def test_estimate_energy_basic():
    row = estimate_energy(latency_s=0.01, power_w=25.0)
    assert row["energy_j"] == pytest.approx(0.25)
    assert row["method"] == "proxy_E_eq_P_times_t"


def test_estimate_energy_with_baseline():
    row = estimate_energy(
        latency_s=0.01,
        power_w=25.0,
        baseline_latency_s=0.02,
        baseline_power_w=35.0,
    )
    assert row["energy_reduction_factor"] == pytest.approx(0.7 / 0.25)


def test_energy_proxy_relative():
    p = energy_proxy_relative(energy_j_fp=1.0, energy_j_binary=0.25)
    assert p["fp"] == 1.0
    assert p["binary"] == pytest.approx(0.25)
    assert p["unit"] == "relative_to_fp"


def test_build_energy_bound_proxy_fields(tmp_path: Path):
    result = build_energy_bound(
        t_fp_s=0.02,
        t_bin_s=0.005,
        power_w_fp=35.0,
        power_w_binary=25.0,
        prefer_rapl=False,
    )
    payload = result.payload
    assert "energy_proxy" in payload
    assert payload["energy_proxy"]["fp"] == 1.0
    assert payload["energy_proxy"]["binary"] == pytest.approx((25.0 * 0.005) / (35.0 * 0.02))
    assert "CLOSED-BY-PROXY" in payload["board_joules_status"]
    assert payload["rapl"]["available"] is False

    out = tmp_path / "energy_bound.json"
    write_energy_bound(result, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["energy_proxy"]["binary"] == payload["energy_proxy"]["binary"]
    assert out.with_suffix(".md").is_file()


def test_detect_rapl_never_raises_on_windows():
    domains = detect_rapl()
    if platform.system() != "Linux":
        assert domains == []
        assert "CLOSED-BY-PROXY" in proxy_status_windows()


def test_rapl_meter_open_raises_without_sysfs():
    if detect_rapl():
        meter = RAPLMeter.open()
        assert meter.domains
        snap = meter.snapshot_uj()
        assert isinstance(snap, dict)
    else:
        with pytest.raises(RAPLUnavailable):
            RAPLMeter.open()


def test_energy_bound_script_smoke(tmp_path: Path):
    wrap = Path("results/wrap_demo.json")
    if not wrap.is_file():
        pytest.skip("wrap_demo.json missing")
    out = tmp_path / "eb.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/energy_bound_measured.py",
            "--wrap-json",
            str(wrap),
            "--out",
            str(out),
            "--no-rapl-probe",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "energy_proxy" in data
    assert data["measurement_method"] == "proxy_E_eq_P_times_t"


def test_energy_rapl_spike_exits_zero(tmp_path: Path):
    out = tmp_path / "spike.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/energy_rapl_spike.py",
            "--out",
            str(out),
            "--sleep-s",
            "0.05",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == "bnn_energy_rapl_spike_v1"
    assert "board_joules_status" in data
    if platform.system() != "Linux":
        assert "CLOSED-BY-PROXY" in data["board_joules_status"]
