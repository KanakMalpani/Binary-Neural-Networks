"""Linux RAPL via ``/sys/class/powercap`` (intel-rapl / amd-rapl).

Windows and locked-down hosts have no portable RAPL in the Python stdlib —
callers must fall back to the energy proxy with ``CLOSED-BY-PROXY`` honesty.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

POWERCAP_ROOT = Path("/sys/class/powercap")


class RAPLUnavailable(RuntimeError):
    """Raised when RAPL sysfs is missing, unreadable, or OS is not Linux."""


@dataclass(frozen=True)
class RAPLDomain:
    """One powercap energy counter."""

    path: Path
    name: str
    energy_uj_path: Path
    max_energy_uj: int | None

    def read_uj(self) -> int:
        try:
            return int(self.energy_uj_path.read_text(encoding="utf-8").strip())
        except OSError as exc:
            raise RAPLUnavailable(f"Cannot read {self.energy_uj_path}: {exc}") from exc


def _read_optional_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def detect_rapl(*, prefer_package: bool = True) -> list[RAPLDomain]:
    """Discover readable RAPL domains under powercap.

    Returns an empty list on non-Linux or when sysfs is absent / unreadable
    (no exception — callers treat empty as proxy path).
    """
    if platform.system() != "Linux":
        return []
    if not POWERCAP_ROOT.is_dir():
        return []

    domains: list[RAPLDomain] = []
    for entry in sorted(POWERCAP_ROOT.iterdir()):
        name = entry.name
        if not name.startswith(("intel-rapl:", "amd-rapl:")):
            continue
        energy = entry / "energy_uj"
        if not energy.is_file():
            continue
        label_path = entry / "name"
        try:
            label = label_path.read_text(encoding="utf-8").strip() if label_path.is_file() else name
        except OSError:
            label = name
        try:
            _ = int(energy.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        max_uj = _read_optional_int(entry / "max_energy_range_uj")
        domains.append(
            RAPLDomain(
                path=entry,
                name=label,
                energy_uj_path=energy,
                max_energy_uj=max_uj,
            )
        )

    if prefer_package and domains:
        package = [d for d in domains if d.name.lower() in {"package-0", "package", "psys"}]
        if package:
            rest = [d for d in domains if d not in package]
            return package + rest
    return domains


def _delta_uj(before: int, after: int, max_energy_uj: int | None) -> int:
    """Non-negative microjoule delta; handles counter wrap when max is known.

    Without ``max_energy_range_uj``, a wrap cannot be recovered honestly — raise
    rather than return a negative Joules delta.
    """
    if after >= before:
        return after - before
    if max_energy_uj and max_energy_uj > 0:
        return after + (max_energy_uj - before)
    raise RAPLUnavailable(
        "RAPL energy_uj wrapped but max_energy_range_uj is missing; "
        "refusing negative Joules"
    )


@dataclass
class RAPLMeter:
    """Sample RAPL energy over a wall-clock interval or callable."""

    domains: list[RAPLDomain]

    @classmethod
    def open(cls) -> RAPLMeter:
        domains = detect_rapl()
        if not domains:
            raise RAPLUnavailable(
                "No readable RAPL powercap domains "
                f"(os={platform.system()}, root={POWERCAP_ROOT})"
            )
        return cls(domains=domains)

    def snapshot_uj(self) -> dict[str, int]:
        return {d.name: d.read_uj() for d in self.domains}

    def measure(
        self,
        fn: Callable[[], None] | None = None,
        *,
        sleep_s: float | None = None,
        domain: str | None = None,
    ) -> dict[str, float | str | dict[str, float] | None]:
        """Return Joules (and seconds) for ``domain`` (default: first / package).

        ``avg_power_w`` is ``None`` when ``elapsed_s`` is zero (degenerate interval).
        """
        if fn is None and sleep_s is None:
            raise ValueError("Provide fn= or sleep_s=")
        target = self._pick_domain(domain)
        before = {d.name: d.read_uj() for d in self.domains}
        t0 = time.perf_counter()
        if fn is not None:
            fn()
        if sleep_s is not None and sleep_s > 0:
            time.sleep(sleep_s)
        t1 = time.perf_counter()
        after = {d.name: d.read_uj() for d in self.domains}

        joules_by_domain: dict[str, float] = {}
        for d in self.domains:
            duj = _delta_uj(before[d.name], after[d.name], d.max_energy_uj)
            joules_by_domain[d.name] = duj / 1e6

        elapsed = t1 - t0
        primary_j = joules_by_domain[target.name]
        avg_w: float | None = (primary_j / elapsed) if elapsed > 0 else None
        return {
            "status": "MEASURED_RAPL",
            "domain": target.name,
            "energy_j": primary_j,
            "elapsed_s": elapsed,
            "avg_power_w": avg_w,
            "energy_j_by_domain": joules_by_domain,
            "backend": "linux_powercap",
        }

    def _pick_domain(self, domain: str | None) -> RAPLDomain:
        if domain is None:
            return self.domains[0]
        for d in self.domains:
            if d.name == domain or d.path.name == domain:
                return d
        raise RAPLUnavailable(
            f"RAPL domain {domain!r} not found; have {[d.name for d in self.domains]}"
        )


def measure_rapl_joules(
    fn: Callable[[], None] | None = None,
    *,
    sleep_s: float | None = None,
    domain: str | None = None,
) -> dict[str, float | str | dict[str, float] | None]:
    """Convenience: open meter and measure once."""
    return RAPLMeter.open().measure(fn, sleep_s=sleep_s, domain=domain)
