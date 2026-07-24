"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from bnn.determinism import set_repro_seed


@pytest.fixture
def seed():
    set_repro_seed(0, deterministic=True, force_cpu=True)
    return 0
