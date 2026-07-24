"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def seed():
    torch.manual_seed(0)
    return 0
