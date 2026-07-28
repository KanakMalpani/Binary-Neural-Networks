"""Shared pytest fixtures.

Also enforces the rule that the *fast* suite never touches the network. A
truncated CIFAR download on a macOS runner is what last turned CI red; keeping
that class of failure out of the default job deserves an explicit guard rather
than a convention people forget.

Tests that legitimately need the network must be marked ``slow`` or ``hf``, or
request the ``allow_network`` fixture.
"""

from __future__ import annotations

import socket

import pytest

from bnn.determinism import set_repro_seed

_REAL_SOCKET = socket.socket
_REAL_CREATE_CONNECTION = socket.create_connection


@pytest.fixture
def seed():
    set_repro_seed(0, deterministic=True, force_cpu=True)
    return 0


class NetworkUseInFastTest(RuntimeError):
    """A fast test tried to open a socket."""


@pytest.fixture
def allow_network(monkeypatch):
    """Opt back into real sockets for a test that genuinely needs them."""
    monkeypatch.setattr(socket, "socket", _REAL_SOCKET)
    monkeypatch.setattr(socket, "create_connection", _REAL_CREATE_CONNECTION)
    return True


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    """Fail fast if a non-slow, non-hf test opens a socket."""
    if request.node.get_closest_marker("slow") or request.node.get_closest_marker("hf"):
        return
    if "allow_network" in request.fixturenames:
        return

    def _deny(*args, **kwargs):
        raise NetworkUseInFastTest(
            "Fast tests must not use the network — it makes CI flaky "
            "(a truncated CIFAR download is exactly how this suite last broke). "
            "Use synthetic data, or mark the test @pytest.mark.slow."
        )

    monkeypatch.setattr(socket, "socket", _deny)
    monkeypatch.setattr(socket, "create_connection", _deny)
