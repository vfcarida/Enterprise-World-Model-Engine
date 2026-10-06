"""Hermetic execution tests: frozen clock, blocked sockets, and filesystem isolation.

In accordance with R03 Task 7:
- time-machine: verifies frozen clock determinism across timestamps and metadata
- pytest-socket: verifies socket blocking (--disable-socket --allow-hosts=127.0.0.1)
- tmp_path: enforces filesystem isolation for all test artifacts
"""

from __future__ import annotations

import datetime
import socket
import time
from pathlib import Path

import pytest
import time_machine

from ewm_engine.cards.models import ScenarioCard


def test_frozen_clock_determinism() -> None:
    """Verify that time-machine freezes datetime and produces identical deterministic timestamps."""
    frozen_point = "2026-10-06 12:00:00 +00:00"

    with time_machine.travel(frozen_point, tick=False):
        t1 = datetime.datetime.now(datetime.UTC)
        epoch1 = time.time()
        time.sleep(0.01)  # Clock is frozen, sleep does not advance time
        t2 = datetime.datetime.now(datetime.UTC)
        epoch2 = time.time()

        assert t1 == t2
        assert epoch1 == epoch2
        assert t1.isoformat().startswith("2026-10-06T12:00:00")


def test_hermetic_socket_blocked_on_external_network() -> None:
    """Verify that external socket connections are blocked by pytest-socket.

    Any accidental external network call in the test suite raises a socket error.
    """
    try:
        from pytest_socket import (
            SocketBlockedError,
            SocketConnectBlockedError,
            disable_socket,
            enable_socket,
        )
    except ImportError:
        pytest.skip("pytest-socket not available")

    disable_socket()
    try:
        with pytest.raises((SocketConnectBlockedError, SocketBlockedError, OSError)):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.1)
            s.connect(("8.8.8.8", 53))
    finally:
        enable_socket()


def test_hermetic_filesystem_isolation(tmp_path: Path) -> None:
    """Verify tests write strictly within isolated tmp_path without leaking into workspace."""
    test_file = tmp_path / "sandbox_audit.json"
    assert not test_file.exists()

    card = ScenarioCard(
        artifact_fingerprint="sha256:test_fingerprint_tmp",
        scenario_id="scen_isolated_tmp",
        description="Testing filesystem isolation",
        horizon=10,
        samples=1,
        seed=42,
    )

    test_file.write_text(card.to_json(), encoding="utf-8")
    assert test_file.exists()
    assert "test_fingerprint_tmp" in test_file.read_text(encoding="utf-8")
    assert tmp_path != Path.cwd()
