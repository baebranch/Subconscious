"""Unit tests for Peer_Engine grace-period lifecycle (dispatch/peer_lifecycle.py).

Covers PeerLifecycleManager.on_unavailable scheduling a delayed removal,
on_available cancelling a pending removal on timely recovery, and removal
actually happening when no recovery occurs before the grace period elapses.

Validates: Requirements 6.3
"""

import asyncio

import pytest

from subconscious.dispatch.peer_lifecycle import PeerLifecycleManager
from subconscious.dispatch.provider_table import ProviderTable

GRACE_PERIOD_S = 0.05


class _FakeConnection:
    """Minimal fake satisfying the ProviderConnection surface for tests."""

    def __init__(self, available: bool = True) -> None:
        self._available = available

    async def send_tool_call(self, correlation_id: str, tool_id: str, input: object) -> None:
        raise AssertionError("not expected to be called in these tests")

    @property
    def is_available(self) -> bool:
        return self._available


class TestOnUnavailableSchedulesRemoval:
    async def test_removal_happens_after_grace_period_elapses(self) -> None:
        table = ProviderTable()
        table.upsert("peer-1", "peer_engine", _FakeConnection(), frozenset({"tool_a"}))
        manager = PeerLifecycleManager(table, GRACE_PERIOD_S)

        manager.on_unavailable("peer-1")
        assert table.providers_for("tool_a") != []

        await asyncio.sleep(GRACE_PERIOD_S * 3)

        assert table.providers_for("tool_a") == []
        assert manager.has_pending_removal("peer-1") is False

    async def test_pending_removal_is_reported_while_timer_is_running(self) -> None:
        table = ProviderTable()
        table.upsert("peer-1", "peer_engine", _FakeConnection(), frozenset({"tool_a"}))
        manager = PeerLifecycleManager(table, GRACE_PERIOD_S)

        manager.on_unavailable("peer-1")
        assert manager.has_pending_removal("peer-1") is True

        await asyncio.sleep(GRACE_PERIOD_S * 3)
        assert manager.has_pending_removal("peer-1") is False


class TestOnAvailableCancelsPendingRemoval:
    async def test_recovery_before_grace_period_cancels_removal(self) -> None:
        table = ProviderTable()
        connection = _FakeConnection()
        table.upsert("peer-1", "peer_engine", connection, frozenset({"tool_a"}))
        manager = PeerLifecycleManager(table, GRACE_PERIOD_S)

        manager.on_unavailable("peer-1")
        manager.on_available("peer-1", "peer_engine", connection, frozenset({"tool_a"}))

        # Even after the original grace period would have elapsed, the
        # provider must still be present because the removal was cancelled.
        await asyncio.sleep(GRACE_PERIOD_S * 3)

        assert [p.provider_id for p in table.providers_for("tool_a")] == ["peer-1"]
        assert manager.has_pending_removal("peer-1") is False

    async def test_on_available_registers_provider_with_no_prior_removal(self) -> None:
        table = ProviderTable()
        connection = _FakeConnection()
        manager = PeerLifecycleManager(table, GRACE_PERIOD_S)

        # No on_unavailable was ever called for this peer — on_available
        # must still work as a plain (re-)registration.
        manager.on_available("peer-1", "peer_engine", connection, frozenset({"tool_a"}))

        assert [p.provider_id for p in table.providers_for("tool_a")] == ["peer-1"]

    async def test_on_available_preserves_execution_profile_root(self) -> None:
        table = ProviderTable()
        connection = _FakeConnection()
        manager = PeerLifecycleManager(table, GRACE_PERIOD_S)

        manager.on_available(
            "peer-1", "peer_engine", connection, frozenset({"tool_a"}), execution_profile_root="/proj-a"
        )

        [provider] = table.providers_for("tool_a")
        assert provider.execution_profile_root == "/proj-a"


class TestReRegistrationReplacesRemoval:
    async def test_second_on_unavailable_replaces_first_timer(self) -> None:
        table = ProviderTable()
        table.upsert("peer-1", "peer_engine", _FakeConnection(), frozenset({"tool_a"}))
        manager = PeerLifecycleManager(table, GRACE_PERIOD_S)

        manager.on_unavailable("peer-1")
        await asyncio.sleep(GRACE_PERIOD_S * 0.5)
        manager.on_unavailable("peer-1")  # restart the timer

        # Original timer would have fired by now had it not been replaced.
        await asyncio.sleep(GRACE_PERIOD_S * 0.7)
        assert table.providers_for("tool_a") != []

        await asyncio.sleep(GRACE_PERIOD_S * 3)
        assert table.providers_for("tool_a") == []


class TestUnrelatedPeersAreIndependent:
    async def test_unavailable_for_one_peer_does_not_affect_another(self) -> None:
        table = ProviderTable()
        table.upsert("peer-1", "peer_engine", _FakeConnection(), frozenset({"tool_a"}))
        table.upsert("peer-2", "peer_engine", _FakeConnection(), frozenset({"tool_b"}))
        manager = PeerLifecycleManager(table, GRACE_PERIOD_S)

        manager.on_unavailable("peer-1")
        await asyncio.sleep(GRACE_PERIOD_S * 3)

        assert table.providers_for("tool_a") == []
        assert [p.provider_id for p in table.providers_for("tool_b")] == ["peer-2"]


class TestCancelAll:
    async def test_cancel_all_prevents_pending_removals_from_firing(self) -> None:
        table = ProviderTable()
        table.upsert("peer-1", "peer_engine", _FakeConnection(), frozenset({"tool_a"}))
        table.upsert("peer-2", "peer_engine", _FakeConnection(), frozenset({"tool_b"}))
        manager = PeerLifecycleManager(table, GRACE_PERIOD_S)

        manager.on_unavailable("peer-1")
        manager.on_unavailable("peer-2")
        manager.cancel_all()

        await asyncio.sleep(GRACE_PERIOD_S * 3)

        assert table.providers_for("tool_a") != []
        assert table.providers_for("tool_b") != []
        assert manager.has_pending_removal("peer-1") is False
        assert manager.has_pending_removal("peer-2") is False
