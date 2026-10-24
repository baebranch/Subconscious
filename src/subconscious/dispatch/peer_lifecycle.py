"""Peer_Engine grace-period lifecycle management for engine tool dispatch.

`PeerLifecycleManager` implements the Peer_Engine grace period described in
design.md's "PeerLifecycleManager" section: when a Peer_Engine's pairing
connection becomes unavailable, its tool ids stay in the `ProviderTable` for
a configured grace period before being removed, so a brief reconnect never
causes tool ids to flicker out of routing consideration. If the pairing
connection recovers before the grace period elapses, the pending removal is
cancelled and the Peer_Engine's entry is (re-)registered instead.

`grace_period_s` mirrors the design's `Config.peer_grace_period_seconds`
(default `30.0`, added by task 9.1); until that config field lands, callers
pass the grace period directly to the constructor.

Requirements: 6.3.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from .provider import ProviderConnection
from .provider_table import ProviderTable


class PeerLifecycleManager:
  """Schedules a Peer_Engine's tool ids for removal from the Provider_Table
     after a grace period once its pairing connection becomes unavailable,
     and cancels that pending removal if the connection recovers before the
     grace period elapses (Requirement 6.3)."""

  def __init__(self, provider_table: ProviderTable, grace_period_s: float) -> None:
    self._table = provider_table
    self._grace_period_s = grace_period_s
    self._pending_removals: dict[str, "asyncio.Task[None]"] = {}

  def on_unavailable(self, peer_id: str) -> None:
    """Schedule removal of peer_id's Provider_Table entries after
       grace_period_s. A subsequent `on_available` for the same peer_id
       before the timer fires cancels this removal (R6.3). If a removal is
       already pending for peer_id, it is replaced — cancelled and
       rescheduled — rather than allowed to accumulate a second timer."""
    self._cancel_pending(peer_id)
    task = asyncio.ensure_future(self._remove_after_delay(peer_id))
    self._pending_removals[peer_id] = task

  def on_available(
    self,
    peer_id: str,
    kind: str,
    connection: ProviderConnection,
    tool_ids: frozenset[str],
    execution_profile_root: Optional[str] = None,
  ) -> None:
    """Cancel any pending removal for peer_id and (re-)register its
       Provider_Table entry — the "recovers within that grace period" case
       of R6.3. Safe to call even when no removal is pending (e.g. the
       Peer_Engine's very first registration)."""
    self._cancel_pending(peer_id)
    self._table.upsert(peer_id, kind, connection, tool_ids, execution_profile_root)

  def has_pending_removal(self, peer_id: str) -> bool:
    """Whether a removal is currently scheduled (not yet fired or
       cancelled) for peer_id. Exposed for observability/testing."""
    task = self._pending_removals.get(peer_id)
    return task is not None and not task.done()

  def cancel_all(self) -> None:
    """Cancel every pending removal without touching the Provider_Table,
       e.g. on shutdown."""
    for peer_id in list(self._pending_removals):
      self._cancel_pending(peer_id)

  def _cancel_pending(self, peer_id: str) -> None:
    task = self._pending_removals.pop(peer_id, None)
    if task is not None:
      task.cancel()

  async def _remove_after_delay(self, peer_id: str) -> None:
    try:
      await asyncio.sleep(self._grace_period_s)
    except asyncio.CancelledError:
      raise
    self._pending_removals.pop(peer_id, None)
    self._table.remove(peer_id)
