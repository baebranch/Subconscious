"""ToolDispatcher: owns correlation ids and the send/await lifecycle for a
single tool call, used identically by a Master_Engine dispatching its own
turn's tool calls (`call`) and by a Peer_Engine forwarding an approved
inbound call to one of its own Providers (`forward_call`).

See design.md's "ToolDispatcher" section for the authoritative shape.
`ToolDispatcher` is deliberately the *only* place that knows about
correlation ids and providers; both entry points resolve exactly one
Provider via `resolve_provider` (R3.7) and short-circuit via
`NoProviderError` when `providers_for(tool_id)` is empty, without ever
invoking `resolve_provider` in that case (R10.3).

Requirements: 3.7, 6.2, 7.5, 8.2, 8.3, 8.4, 10.3.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Optional

from .claim import resolve_provider
from .provider_table import ProviderTable


class NoProviderError(Exception):
  """Raised when providers_for(tool_id) is empty (R10.3) — Claim resolution
     is never attempted in this case."""


@dataclass
class DispatchOutcome:
  ok: bool
  output: Optional[object] = None
  error: Optional[object] = None


class ToolDispatcher:
  """Resolves a tool id to exactly one Provider and manages the
     correlation-id-keyed send/await lifecycle for its `tool.call` /
     `tool.result` round trip (Requirement 8)."""

  def __init__(self, provider_table: ProviderTable) -> None:
    self._table = provider_table
    self._pending: dict[str, "asyncio.Future[DispatchOutcome]"] = {}

  async def call(
    self,
    tool_id: str,
    input: object,
    routing_key: Optional[str],
  ) -> DispatchOutcome:
    """Dispatch a locally-originated tool call, minting a fresh correlation
       id for it. Resolves exactly one Provider and dispatches to it
       (R3.7). Raises `NoProviderError` when no candidate exists for
       `tool_id` (R10.3) — `resolve_provider` is never invoked in that
       case."""
    correlation_id = str(uuid.uuid4())
    return await self._dispatch(tool_id, input, routing_key, correlation_id)

  async def forward_call(
    self,
    tool_id: str,
    input: object,
    routing_key: Optional[str],
    correlation_id: str,
  ) -> DispatchOutcome:
    """Dispatch a peer-forwarded tool call, reusing the caller-supplied
       `correlation_id` end to end rather than minting a new one (R7.5) —
       used when a Peer_Engine forwards an approved inbound `tool.call` to
       one of its own connected Providers. Otherwise identical to `call`:
       resolves exactly one Provider (R3.7) and raises `NoProviderError`
       without invoking `resolve_provider` when there are no candidates
       (R10.3)."""
    return await self._dispatch(tool_id, input, routing_key, correlation_id)

  async def _dispatch(
    self,
    tool_id: str,
    input: object,
    routing_key: Optional[str],
    correlation_id: str,
  ) -> DispatchOutcome:
    candidates = self._table.providers_for(tool_id)
    if not candidates:
      raise NoProviderError(tool_id)
    result = resolve_provider(candidates, routing_key)
    provider = result.provider
    assert provider is not None  # non-empty candidates always resolve to one provider

    fut: "asyncio.Future[DispatchOutcome]" = asyncio.get_running_loop().create_future()
    self._pending[correlation_id] = fut
    try:
      await provider.connection.send_tool_call(correlation_id, tool_id, input)
      return await fut
    finally:
      self._pending.pop(correlation_id, None)

  def resolve_result(self, correlation_id: str, outcome: DispatchOutcome) -> None:
    """Called when a `tool.result` frame arrives from ANY provider kind,
       using the same correlation-id matching regardless of kind (R6.2,
       R8.2, R8.4). A `correlation_id` with no matching pending call (e.g.
       already resolved, or unknown) is a no-op."""
    fut = self._pending.get(correlation_id)
    if fut is not None and not fut.done():
      fut.set_result(outcome)
