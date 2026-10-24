"""Inbound peer `tool.call` gateway: composes `PeerApprovalGate.evaluate`
with `ToolDispatcher.forward_call` to handle a `tool.call` frame arriving
from a Master_Engine over the (future) engine-to-engine transport.

See design.md's "PeerApprovalGate" section and the "peer engine as provider,
approval enforced locally" sequence diagram for the authoritative flow: an
inbound `tool.call` is routed through `PeerApprovalGate.evaluate` before it
ever reaches this Engine's own `ToolDispatcher`. A decline produces a
`tool.result`-shaped rejection without invoking the tool at all (R7.4). An
approval forwards the call to this Engine's own `ProviderTable` via
`ToolDispatcher.forward_call`, reusing — never minting — the inbound
correlation id (R7.5), using the same frame fields and correlation matching
regardless of Provider kind (R6.2, R8.2, R8.4). A Peer_Engine only ever
receives `tool.call` frames for a turn it was asked to assist with, never a
`chat.send` (R6.4) — this module's only inbound concern is the former.

The concrete engine-to-engine transport is out of scope for this spec; this
module is written against the `PeerEngineTransport` Protocol in
`dispatch/provider.py` and is exercised in tests only via an in-memory fake
satisfying that Protocol.

Requirements: 6.2, 6.4, 7.4, 7.5, 8.2, 8.4.
"""

from __future__ import annotations

from typing import Optional

from .dispatcher import DispatchOutcome, ToolDispatcher
from .peer_approval import ApprovalDeclinedError, PeerApprovalGate

REJECTED_ERROR_KIND = "rejected"


async def handle_peer_tool_call(
  correlation_id: str,
  tool_id: str,
  tool_name: str,
  input: object,
  routing_key: Optional[str],
  approval_gate: PeerApprovalGate,
  dispatcher: ToolDispatcher,
) -> DispatchOutcome:
  """Handle one inbound `tool.call` from a Master_Engine and return the
     `DispatchOutcome` to be sent back as its `tool.result`.

     On decline (R7.4): `dispatcher` is never invoked — the underlying tool
     is never called — and the returned `DispatchOutcome` carries a
     rejection error under `correlation_id`.

     On approval: forwards to this Engine's own `ProviderTable` via
     `dispatcher.forward_call`, reusing `correlation_id` unchanged rather
     than minting a new one (R7.5). Any `NoProviderError` `forward_call`
     raises propagates uncaught, to be handled the same way a Master_Engine
     handles it for its own local turn handling."""
  try:
    await approval_gate.evaluate(tool_name)
  except ApprovalDeclinedError:
    return DispatchOutcome(ok=False, error={"kind": REJECTED_ERROR_KIND, "tool_name": tool_name})

  return await dispatcher.forward_call(tool_id, input, routing_key, correlation_id)
