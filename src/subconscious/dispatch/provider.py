"""Provider abstraction for engine tool dispatch.

A `Provider` is a dispatch-table entry that is either a `LocalClientProvider`
(a connected, handshaken Client over loopback) or a `PeerEngineProvider` (a
paired Engine reachable over the future engine-to-engine transport). Both
expose the same narrow surface a dispatcher needs: a set of tool ids, an
optional Execution_Profile root, a registration sequence number, and a way to
send a `tool.call` and receive its `tool.result`.

See design.md's "Provider abstraction" section and "Data Models" ->
"Provider Table entry" for the authoritative shape. Requirement 8.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol


class ProviderConnection(Protocol):
  """Narrow send/receive surface a dispatcher needs from any Provider,
     regardless of whether it is backed by a loopback WebSocket (Local_Client)
     or a future engine-to-engine transport (Peer_Engine). Requirement 8."""

  async def send_tool_call(self, correlation_id: str, tool_id: str, input: object) -> None:
    ...

  @property
  def is_available(self) -> bool:
    ...


class PeerEngineTransport(ProviderConnection, Protocol):
  """The Protocol a future engine-to-engine transport implements; it
     satisfies `ProviderConnection` plus availability-change notifications
     consumed by `PeerLifecycleManager`."""

  def on_availability_changed(self, callback: Callable[[bool], None]) -> None:
    """Invoke callback(True/False) when the pairing connection becomes
       available/unavailable, driving PeerLifecycleManager's grace period."""
    ...

  def advertised_tool_ids(self) -> frozenset[str]:
    ...


@dataclass
class Provider:
  """A single Provider_Table entry (design.md "Data Models" ->
     "Provider Table entry"). Not serialized over the wire."""

  kind: str                        # "local_client" | "peer_engine"
  provider_id: str                 # Client_ID or peer Engine node_id
  connection: ProviderConnection
  tool_ids: frozenset[str]
  execution_profile_root: Optional[str]
  sequence: int                    # monotonically increasing registration order, for tie-breaks
  # Per-tool id -> {"description": str, "inputSchema": dict} as advertised in the
  # `tool.register` frame's `RegisteredToolDescriptor` entries. Used to build the
  # per-turn agent tools that bridge a Provider's tools into the model's toolset
  # (single-engine-architecture "provider tool bridge"). Empty for a Provider whose
  # descriptors are unknown/not yet advertised (e.g. before the first tool.register).
  tool_descriptors: dict[str, dict] = field(default_factory=dict)
