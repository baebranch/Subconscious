"""Connection classification for engine tool dispatch.

`classify_connection` determines whether a connection is a Local_Client or a
Remote_Client purely from the transport's own peer address — never from any
value the connection self-reports (Requirement 4.4). `events_ws` calls this
once per connection, before the handshake, using the peer address read from
the ASGI/WebSocket connection itself (e.g. `websocket.client.host`).

`should_reject_tool_register` is the pure predicate a caller uses to reject
`tool.register` frames from `REMOTE_CLIENT` connections before any
`ProviderTable` mutation is attempted (Requirement 4.2, 4.3): a Remote_Client
connection must never be passed to `ProviderTable.upsert` under any
circumstance.

See design.md's "ConnectionClassifier" section for the authoritative shape.
"""

from __future__ import annotations

from enum import Enum

_LOOPBACK_HOSTS = ("127.0.0.1", "::1")


class ConnectionKind(str, Enum):
  LOCAL_CLIENT = "local_client"
  REMOTE_CLIENT = "remote_client"


def classify_connection(peer_host: str) -> ConnectionKind:
  """R4.4: classification is derived from the transport (loopback address)
     — never from a client-supplied value. `peer_host` is read from the
     ASGI/WebSocket connection's own peer address, e.g.
     `websocket.client.host`."""
  return ConnectionKind.LOCAL_CLIENT if peer_host in _LOOPBACK_HOSTS else ConnectionKind.REMOTE_CLIENT


def should_reject_tool_register(connection_kind: ConnectionKind) -> bool:
  """R4.2/R4.3: a pure predicate used to reject `tool.register` frames from
     `REMOTE_CLIENT` connections before any `ProviderTable` mutation is
     attempted. Returns True iff the connection must never become a
     Provider."""
  return connection_kind == ConnectionKind.REMOTE_CLIENT
