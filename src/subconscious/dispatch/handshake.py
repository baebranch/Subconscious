"""Client handshake and identity for engine tool dispatch.

`HandshakeService` assigns or validates a `Client_ID` for a connection and
tracks which ids are currently held by active connections, so a conflicting
handshake attempt can be rejected without side effects (Requirement 1.2,
1.3). `release` frees a `Client_ID` when its connection disconnects, and
every subsequent handshake — including one re-presenting the same id after a
reconnect — is a brand-new handshake with no carried-over state
(Requirement 1.5).

`should_accept_frame` is the pure frame-gating helper: it decides whether a
`tool.register`/`chat.send` frame is accepted based only on whether the
connection's handshake has completed (Requirement 1.1).

See design.md's "HandshakeService" section for the authoritative shape.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass
class HandshakeResult:
  ok: bool
  client_id: Optional[str] = None
  reason: Optional[str] = None  # "client_id_in_use" when ok is False


class HandshakeService:
  """Assigns or validates a Client_ID for one connection (Requirement 1)."""

  def __init__(self) -> None:
    self._active_client_ids: set[str] = set()

  def handshake(self, requested_client_id: Optional[str] = None) -> HandshakeResult:
    """Assign a fresh id when none is requested, or validate/reserve the
       requested one. Returns ok=False without mutating state when the
       requested id is already held by another connection (R1.3)."""
    if requested_client_id is not None:
      if requested_client_id in self._active_client_ids:
        return HandshakeResult(ok=False, reason="client_id_in_use")
      self._active_client_ids.add(requested_client_id)
      return HandshakeResult(ok=True, client_id=requested_client_id)

    new_client_id = str(uuid.uuid4())
    self._active_client_ids.add(new_client_id)
    return HandshakeResult(ok=True, client_id=new_client_id)

  def release(self, client_id: str) -> None:
    """Free a Client_ID on disconnect so it can be reused by a later
       handshake (R1.5 treats every reconnect as a brand-new handshake)."""
    self._active_client_ids.discard(client_id)


def should_accept_frame(handshake_complete: bool) -> bool:
  """R1.1: a `tool.register`/`chat.send` frame is accepted iff the
     connection's handshake has already completed. This is a pure function
     of the connection's own handshake state only — it takes no frame
     content, connection kind, or Client_ID into account."""
  return handshake_complete
