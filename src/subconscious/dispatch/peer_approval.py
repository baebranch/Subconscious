"""Peer approval enforcement: the Engine-scoped gate a Peer_Engine applies to
every inbound `tool.call`, independent of who is asking.

See design.md's "PeerApprovalGate" section for the authoritative behavior.
The decision made by `PeerApprovalGate.evaluate` is a pure function of two
things only:

  1. `classify_operation(tool_name)` — the same query/mutation
     classification this Engine already applies to its own agent loop's
     tool calls via `ApprovalRequiredToolset` (R7.2).
  2. This Engine's own locally-configured `PeerApprovalSettings`, obtained
     from `settings_provider` — never from the requesting Master_Engine, and
     never from any "pre-approved" marking on the inbound frame, because no
     such marking is ever read here: `evaluate`'s signature accepts only a
     `tool_name` (R7.3, R7.6).

Requirements: 7.1, 7.2, 7.3, 7.6.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from ..tools import QUERY, classify_operation


@dataclass
class PeerApprovalSettings:
  """This Engine's own gate for tool calls it is asked to execute on behalf
     of a Peer_Engine — configured locally, never influenced by the
     requesting Master_Engine (R7.6)."""

  query_requires_approval: bool = False
  mutation_requires_approval: bool = True


class ApprovalDeclinedError(Exception):
  """Raised by `PeerApprovalGate.evaluate` when a tool call must not
     execute (R7.4)."""


class PeerApprovalGate:
  """Evaluates an inbound peer `tool.call` against this Engine's own
     approval policy (Requirement 7)."""

  def __init__(
    self,
    settings_provider: Callable[[], PeerApprovalSettings],
    prompt: Callable[[str], "asyncio.Future[bool]"],
  ) -> None:
    self._settings_provider = settings_provider
    self._prompt = prompt

  async def evaluate(self, tool_name: str) -> None:
    """Raises `ApprovalDeclinedError` when the call must not execute.

    The decision depends only on `tool_name` (via `classify_operation`) and
    this Engine's own settings (via `settings_provider`) — the same
    evaluation applies whether the call originated from this Engine's own
    local turn handling or from a Peer_Engine's request (R7.1, R7.2). Any
    "pre-approved" marking on an inbound frame is never read: it is not part
    of this method's signature (R7.3)."""
    settings = self._settings_provider()
    operation = classify_operation(tool_name)
    requires_approval = (
      settings.query_requires_approval if operation == QUERY else settings.mutation_requires_approval
    )
    if not requires_approval:
      return
    approved = await self._prompt(tool_name)
    if not approved:
      raise ApprovalDeclinedError(tool_name)
