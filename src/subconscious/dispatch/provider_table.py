"""ProviderTable: the Master_Engine's aggregation of every currently-known
Provider's tool ids, keyed by provider_id and queried by tool id.

See design.md's "ProviderTable" section and "Data Models" -> "Provider Table
entry" for the authoritative shape and semantics. Requirements 1.6, 2.1-2.5,
6.1, 8.1, 10.3.
"""

from __future__ import annotations

from typing import Optional

from .provider import Provider, ProviderConnection


class ProviderTable:
  """Aggregates every currently-known Provider's tool ids (Requirement 2, 6, 8.1)."""

  def __init__(self) -> None:
    self._providers: dict[str, Provider] = {}   # keyed by provider_id
    self._next_sequence = 0

  def upsert(
    self,
    provider_id: str,
    kind: str,
    connection: ProviderConnection,
    tool_ids: frozenset[str],
    execution_profile_root: Optional[str] = None,
    tool_descriptors: Optional[dict[str, dict]] = None,
  ) -> None:
    """Replace only this provider_id's prior entry (R2.1, R2.3); an empty
       tool_ids set is equivalent to having no entries for this provider
       (R2.4's "empty tool list" case) since `providers_for` filters on
       membership in `tool_ids`.

       A provider_id's `sequence` is assigned once, from the table's own
       monotonic counter, at its first registration, and is preserved
       across every subsequent re-registration for that same provider_id
       so earliest-registration tie-breaks (R3.4, R3.5) stay stable.

       `tool_descriptors` (id -> {"description", "inputSchema"}) is the
       advertised per-tool metadata from the `tool.register` frame, used to
       build the per-turn bridge tools exposed to the agent. Defaults to an
       empty mapping when omitted."""
    tool_ids = frozenset(tool_ids)
    existing = self._providers.get(provider_id)
    if existing is not None:
      sequence = existing.sequence
    else:
      sequence = self._next_sequence
      self._next_sequence += 1
    self._providers[provider_id] = Provider(
      kind=kind,
      provider_id=provider_id,
      connection=connection,
      tool_ids=tool_ids,
      execution_profile_root=execution_profile_root,
      sequence=sequence,
      tool_descriptors=dict(tool_descriptors or {}),
    )

  def remove(self, provider_id: str) -> None:
    """R1.6/R2.4: remove every entry attributed to provider_id. A tool id
       remains routable afterward iff another provider still offers it."""
    self._providers.pop(provider_id, None)

  def providers_for(self, tool_id: str) -> list[Provider]:
    """Candidates for a tool id, in registration order. Empty when no
       provider currently offers it (feeds the no-provider path, R10.3)."""
    candidates = [p for p in self._providers.values() if tool_id in p.tool_ids]
    candidates.sort(key=lambda p: p.sequence)
    return candidates

  def known_tool_ids(self) -> frozenset[str]:
    """Every tool id currently offered by any Provider in this table. Used
       by the agent-tool bridge (`dispatch/provider_tools.py`) to build one
       agent-callable tool per Provider-advertised tool id, and by callers
       that need to give a connected client's tools precedence over
       same-named local built-in tools."""
    ids: set[str] = set()
    for provider in self._providers.values():
      ids |= provider.tool_ids
    return frozenset(ids)

  def descriptor_for(self, tool_id: str) -> Optional[dict]:
    """The advertised `{"description", "inputSchema"}` descriptor for
       `tool_id`, taken from the earliest-registered Provider that recorded
       one for it (via `tool.register`). Returns None when no currently
       registered Provider offers `tool_id` or none recorded a descriptor
       for it. Used only to build the tool's model-facing schema/description;
       the actual per-call routing decision is made independently by
       `resolve_provider` at call time."""
    for provider in self.providers_for(tool_id):
      descriptor = provider.tool_descriptors.get(tool_id)
      if descriptor:
        return descriptor
    return None

  def snapshot(self) -> "ProviderTable":
    """An immutable-for-this-turn view used as the Routing basis for a
       single chat.send (R5.5) — never merged with another Engine's table.

       Copies the provider mapping and sequence counter so that later
       mutations (upsert/remove) on the live table never become visible
       through a snapshot already taken, and vice versa."""
    snap = ProviderTable()
    snap._providers = dict(self._providers)
    snap._next_sequence = self._next_sequence
    return snap
