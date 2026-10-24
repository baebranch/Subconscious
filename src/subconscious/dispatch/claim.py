"""Claim resolution: the pure, deterministic routing algorithm used to pick
exactly one Provider for a tool id when more than one is registered for it.

See design.md's "ClaimResolver" section for the authoritative behavior:

  0 candidates  -> ResolutionResult(None, claim_evaluated=False)
  1 candidate   -> that candidate, no Claim evaluated               (R3.1)
  >1 candidates -> evaluate a Claim per candidate: does
                    candidate.execution_profile_root == routing_key? (R3.2)
                    - exactly one match  -> that candidate            (R3.3)
                    - multiple matches   -> lowest `sequence` among
                                            the matches                (R3.4)
                    - zero matches       -> lowest `sequence` among
                                            ALL candidates             (R3.5)

The same function runs unmodified whether every candidate is a
LocalClientProvider, every candidate is a PeerEngineProvider, or the list
mixes both kinds (R3.6). The caller (ToolDispatcher) is the one invariant
that guarantees a single Provider is ever dispatched to (R3.7) —
`resolve_provider` itself never returns more than one.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 9.1, 9.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .provider import Provider


@dataclass
class ResolutionResult:
  """The outcome of resolving a tool id's candidate Providers to exactly one
     Provider (or none, when there were no candidates)."""

  provider: Optional[Provider]      # None iff candidates was empty
  claim_evaluated: bool             # False when len(candidates) <= 1 (R3.1)


def resolve_provider(candidates: list[Provider], routing_key: Optional[str]) -> ResolutionResult:
  """Resolve a list of candidate Providers for a single tool id to exactly
     one Provider, per Requirement 3 (see module docstring for the full
     rule)."""
  if not candidates:
    return ResolutionResult(provider=None, claim_evaluated=False)

  if len(candidates) == 1:
    return ResolutionResult(provider=candidates[0], claim_evaluated=False)

  matches = [c for c in candidates if c.execution_profile_root == routing_key]
  pool = matches if matches else candidates
  winner = min(pool, key=lambda c: c.sequence)
  return ResolutionResult(provider=winner, claim_evaluated=True)
