"""Unit tests for Claim resolution (dispatch/claim.py).

Covers resolve_provider's 0/1/>1 candidate branches, Claim matching against
the Routing_Key, tie-breaking by lowest `sequence`, and the no-match
fallback to lowest `sequence` among all candidates.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 9.1, 9.2
"""

from subconscious.dispatch.claim import ResolutionResult, resolve_provider
from subconscious.dispatch.provider import Provider


def _provider(provider_id: str, sequence: int, execution_profile_root: str | None,
              kind: str = "local_client") -> Provider:
    return Provider(
        kind=kind,
        provider_id=provider_id,
        connection=None,  # type: ignore[arg-type]
        tool_ids=frozenset({"x"}),
        execution_profile_root=execution_profile_root,
        sequence=sequence,
    )


class TestZeroCandidates:
    def test_no_candidates_returns_none_without_claim_evaluation(self) -> None:
        result = resolve_provider([], None)
        assert result == ResolutionResult(provider=None, claim_evaluated=False)

    def test_no_candidates_ignores_routing_key(self) -> None:
        result = resolve_provider([], "/some/root")
        assert result.provider is None
        assert result.claim_evaluated is False


class TestSingleCandidate:
    def test_single_candidate_wins_without_claim_evaluation(self) -> None:
        p = _provider("a", 0, "/proj-a")
        result = resolve_provider([p], "/does-not-matter")
        assert result.provider is p
        assert result.claim_evaluated is False

    def test_single_candidate_wins_even_with_no_routing_key(self) -> None:
        # Requirement 9.2: no Execution_Profile / Claim evaluation required
        # when a Provider is the only candidate for its tool id.
        p = _provider("a", 0, None)
        result = resolve_provider([p], None)
        assert result.provider is p
        assert result.claim_evaluated is False


class TestMultipleCandidatesExactMatch:
    def test_exactly_one_match_wins(self) -> None:
        a = _provider("a", 0, "/proj-a")
        b = _provider("b", 1, "/proj-b")
        result = resolve_provider([a, b], "/proj-b")
        assert result.provider is b
        assert result.claim_evaluated is True


class TestMultipleCandidatesTieBreak:
    def test_multiple_matches_tie_broken_by_lowest_sequence(self) -> None:
        a = _provider("a", 2, "/proj-a")
        b = _provider("b", 0, "/proj-a")
        c = _provider("c", 1, "/proj-a")
        result = resolve_provider([a, b, c], "/proj-a")
        assert result.provider is b
        assert result.claim_evaluated is True


class TestMultipleCandidatesNoMatchFallback:
    def test_no_match_falls_back_to_lowest_sequence_among_all(self) -> None:
        a = _provider("a", 2, "/proj-a")
        b = _provider("b", 0, "/proj-b")
        c = _provider("c", 1, "/proj-c")
        result = resolve_provider([a, b, c], "/does-not-exist")
        assert result.provider is b
        assert result.claim_evaluated is True

    def test_no_match_with_none_routing_key_falls_back(self) -> None:
        a = _provider("a", 1, "/proj-a")
        b = _provider("b", 0, "/proj-b")
        result = resolve_provider([a, b], None)
        assert result.provider is b
        assert result.claim_evaluated is True


class TestProviderKindIndependence:
    def test_resolution_is_the_same_regardless_of_provider_kind(self) -> None:
        # Requirement 3.6: same rule regardless of Local_Client vs Peer_Engine
        # candidates, or a mixture of both.
        a = _provider("a", 0, "/proj-a", kind="local_client")
        b = _provider("b", 1, "/proj-b", kind="peer_engine")
        result = resolve_provider([a, b], "/proj-b")
        assert result.provider is b
        assert result.claim_evaluated is True

        result_swapped_kinds = resolve_provider(
            [_provider("a", 0, "/proj-a", kind="peer_engine"),
             _provider("b", 1, "/proj-b", kind="local_client")],
            "/proj-b",
        )
        assert result_swapped_kinds.provider.provider_id == "b"
        assert result_swapped_kinds.claim_evaluated is True


class TestSingleProviderInvariant:
    def test_never_returns_more_than_one_provider(self) -> None:
        # Requirement 3.7: resolve_provider itself never returns more than
        # one Provider - the return type is a single Optional[Provider].
        candidates = [_provider(str(i), i, "/proj") for i in range(5)]
        result = resolve_provider(candidates, "/proj")
        assert not isinstance(result.provider, list)
        assert result.provider in candidates
