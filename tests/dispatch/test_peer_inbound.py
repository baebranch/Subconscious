"""Unit tests for the inbound peer `tool.call` gateway (dispatch/peer_inbound.py).

Covers `handle_peer_tool_call` composing `PeerApprovalGate.evaluate` with
`ToolDispatcher.forward_call`: a decline never invokes the dispatcher (the
tool is never called) and yields a rejection outcome under the inbound
correlation id; an approval forwards to the Engine's own `ProviderTable`
via `forward_call`, reusing (never minting) the inbound correlation id.

Exercised against the `PeerEngineTransport` Protocol via an in-memory fake,
since the concrete transport is out of scope for this spec.

Validates: Requirements 6.2, 6.4, 7.4, 7.5, 8.2, 8.4
"""

from __future__ import annotations

import asyncio

import pytest

from subconscious.dispatch.dispatcher import DispatchOutcome, ToolDispatcher
from subconscious.dispatch.peer_approval import PeerApprovalGate, PeerApprovalSettings
from subconscious.dispatch.peer_inbound import REJECTED_ERROR_KIND, handle_peer_tool_call
from subconscious.dispatch.provider_table import ProviderTable


class FakePeerEngineTransport:
  """An in-memory fake satisfying the `PeerEngineTransport`/`ProviderConnection`
     surface: records every outbound `tool.call` it is asked to send. Stands
     in for the concrete (out-of-scope) engine-to-engine transport."""

  def __init__(self) -> None:
    self.sent: list[tuple[str, str, object]] = []

  async def send_tool_call(self, correlation_id: str, tool_id: str, input: object) -> None:
    self.sent.append((correlation_id, tool_id, input))

  @property
  def is_available(self) -> bool:
    return True


def _gate(*, mutation_requires_approval: bool, approved: bool = True) -> PeerApprovalGate:
  settings = PeerApprovalSettings(
    query_requires_approval=False,
    mutation_requires_approval=mutation_requires_approval,
  )

  async def prompt(tool_name: str) -> bool:
    return approved

  return PeerApprovalGate(lambda: settings, prompt)


class TestDeclineNeverInvokesDispatcher:
  @pytest.mark.asyncio
  async def test_decline_returns_rejection_without_calling_dispatcher(self) -> None:
    connection = FakePeerEngineTransport()
    table = ProviderTable()
    table.upsert("local-1", "local_client", connection, frozenset({"delete_note"}))
    dispatcher = ToolDispatcher(table)
    gate = _gate(mutation_requires_approval=True, approved=False)

    outcome = await handle_peer_tool_call(
      correlation_id="corr-peer-1",
      tool_id="delete_note",
      tool_name="delete_note",
      input={"id": 1},
      routing_key=None,
      approval_gate=gate,
      dispatcher=dispatcher,
    )

    assert outcome.ok is False
    assert outcome.error["kind"] == REJECTED_ERROR_KIND
    # The underlying tool is never invoked: nothing was forwarded downstream.
    assert connection.sent == []

  @pytest.mark.asyncio
  async def test_query_tool_needs_no_approval_and_is_forwarded(self) -> None:
    connection = FakePeerEngineTransport()
    table = ProviderTable()
    table.upsert("local-1", "local_client", connection, frozenset({"list_notes"}))
    dispatcher = ToolDispatcher(table)
    gate = _gate(mutation_requires_approval=True, approved=False)  # would decline mutations

    task = asyncio.ensure_future(
      handle_peer_tool_call(
        correlation_id="corr-peer-2",
        tool_id="list_notes",
        tool_name="list_notes",
        input={},
        routing_key=None,
        approval_gate=gate,
        dispatcher=dispatcher,
      )
    )
    await asyncio.sleep(0)

    assert len(connection.sent) == 1
    correlation_id, tool_id, _ = connection.sent[0]
    assert correlation_id == "corr-peer-2"
    assert tool_id == "list_notes"

    dispatcher.resolve_result("corr-peer-2", DispatchOutcome(ok=True, output=["a"]))
    outcome = await task
    assert outcome == DispatchOutcome(ok=True, output=["a"])


class TestApprovalForwardsAndReusesCorrelationId:
  @pytest.mark.asyncio
  async def test_approval_forwards_to_own_provider_table_reusing_correlation_id(self) -> None:
    connection = FakePeerEngineTransport()
    table = ProviderTable()
    table.upsert("local-1", "local_client", connection, frozenset({"create_file"}))
    dispatcher = ToolDispatcher(table)
    gate = _gate(mutation_requires_approval=True, approved=True)

    inbound_correlation_id = "peer-supplied-corr-42"
    task = asyncio.ensure_future(
      handle_peer_tool_call(
        correlation_id=inbound_correlation_id,
        tool_id="create_file",
        tool_name="create_file",
        input={"path": "a.py"},
        routing_key=None,
        approval_gate=gate,
        dispatcher=dispatcher,
      )
    )
    await asyncio.sleep(0)

    # Forwarded downstream via this Engine's own ProviderTable/ToolDispatcher.
    assert len(connection.sent) == 1
    correlation_id, tool_id, input_ = connection.sent[0]
    assert correlation_id == inbound_correlation_id  # reused, never minted
    assert tool_id == "create_file"
    assert input_ == {"path": "a.py"}

    dispatcher.resolve_result(inbound_correlation_id, DispatchOutcome(ok=True, output="done"))
    outcome = await task
    assert outcome == DispatchOutcome(ok=True, output="done")

  @pytest.mark.asyncio
  async def test_approved_mutation_is_forwarded(self) -> None:
    connection = FakePeerEngineTransport()
    table = ProviderTable()
    table.upsert("local-1", "local_client", connection, frozenset({"delete_note"}))
    dispatcher = ToolDispatcher(table)
    gate = _gate(mutation_requires_approval=True, approved=True)

    task = asyncio.ensure_future(
      handle_peer_tool_call(
        correlation_id="corr-peer-3",
        tool_id="delete_note",
        tool_name="delete_note",
        input={"id": 2},
        routing_key=None,
        approval_gate=gate,
        dispatcher=dispatcher,
      )
    )
    await asyncio.sleep(0)

    assert len(connection.sent) == 1
    correlation_id, _, _ = connection.sent[0]
    assert correlation_id == "corr-peer-3"

    dispatcher.resolve_result("corr-peer-3", DispatchOutcome(ok=True, output=None))
    outcome = await task
    assert outcome.ok is True


class TestNoProviderPropagates:
  @pytest.mark.asyncio
  async def test_no_provider_error_propagates_when_approved_but_unroutable(self) -> None:
    from subconscious.dispatch.dispatcher import NoProviderError

    table = ProviderTable()  # no providers registered at all
    dispatcher = ToolDispatcher(table)
    gate = _gate(mutation_requires_approval=True, approved=True)

    with pytest.raises(NoProviderError):
      await handle_peer_tool_call(
        correlation_id="corr-peer-4",
        tool_id="missing_tool",
        tool_name="missing_tool",
        input={},
        routing_key=None,
        approval_gate=gate,
        dispatcher=dispatcher,
      )
