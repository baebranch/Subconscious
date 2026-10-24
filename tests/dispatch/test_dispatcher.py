"""Unit tests for ToolDispatcher (dispatch/dispatcher.py).

Covers `call` minting its own correlation id, `forward_call` preserving a
caller-supplied correlation id end to end, `resolve_result` matching a
`tool.result` back to its pending call by correlation id, and the
no-provider short circuit that never invokes `resolve_provider` when there
are no candidates.

Validates: Requirements 3.7, 6.2, 7.5, 8.2, 8.3, 8.4, 10.3
"""

from __future__ import annotations

import asyncio

import pytest

from subconscious.dispatch.dispatcher import DispatchOutcome, NoProviderError, ToolDispatcher
from subconscious.dispatch.provider_table import ProviderTable


class FakeConnection:
  """A minimal `ProviderConnection` fake that records every `tool.call` it
     is asked to send, without resolving the dispatcher's pending future
     itself — the test drives `resolve_result` explicitly."""

  def __init__(self) -> None:
    self.sent: list[tuple[str, str, object]] = []

  async def send_tool_call(self, correlation_id: str, tool_id: str, input: object) -> None:
    self.sent.append((correlation_id, tool_id, input))

  @property
  def is_available(self) -> bool:
    return True


def _table_with_one_provider(tool_id: str, connection: FakeConnection) -> ProviderTable:
  table = ProviderTable()
  table.upsert("p1", "local_client", connection, frozenset({tool_id}))
  return table


class TestCallMintsCorrelationId:
  @pytest.mark.asyncio
  async def test_call_sends_a_fresh_correlation_id_not_supplied_by_caller(self) -> None:
    connection = FakeConnection()
    table = _table_with_one_provider("edit_file", connection)
    dispatcher = ToolDispatcher(table)

    task = asyncio.ensure_future(dispatcher.call("edit_file", {"path": "a.py"}, routing_key=None))
    await asyncio.sleep(0)  # let call() run up to awaiting the pending future

    assert len(connection.sent) == 1
    correlation_id, tool_id, input_ = connection.sent[0]
    assert tool_id == "edit_file"
    assert input_ == {"path": "a.py"}
    assert correlation_id  # minted, non-empty

    dispatcher.resolve_result(correlation_id, DispatchOutcome(ok=True, output="done"))
    outcome = await task
    assert outcome == DispatchOutcome(ok=True, output="done")

  @pytest.mark.asyncio
  async def test_two_calls_get_distinct_correlation_ids(self) -> None:
    connection = FakeConnection()
    table = _table_with_one_provider("edit_file", connection)
    dispatcher = ToolDispatcher(table)

    task_a = asyncio.ensure_future(dispatcher.call("edit_file", {}, routing_key=None))
    task_b = asyncio.ensure_future(dispatcher.call("edit_file", {}, routing_key=None))
    await asyncio.sleep(0)

    assert len(connection.sent) == 2
    corr_a, _, _ = connection.sent[0]
    corr_b, _, _ = connection.sent[1]
    assert corr_a != corr_b

    dispatcher.resolve_result(corr_a, DispatchOutcome(ok=True, output="a"))
    dispatcher.resolve_result(corr_b, DispatchOutcome(ok=True, output="b"))
    assert await task_a == DispatchOutcome(ok=True, output="a")
    assert await task_b == DispatchOutcome(ok=True, output="b")


class TestForwardCallPreservesCorrelationId:
  @pytest.mark.asyncio
  async def test_forward_call_uses_the_caller_supplied_correlation_id(self) -> None:
    connection = FakeConnection()
    table = _table_with_one_provider("edit_file", connection)
    dispatcher = ToolDispatcher(table)

    inbound_correlation_id = "peer-supplied-corr-123"
    task = asyncio.ensure_future(
      dispatcher.forward_call("edit_file", {"path": "a.py"}, routing_key=None,
                               correlation_id=inbound_correlation_id)
    )
    await asyncio.sleep(0)

    assert len(connection.sent) == 1
    correlation_id, tool_id, input_ = connection.sent[0]
    assert correlation_id == inbound_correlation_id
    assert tool_id == "edit_file"

    dispatcher.resolve_result(inbound_correlation_id, DispatchOutcome(ok=True, output="done"))
    outcome = await task
    assert outcome == DispatchOutcome(ok=True, output="done")


class TestResolveResultMatchingByCorrelationId:
  @pytest.mark.asyncio
  async def test_resolve_result_only_resolves_the_matching_pending_call(self) -> None:
    connection = FakeConnection()
    table = _table_with_one_provider("edit_file", connection)
    dispatcher = ToolDispatcher(table)

    task_a = asyncio.ensure_future(dispatcher.call("edit_file", {}, routing_key=None))
    task_b = asyncio.ensure_future(dispatcher.call("edit_file", {}, routing_key=None))
    await asyncio.sleep(0)
    corr_a, _, _ = connection.sent[0]
    corr_b, _, _ = connection.sent[1]

    # Resolve b first, out of send order, to prove matching is by id not order.
    dispatcher.resolve_result(corr_b, DispatchOutcome(ok=True, output="b"))
    assert not task_a.done()
    outcome_b = await task_b
    assert outcome_b == DispatchOutcome(ok=True, output="b")

    dispatcher.resolve_result(corr_a, DispatchOutcome(ok=True, output="a"))
    outcome_a = await task_a
    assert outcome_a == DispatchOutcome(ok=True, output="a")

  def test_resolve_result_for_unknown_correlation_id_is_a_no_op(self) -> None:
    connection = FakeConnection()
    table = _table_with_one_provider("edit_file", connection)
    dispatcher = ToolDispatcher(table)

    # Should not raise even though nothing is pending under this id.
    dispatcher.resolve_result("no-such-id", DispatchOutcome(ok=False, error="ignored"))


class TestNoProviderShortCircuit:
  @pytest.mark.asyncio
  async def test_call_raises_no_provider_error_when_no_candidates(self) -> None:
    table = ProviderTable()  # empty: no providers registered for any tool id
    dispatcher = ToolDispatcher(table)

    with pytest.raises(NoProviderError):
      await dispatcher.call("missing_tool", {}, routing_key=None)

  @pytest.mark.asyncio
  async def test_forward_call_raises_no_provider_error_when_no_candidates(self) -> None:
    table = ProviderTable()
    dispatcher = ToolDispatcher(table)

    with pytest.raises(NoProviderError):
      await dispatcher.forward_call("missing_tool", {}, routing_key=None, correlation_id="corr-1")

  @pytest.mark.asyncio
  async def test_no_provider_error_carries_the_tool_id(self) -> None:
    table = ProviderTable()
    dispatcher = ToolDispatcher(table)

    with pytest.raises(NoProviderError) as exc_info:
      await dispatcher.call("missing_tool", {}, routing_key=None)
    assert exc_info.value.args[0] == "missing_tool"


class TestUniformDispatchAcrossProviderKinds:
  @pytest.mark.asyncio
  async def test_call_sends_the_same_frame_shape_regardless_of_provider_kind(self) -> None:
    # Requirement 8.2/8.3: same fields (correlation id, tool id, input) sent
    # to exactly one Provider's connection regardless of whether it is a
    # local_client or a peer_engine.
    for kind in ("local_client", "peer_engine"):
      connection = FakeConnection()
      table = ProviderTable()
      table.upsert("p1", kind, connection, frozenset({"edit_file"}))
      dispatcher = ToolDispatcher(table)

      task = asyncio.ensure_future(dispatcher.call("edit_file", {"a": 1}, routing_key=None))
      await asyncio.sleep(0)

      assert len(connection.sent) == 1
      correlation_id, tool_id, input_ = connection.sent[0]
      assert tool_id == "edit_file"
      assert input_ == {"a": 1}

      dispatcher.resolve_result(correlation_id, DispatchOutcome(ok=True, output=None))
      await task
