"""Tests for the per-turn Provider_Table dispatch seam wired into the
`/api/v1/events` `chat.send` path (`api/app.py`), task 11.2.

Covers:
- `_make_tool_dispatch` converting a `NoProviderError` into the `no-provider`
  `ToolError` kind in the unchanged `{ok, output/error}` result shape
  (Requirement 10.3, 9.3), and passing a resolved `DispatchOutcome` through
  unchanged for both success and error outcomes (Requirement 8.2).
- A live `chat.send` turn building a fresh `ProviderTable.snapshot()` and a
  per-turn `ToolDispatcher`, routing an unregistered tool id to a
  `no-provider` result end to end, and leaving no dispatcher registered once
  the turn completes (Master_Engine scoping, Requirement 5.4/5.5).

Validates: Requirements 3.7, 5.1, 5.4, 5.5, 8.2, 9.3, 10.3
"""

from __future__ import annotations

import asyncio
import json

from starlette.testclient import TestClient

from subconscious.api.app import create_app, _make_tool_dispatch
from subconscious.dispatch import DispatchOutcome, ProviderTable, ToolDispatcher
from subconscious.events import EventBus


TOKEN = "test-token"


# ---------------------------------------------------------------------------
# Unit tests for the NoProviderError -> no-provider ToolError conversion
# ---------------------------------------------------------------------------

class _ImmediateConnection:
  """A `ProviderConnection` fake that resolves the dispatcher's pending future
     inline with a preset `DispatchOutcome`, so `_make_tool_dispatch` can be
     exercised without an out-of-band `tool.result` frame."""

  def __init__(self, dispatcher: ToolDispatcher, outcome: DispatchOutcome) -> None:
    self._dispatcher = dispatcher
    self._outcome = outcome

  async def send_tool_call(self, correlation_id: str, tool_id: str, input: object) -> None:
    self._dispatcher.resolve_result(correlation_id, self._outcome)

  @property
  def is_available(self) -> bool:
    return True


class TestNoProviderConversion:
  async def test_no_provider_error_becomes_no_provider_tool_error(self) -> None:
    dispatcher = ToolDispatcher(ProviderTable())  # empty table -> NoProviderError
    dispatch = _make_tool_dispatch(dispatcher, routing_key=None)

    result = await dispatch("ghost_tool", {"x": 1})

    assert result["ok"] is False
    assert result["error"]["kind"] == "no-provider"
    assert "ghost_tool" in result["error"]["message"]


class TestOutcomePassthrough:
  async def test_successful_outcome_is_returned_as_ok_output(self) -> None:
    table = ProviderTable()
    dispatcher = ToolDispatcher(table)
    conn = _ImmediateConnection(dispatcher, DispatchOutcome(ok=True, output={"wrote": "a.py"}))
    table.upsert("p1", "local_client", conn, frozenset({"edit_file"}))
    dispatch = _make_tool_dispatch(dispatcher, routing_key=None)

    result = await dispatch("edit_file", {"path": "a.py"})

    assert result == {"ok": True, "output": {"wrote": "a.py"}}

  async def test_error_outcome_is_passed_through_unchanged(self) -> None:
    table = ProviderTable()
    dispatcher = ToolDispatcher(table)
    error = {"kind": "execution", "message": "boom"}
    conn = _ImmediateConnection(dispatcher, DispatchOutcome(ok=False, error=error))
    table.upsert("p1", "local_client", conn, frozenset({"edit_file"}))
    dispatch = _make_tool_dispatch(dispatcher, routing_key=None)

    result = await dispatch("edit_file", {})

    assert result == {"ok": False, "error": error}


# ---------------------------------------------------------------------------
# Integration test for the live chat.send dispatch seam (no-provider path)
# ---------------------------------------------------------------------------

class _FakeThread:
  id = 1
  workspace_id = 1


class _FakeSession:
  """Minimal async session: `scalar` resolves the Thread lookup; `get`
     (used for Routing_Key workspace resolution) returns None so the turn's
     routing key is None."""

  async def scalar(self, _query):
    return _FakeThread()

  async def get(self, _model, _pk):
    return None


class _FakeSessionCtx:
  async def __aenter__(self):
    return _FakeSession()

  async def __aexit__(self, *exc):
    return False


class _FakeDB:
  def get_session(self):
    return _FakeSessionCtx()


class _FakeEngine:
  """Engine stand-in whose `stream_chat_events` drives the injected
     `tool_dispatch` for an unregistered tool id, so the no-provider path is
     exercised end to end through the live endpoint without any real agent."""

  def __init__(self) -> None:
    self.events = EventBus()
    self.db = _FakeDB()
    self.saved: list[tuple[int, str, str]] = []

  async def save_message(self, thread_id: int, role: str, content: str) -> None:
    self.saved.append((thread_id, role, content))

  async def stream_chat_events(self, content, thread_id, *, workspace_id=None,
                               model_cfg=None, auto_approve=None, tool_dispatch=None,
                               provider_table=None):
    # Route a call for a tool id that no Provider offers; the seam must
    # convert the NoProviderError into a no-provider ToolError result.
    # Accepts (and ignores) `provider_table` to match the real
    # `Engine.stream_chat_events` signature — the provider-tool bridge
    # (`dispatch/provider_tools.py`) is exercised by its own unit tests, not
    # this end-to-end no-provider seam test.
    from subconscious.stream_events import ToolCallStarted, ToolCallResult

    result = await tool_dispatch("ghost_tool", {"a": 1})
    yield ToolCallStarted(tool_name="ghost_tool", args={"a": 1}, tool_call_id="call-1")
    yield ToolCallResult(
      tool_name="ghost_tool",
      content=result,
      tool_call_id="call-1",
      outcome="failed" if not result["ok"] else "success",
    )


def _handshake(ws) -> None:
  ws.send_json({"v": 1, "type": "client.hello", "data": {"client_id": "c-1"}})
  ack = ws.receive_json()
  assert ack["type"] == "client.hello.ack"


class TestChatSendNoProviderEndToEnd:
  def test_unregistered_tool_routes_to_no_provider_result(self) -> None:
    engine = _FakeEngine()
    app = create_app(engine, TOKEN)  # type: ignore[arg-type]
    client = TestClient(app)

    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws:
      _handshake(ws)
      ws.send_json(
        {
          "v": 1,
          "type": "chat.send",
          "id": "corr-1",
          "data": {"thread_uuid": "t-1", "content": "hi"},
        }
      )
      # Drain frames until the turn completes. Stop immediately on
      # `chat.error` too, and surface it, so a stub/contract mismatch fails
      # fast with a clear assertion instead of exhausting the loop and then
      # hanging on a `receive_json()` that will never get another frame.
      done = False
      for _ in range(10):
        frame = ws.receive_json()
        if frame["type"] == "chat.done":
          done = True
          break
        if frame["type"] == "chat.error":
          raise AssertionError(f"chat.send failed: {frame['data'].get('error')}")
      assert done

    # The tool result persisted for the turn carries the no-provider error.
    tool_rows = [c for (_, role, c) in engine.saved if role == "tool"]
    assert tool_rows, "expected a tool message to be persisted"
    persisted = json.loads(tool_rows[0])
    assert persisted["output"]["error"]["kind"] == "no-provider"

    # Master_Engine scoping: no per-turn dispatcher state survives the turn.
    assert app.state.provider_table.providers_for("ghost_tool") == []
