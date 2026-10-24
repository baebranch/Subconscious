"""Integration tests for handshake and connection-classification wiring in
the `/api/v1/events` WebSocket endpoint (`api/app.py`).

Covers: `client.hello` / `client.hello.ack` / `client.hello.reject` handling,
per-connection handshake-gating of `tool.register`/`chat.send`, and
`ProviderTable.remove` being called for the connection's Client_ID on
disconnect.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4.1, 4.4, 4.5
"""

from __future__ import annotations

from starlette.testclient import TestClient

from subconscious.api.app import create_app
from subconscious.events import EventBus


TOKEN = "test-token"


class FakeEngine:
  """Minimal Engine stand-in exposing only what `create_app`'s closures
     touch before a `chat.send` frame is actually processed."""

  def __init__(self) -> None:
    self.events = EventBus()


def _make_client() -> TestClient:
  engine = FakeEngine()
  app = create_app(engine, TOKEN)  # type: ignore[arg-type]
  return TestClient(app)


class TestHandshakeAckAssignsClientId:
  def test_hello_without_client_id_is_acked_with_an_assigned_id(self) -> None:
    client = _make_client()
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws:
      ws.send_json({"v": 1, "type": "client.hello", "id": "corr-1", "data": {}})
      ack = ws.receive_json()
      assert ack["type"] == "client.hello.ack"
      assert ack["id"] == "corr-1"
      assert ack["data"]["client_id"]

  def test_hello_with_requested_client_id_is_validated_and_echoed(self) -> None:
    client = _make_client()
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws:
      ws.send_json({"v": 1, "type": "client.hello", "data": {"client_id": "my-id"}})
      ack = ws.receive_json()
      assert ack["type"] == "client.hello.ack"
      assert ack["data"]["client_id"] == "my-id"


class TestHandshakeConflictRejected:
  def test_conflicting_client_id_is_rejected(self) -> None:
    client = _make_client()
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as first_ws:
      first_ws.send_json({"v": 1, "type": "client.hello", "data": {"client_id": "dup-id"}})
      first_ack = first_ws.receive_json()
      assert first_ack["type"] == "client.hello.ack"

      with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as second_ws:
        second_ws.send_json({"v": 1, "type": "client.hello", "data": {"client_id": "dup-id"}})
        reject = second_ws.receive_json()
        assert reject["type"] == "client.hello.reject"
        assert reject["data"]["reason"] == "client_id_in_use"


class TestFrameGatingBeforeHandshake:
  def test_chat_send_before_handshake_is_dropped_not_erroring(self) -> None:
    client = _make_client()
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws:
      # Sent before any client.hello — must be dropped silently (no
      # chat.error, no chat.delta/chat.done) rather than processed.
      ws.send_json(
        {"v": 1, "type": "chat.send", "id": "corr-x", "data": {"thread_uuid": "t", "content": "hi"}}
      )
      # Now complete the handshake; the ack must be the next frame received,
      # proving the earlier chat.send was never enqueued for processing.
      ws.send_json({"v": 1, "type": "client.hello", "data": {}})
      ack = ws.receive_json()
      assert ack["type"] == "client.hello.ack"

  def test_tool_register_before_handshake_is_dropped(self) -> None:
    client = _make_client()
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws:
      ws.send_json(
        {"v": 1, "type": "tool.register", "data": {"tool_ids": ["edit_file"]}}
      )
      ws.send_json({"v": 1, "type": "client.hello", "data": {}})
      ack = ws.receive_json()
      assert ack["type"] == "client.hello.ack"


class TestDisconnectReleasesClientIdAndProvider:
  def test_client_id_is_freed_on_disconnect_so_it_can_be_reused(self) -> None:
    client = _make_client()
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws:
      ws.send_json({"v": 1, "type": "client.hello", "data": {"client_id": "reuse-me"}})
      ack = ws.receive_json()
      assert ack["data"]["client_id"] == "reuse-me"
    # First connection closed; the id must be free for a brand-new handshake.
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws2:
      ws2.send_json({"v": 1, "type": "client.hello", "data": {"client_id": "reuse-me"}})
      ack2 = ws2.receive_json()
      assert ack2["type"] == "client.hello.ack"
      assert ack2["data"]["client_id"] == "reuse-me"

  def test_provider_table_removes_entry_for_client_id_on_disconnect(self) -> None:
    engine = FakeEngine()
    app = create_app(engine, TOKEN)  # type: ignore[arg-type]
    client = TestClient(app)
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws:
      ws.send_json({"v": 1, "type": "client.hello", "data": {"client_id": "provider-1"}})
      ws.receive_json()
      # Manually populate the table as though tool.register had run (task
      # 11.1 wires the real upsert path) so we can observe removal here.
      app.state.provider_table.upsert("provider-1", "local_client", object(), frozenset({"edit_file"}))
      assert app.state.provider_table.providers_for("edit_file")
    assert app.state.provider_table.providers_for("edit_file") == []
