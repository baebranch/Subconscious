"""Integration tests for `tool.register` wiring in the `/api/v1/events`
WebSocket endpoint (`api/app.py`).

Covers task 11.1: for a LOCAL_CLIENT connection a `tool.register` frame calls
`ProviderTable.upsert` with the advertised tool ids and the Execution_Profile
root advertised via `profile.set`; a re-registration replaces only that
Client's prior set; and a REMOTE_CLIENT connection is rejected without any
Provider_Table mutation.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 4.2, 4.3
"""

from __future__ import annotations

from starlette.testclient import TestClient

from subconscious.api.app import create_app
from subconscious.events import EventBus


TOKEN = "test-token"


class FakeEngine:
  """Minimal Engine stand-in exposing only what `create_app`'s closures touch
     before a `chat.send` frame is actually processed."""

  def __init__(self) -> None:
    self.events = EventBus()


def _make_app_and_client() -> tuple:
  engine = FakeEngine()
  app = create_app(engine, TOKEN)  # type: ignore[arg-type]
  return app, TestClient(app)


def _handshake(ws, client_id: str) -> None:
  ws.send_json({"v": 1, "type": "client.hello", "data": {"client_id": client_id}})
  ack = ws.receive_json()
  assert ack["type"] == "client.hello.ack"


def _barrier(ws) -> None:
  """Synchronisation barrier: frames are processed in order by the command
     pump, so issuing a fresh handshake (unassigned id) and awaiting its ack
     proves every previously-sent frame has already been processed. Using an
     unassigned id avoids a self-conflict with the connection's own id."""
  ws.send_json({"v": 1, "type": "client.hello", "data": {}})
  ack = ws.receive_json()
  assert ack["type"] == "client.hello.ack"


import pytest


class TestLocalClientToolRegister:
  @pytest.fixture(autouse=True)
  def _force_local_classification(self, monkeypatch) -> None:
    # The Starlette TestClient's transport reports a non-loopback peer host
    # ("testclient"), so force LOCAL_CLIENT classification here — real
    # loopback connections classify as LOCAL_CLIENT via `127.0.0.1`/`::1`.
    import subconscious.api.app as app_module
    from subconscious.dispatch import ConnectionKind

    monkeypatch.setattr(
      app_module, "classify_connection", lambda peer_host: ConnectionKind.LOCAL_CLIENT
    )

  def test_register_upserts_advertised_tool_ids(self) -> None:
    app, client = _make_app_and_client()
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws:
      _handshake(ws, "local-1")
      ws.send_json(
        {
          "v": 1,
          "type": "tool.register",
          "data": {"tools": [{"id": "edit_file"}, {"id": "run_terminal"}]},
        }
      )
      _barrier(ws)

      edit_providers = app.state.provider_table.providers_for("edit_file")
      term_providers = app.state.provider_table.providers_for("run_terminal")
      assert [p.provider_id for p in edit_providers] == ["local-1"]
      assert [p.provider_id for p in term_providers] == ["local-1"]
      assert edit_providers[0].kind == "local_client"

  def test_register_attributes_profile_root_when_profile_set_first(self) -> None:
    app, client = _make_app_and_client()
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws:
      _handshake(ws, "local-2")
      ws.send_json(
        {"v": 1, "type": "profile.set", "data": {"profile": {"root": "/proj-a"}}}
      )
      ws.send_json(
        {"v": 1, "type": "tool.register", "data": {"tools": [{"id": "edit_file"}]}}
      )
      _barrier(ws)

      providers = app.state.provider_table.providers_for("edit_file")
      assert len(providers) == 1
      assert providers[0].execution_profile_root == "/proj-a"

  def test_profile_set_after_register_reattributes_root(self) -> None:
    app, client = _make_app_and_client()
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws:
      _handshake(ws, "local-3")
      ws.send_json(
        {"v": 1, "type": "tool.register", "data": {"tools": [{"id": "edit_file"}]}}
      )
      ws.send_json(
        {"v": 1, "type": "profile.set", "data": {"profile": {"root": "/proj-b"}}}
      )
      _barrier(ws)

      providers = app.state.provider_table.providers_for("edit_file")
      assert len(providers) == 1
      assert providers[0].execution_profile_root == "/proj-b"

  def test_reregistration_replaces_only_this_clients_set(self) -> None:
    app, client = _make_app_and_client()
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws:
      _handshake(ws, "local-4")
      ws.send_json(
        {"v": 1, "type": "tool.register", "data": {"tools": [{"id": "a"}, {"id": "b"}]}}
      )
      # Updated registration drops "b" and adds "c".
      ws.send_json(
        {"v": 1, "type": "tool.register", "data": {"tools": [{"id": "a"}, {"id": "c"}]}}
      )
      _barrier(ws)

      assert [p.provider_id for p in app.state.provider_table.providers_for("a")] == ["local-4"]
      assert app.state.provider_table.providers_for("b") == []
      assert [p.provider_id for p in app.state.provider_table.providers_for("c")] == ["local-4"]

  def test_empty_registration_removes_this_clients_tools(self) -> None:
    app, client = _make_app_and_client()
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws:
      _handshake(ws, "local-5")
      ws.send_json(
        {"v": 1, "type": "tool.register", "data": {"tools": [{"id": "edit_file"}]}}
      )
      ws.send_json({"v": 1, "type": "tool.register", "data": {"tools": []}})
      _barrier(ws)

      assert app.state.provider_table.providers_for("edit_file") == []


class TestRemoteClientToolRegisterRejected:
  def test_remote_client_register_is_rejected_without_upsert(self, monkeypatch) -> None:
    # Force the connection to classify as REMOTE_CLIENT regardless of the
    # loopback TestClient transport, since classification is transport-derived.
    import subconscious.api.app as app_module
    from subconscious.dispatch import ConnectionKind

    monkeypatch.setattr(
      app_module, "classify_connection", lambda peer_host: ConnectionKind.REMOTE_CLIENT
    )

    app, client = _make_app_and_client()
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws:
      _handshake(ws, "remote-1")
      ws.send_json(
        {"v": 1, "type": "tool.register", "data": {"tools": [{"id": "edit_file"}]}}
      )
      reject = ws.receive_json()
      assert reject["type"] == "tool.register.reject"
      assert app.state.provider_table.providers_for("edit_file") == []
