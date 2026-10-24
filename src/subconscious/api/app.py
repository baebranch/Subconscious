""" Local engine API — FastAPI app factory.

    Exposes a loopback-only REST + WebSocket surface over a running ``Engine``
    All local clients use a single subconscious daemon on desktop
"""
from __future__ import annotations

import asyncio
import logging
import time
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Awaitable, Callable, Optional, AsyncIterator, TYPE_CHECKING
from fastapi import FastAPI, Depends, HTTPException, Header, WebSocket, WebSocketDisconnect, status

from ..constants import VERSION
from ..db.models import Workspace, Thread, Message
from ..dispatch import (
  ConnectionKind,
  DispatchOutcome,
  HandshakeService,
  LocalClientConnection,
  NoProviderError,
  ProviderTable,
  ToolDispatcher,
  classify_connection,
  should_accept_frame,
  should_reject_tool_register,
)
from ..stream_events import (
  TextDelta,
  ToolCallStarted,
  ToolCallResult,
  TurnUsage,
  strip_attached_context,
  tool_block_to_json,
)
from .schemas import (
  ThreadDTO,
  MessageDTO,
  WorkspaceDTO,
  HealthResponse,
  ModelConfigDTO,
  CreateThreadRequest,
  UpdateThreadRequest,
)

if TYPE_CHECKING:
  from ..engine import Engine


# Loggin and env setup
API_PREFIX = "/api/v1"
logger = logging.getLogger("subconscious")

# Keep these in sync with the formats configured in cli/__init__.py so the
# uvicorn/FastAPI logs read identically to the rest of the application.
_DEV_LOG_FORMAT = "[%(levelname)s|%(asctime)s.%(msecs)04d|%(filename)s|%(lineno)d] %(message)s"
_LOG_FORMAT = "[%(asctime)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def build_uvicorn_log_config(dev: bool = False) -> dict:
  """ Return a uvicorn ``log_config`` (dictConfig) that matches the CLI format.

      Mirrors the formatters set up in ``cli.__init__.main``: the verbose
      level/file/line layout in dev mode, and the terse timestamped layout
      otherwise. ``disable_existing_loggers`` is False so the app's own
      ``subconscious`` logger keeps working.
  """
  fmt = _DEV_LOG_FORMAT if dev else _LOG_FORMAT
  return {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
      "default": {"format": fmt, "datefmt": _LOG_DATEFMT},
      "access": {"format": fmt, "datefmt": _LOG_DATEFMT},
    },
    "handlers": {
      "default": {
        "class": "logging.StreamHandler",
        "formatter": "default",
        "stream": "ext://sys.stderr",
      },
      "access": {
        "class": "logging.StreamHandler",
        "formatter": "access",
        "stream": "ext://sys.stdout",
      },
    },
    "loggers": {
      "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
      "uvicorn.error": {"level": "INFO"},
      "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
  }


def create_app(engine: Engine, token: str) -> FastAPI:
  app = FastAPI(
    title="Subconscious Engine API",
    description="Local loopback API for Subconscious clients.",
    version=VERSION.lstrip("v"),
    docs_url=None,
    redoc_url=None,
  )
  app.state.token = token
  app.state.engine = engine
  # Scoped per-app (not per-connection): every connection to this Engine
  # shares one Provider_Table and one HandshakeService so Client_IDs are
  # unique across every currently-connected Client (Requirement 1.2, 1.3).
  app.state.provider_table = ProviderTable()
  app.state.handshake_service = HandshakeService()

  def require_token(authorization: Optional[str] = Header(default=None)) -> None:
    """ Auth """
    expected = f"Bearer {token}"
    if authorization != expected:
      raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing token")

  async def get_db() -> AsyncIterator[AsyncSession]:
    """ Request-scoped database session dependency.

        Yields a single session for the lifetime of the request and closes it
        afterwards. Endpoints are still responsible for calling ``commit()`` on
        writes — the dependency only manages the session lifecycle.
    """
    async with engine.db.get_session() as session:
      yield session

  ###############################################################
  # Helpers
  ###############################################################
  async def _thread_by_uuid(session, thread_uuid: str) -> Thread:
    th = await session.scalar(select(Thread).where(Thread.uuid == thread_uuid))
    if not th:
      raise HTTPException(status.HTTP_404_NOT_FOUND, "Thread not found")
    return th

  async def _workspace_by_uuid(session, workspace_uuid: str) -> Workspace:
    ws = await session.scalar(select(Workspace).where(Workspace.uuid == workspace_uuid))
    if not ws:
      raise HTTPException(status.HTTP_404_NOT_FOUND, "Workspace not found")
    return ws

  ###############################################################
  # API Endpoints
  ###############################################################
  @app.get(f"{API_PREFIX}/health", response_model=HealthResponse)
  async def health() -> HealthResponse:
    return HealthResponse(version=VERSION, node_id=getattr(engine.config, "node_id", None))

  @app.get(f"{API_PREFIX}/models", response_model=list[ModelConfigDTO], dependencies=[Depends(require_token)])
  async def list_models() -> list[ModelConfigDTO]:
    """ List configured models. The API key is never returned.

        The first configured model is flagged ``is_default`` — it's the one used
        when a chat.send frame omits ``model_id`` (mirrors the UI behaviour).
    """
    cfgs = engine.agent_manager.list_model_cfgs()
    return [
      ModelConfigDTO(
        id=c["id"],
        alias=c.get("alias"),
        provider=c.get("provider"),
        model=c.get("model"),
        system_prompt=c.get("system_prompt"),
        base_url=c.get("base_url"),
        is_default=(i == 0),
      )
      for i, c in enumerate(cfgs)
    ]

  @app.get(f"{API_PREFIX}/workspaces", response_model=list[WorkspaceDTO], dependencies=[Depends(require_token)])
  async def list_workspaces(session: AsyncSession = Depends(get_db)) -> list[WorkspaceDTO]:
    rows = (await session.scalars(select(Workspace).order_by(Workspace.created_at))).all()
    return [
      WorkspaceDTO(
        uuid=w.uuid, name=w.name, description=w.description,
        created_at=w.created_at, updated_at=w.updated_at,
      )
      for w in rows
    ]

  @app.get(
    f"{API_PREFIX}/workspaces/{{workspace_uuid}}/threads",
    response_model=list[ThreadDTO], dependencies=[Depends(require_token)],
  )
  async def list_threads(workspace_uuid: str, session: AsyncSession = Depends(get_db)) -> list[ThreadDTO]:
    ws = await _workspace_by_uuid(session, workspace_uuid)
    rows = (
      await session.scalars(
        select(Thread).where(Thread.workspace_id == ws.id)
        .order_by(Thread.updated_at.desc())
      )
    ).all()
    return [
      ThreadDTO(
        uuid=t.uuid, workspace_uuid=workspace_uuid, title=t.title,
        description=t.description, default_model_id=t.default_model_id,
        created_at=t.created_at, updated_at=t.updated_at,
      )
      for t in rows
    ]

  @app.post(f"{API_PREFIX}/threads", response_model=ThreadDTO, dependencies=[Depends(require_token)])
  async def create_thread(req: CreateThreadRequest, session: AsyncSession = Depends(get_db)) -> ThreadDTO:
    """ Create thread """
    ws = await _workspace_by_uuid(session, req.workspace_uuid)
    workspace_id = ws.id
    thread = await engine.get_or_create_thread(
      content=req.title or "New Thread", workspace_id=workspace_id,
    )

    if req.title or req.description:
      th = await session.get(Thread, thread.id)
      if req.title:
        th.title = req.title
      if req.description:
        th.description = req.description
      await session.commit()

    return ThreadDTO(
      created_at=thread.created_at, updated_at=thread.updated_at,
      description=req.description, default_model_id=thread.default_model_id,
      uuid=thread.uuid, workspace_uuid=req.workspace_uuid, title=req.title or thread.title,
    )

  @app.patch(f"{API_PREFIX}/threads/{{thread_uuid}}", response_model=ThreadDTO, dependencies=[Depends(require_token)])
  async def update_thread(thread_uuid: str, req: UpdateThreadRequest, session: AsyncSession = Depends(get_db)) -> ThreadDTO:
    """ Update thread """
    th = await _thread_by_uuid(session, thread_uuid)
    if req.title is not None:
      th.title = req.title
    if req.description is not None:
      th.description = req.description
    if req.default_model_id is not None:
      th.default_model_id = req.default_model_id
    await session.commit()
    ws = await session.get(Workspace, th.workspace_id)
    dto = ThreadDTO(
      uuid=th.uuid, workspace_uuid=ws.uuid if ws else "", title=th.title,
      description=th.description, default_model_id=th.default_model_id,
      created_at=th.created_at, updated_at=th.updated_at,
    )
    await engine.events.publish({"type": "thread.updated", "data": dto.model_dump(mode="json")})
    return dto

  @app.get(f"{API_PREFIX}/threads/{{thread_uuid}}/messages", response_model=list[MessageDTO], dependencies=[Depends(require_token)])
  async def list_messages(thread_uuid: str, session: AsyncSession = Depends(get_db)) -> list[MessageDTO]:
    th = await _thread_by_uuid(session, thread_uuid)
    rows = (
      await session.scalars(
        select(Message).where(Message.thread_id == th.id).order_by(Message.created_at)
      )
    ).all()
    return [
      MessageDTO(
        uuid=m.uuid, thread_uuid=thread_uuid, role=m.role,
        content=m.content, created_at=m.created_at,
      )
      for m in rows
    ]

  @app.websocket(f"{API_PREFIX}/events")
  async def events_ws(ws: WebSocket) -> None:
    # Auth via query param (browsers/extensions can't set WS headers reliably).
    if ws.query_params.get("token") != token:
      await ws.close(code=status.WS_1008_POLICY_VIOLATION)
      return
    await ws.accept()
    queue = engine.events.subscribe()

    # A single WebSocket carries three concurrent writers once `chat.send` is
    # handled as a background task (below): the event pump, one or more
    # in-flight `chat.send` turns (each streaming `chat.delta`/`chat.error`/
    # `chat.done`), and `tool.register`/`client.hello` replies. Starlette's
    # `WebSocket.send_json` is not safe to call concurrently from multiple
    # tasks on the same connection, so every outbound write goes through this
    # lock-guarded helper rather than calling `ws.send_json` directly.
    send_lock = asyncio.Lock()

    async def _send_json(payload: dict) -> None:
      async with send_lock:
        await ws.send_json(payload)

    # `chat.send` frames are handled as background tasks (see `pump_commands`)
    # so the read loop keeps servicing inbound frames — most importantly this
    # connection's own `tool.result` replies to `tool.call` frames dispatched
    # during the turn — while a turn is still streaming. Handling a
    # `chat.send` inline would block `ws.receive_json()` for the turn's
    # entire duration, so a `tool.result` the client sends back on this same
    # socket could never be read, the pending `ToolDispatcher.call` would
    # hang until the model's inactivity timeout fired, and (because the
    # turn's `finally` never reaches its persistence step in time, or the
    # turn errors out first) neither the tool result nor anything after it
    # gets saved — matching both "no output after a tool call" and "history
    # missing on reload" for extension-driven turns. Tracked here so pending
    # turns can be cancelled on disconnect (Requirement 5.4 — no turn state
    # outlives its connection).
    chat_tasks: set[asyncio.Task] = set()

    # In-flight turns keyed by their `chat.send` correlation id, so a `chat.cancel`
    # ("stop" button) can target exactly the turn the user is watching.
    chat_tasks_by_corr: dict[str, asyncio.Task] = {}

    # Correlation ids the user explicitly asked to stop. A turn consults this on
    # cancellation to tell a deliberate stop apart from an internal teardown, and
    # reports the former as a normal cancelled completion rather than an error.
    cancelled_turns: set[str] = set()

    # Classification is derived only from the transport's own peer address,
    # never from any client-supplied value (Requirement 4.4, 4.5). A
    # Local_Client's Device identity remains implicit — this Engine's own
    # `node_id` — so the handshake below never reads a device id from the
    # client, only an optional Client_ID (Requirement 1.4).
    peer_host = ws.client.host if ws.client is not None else ""
    connection_kind = classify_connection(peer_host)

    # Per-connection handshake state (Requirement 1.1, 1.2, 1.3, 1.5). No
    # frame requiring identity is accepted until handshake_complete is True.
    client_id: Optional[str] = None
    handshake_complete = False

    # Per-connection Provider state (Requirement 2). `local_connection` is the
    # `ProviderConnection` this Local_Client is represented by in the shared
    # Provider_Table; it is created lazily on first `tool.register`. The
    # Execution_Profile root advertised via `profile.set` is tracked so it can
    # be attributed to this Client's Provider entry as the Claim key
    # (Requirement 3.2) when its tools are (re-)registered.
    local_connection: Optional[LocalClientConnection] = None
    profile_root: Optional[str] = None
    registered_tool_ids: Optional[frozenset[str]] = None
    # Per-tool id -> {"description", "inputSchema"} advertised in the most recent
    # `tool.register` frame's `RegisteredToolDescriptor` entries, forwarded into the
    # Provider_Table entry so the agent-tool bridge (dispatch/provider_tools.py) can
    # build a model-facing tool for each of this Local_Client's registered tool ids.
    registered_tool_descriptors: dict[str, dict] = {}

    # Per-turn ToolDispatchers currently awaiting `tool.result` frames on this
    # connection. Each `chat.send` builds its own dispatcher over a fresh
    # `ProviderTable.snapshot()` (Master_Engine scoping — no persisted state,
    # Requirement 5.4/5.5) and registers it here for the duration of the turn;
    # inbound `tool.result` frames are routed to every active dispatcher by
    # correlation id (a no-op for dispatchers that didn't originate the call).
    active_dispatchers: set[ToolDispatcher] = set()

    async def _handle_client_hello(frame: dict) -> bool:
      """Process a `client.hello` frame, assigning/validating a Client_ID via
         the app-scoped HandshakeService. Returns False when the handshake
         was rejected, in which case the caller must stop reading further
         frames from this connection (Requirement 1.2, 1.3)."""
      nonlocal client_id, handshake_complete
      corr = frame.get("id")
      requested_client_id = frame.get("data", {}).get("client_id")
      result = app.state.handshake_service.handshake(requested_client_id)
      if not result.ok:
        await _send_json(
          {"v": 1, "type": "client.hello.reject", "id": corr, "data": {"reason": result.reason}}
        )
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return False
      client_id = result.client_id
      handshake_complete = True
      await _send_json(
        {"v": 1, "type": "client.hello.ack", "id": corr, "data": {"client_id": client_id}}
      )
      return True

    def _upsert_local_provider() -> None:
      """(Re-)register this Local_Client's currently-advertised tool ids in the
         shared Provider_Table under its Client_ID, attributing the tools to
         its most recent Execution_Profile root (Requirement 2.1, 2.2, 2.3,
         3.2). `ProviderTable.upsert` replaces only this Client_ID's prior
         entry, so other connected Clients' registrations are untouched
         (Requirement 2.1, 2.3). An empty tool set behaves as no entry
         (Requirement 2.4, 2.5)."""
      nonlocal local_connection
      if client_id is None or registered_tool_ids is None:
        return
      if local_connection is None:
        local_connection = LocalClientConnection(_send_json)
      app.state.provider_table.upsert(
        provider_id=client_id,
        kind=ConnectionKind.LOCAL_CLIENT.value,
        connection=local_connection,
        tool_ids=registered_tool_ids,
        execution_profile_root=profile_root,
        tool_descriptors=registered_tool_descriptors,
      )

    async def _handle_tool_register(frame: dict) -> None:
      """Handle a `tool.register` frame. A Local_Client's advertised tool ids
         are recorded in the Provider_Table (Requirement 2.1-2.5); a
         Remote_Client is rejected without any table mutation (Requirement
         4.2, 4.3) — a Remote_Client never becomes a Provider."""
      nonlocal registered_tool_ids, registered_tool_descriptors
      corr = frame.get("id")
      if should_reject_tool_register(connection_kind):
        # Requirement 4.2/4.3: never mutate the Provider_Table for a
        # Remote_Client — reject the registration outright.
        await _send_json(
          {
            "v": 1,
            "type": "tool.register.reject",
            "id": corr,
            "data": {"reason": "remote_clients_cannot_register_tools"},
          }
        )
        return
      tools = frame.get("data", {}).get("tools") or []
      registered_tool_ids = frozenset(
        t["id"] for t in tools if isinstance(t, dict) and "id" in t
      )
      registered_tool_descriptors = {
        t["id"]: {
          "description": t.get("description") or "",
          "inputSchema": t.get("inputSchema") or {"type": "object"},
        }
        for t in tools
        if isinstance(t, dict) and "id" in t
      }
      _upsert_local_provider()

    def _handle_profile_set(frame: dict) -> None:
      """Record the Execution_Profile root advertised via `profile.set` so it
         can be attributed to this Local_Client's Provider entry as its Claim
         key (Requirement 3.2). If the Client has already registered tools,
         re-attribute them to the updated root."""
      nonlocal profile_root
      profile = frame.get("data", {}).get("profile") or {}
      profile_root = profile.get("root")
      if connection_kind == ConnectionKind.LOCAL_CLIENT and registered_tool_ids is not None:
        _upsert_local_provider()

    def _handle_tool_result(frame: dict) -> None:
      """Route an inbound `tool.result` frame to the turn's ToolDispatcher by
         correlation id, using the same correlation matching regardless of
         Provider kind (Requirement 6.2, 8.2, 8.4). The result data reuses the
         unchanged `tool.result` frame shape (`ok`/`output`/`error`, Requirement
         9.3). Broadcasting to every active dispatcher is safe because
         `resolve_result` is a no-op for correlation ids it did not mint."""
      correlation_id = frame.get("id")
      if correlation_id is None:
        return
      data = frame.get("data", {}) or {}
      ok = bool(data.get("ok"))
      outcome = DispatchOutcome(
        ok=ok,
        output=data.get("output") if ok else None,
        error=None if ok else data.get("error"),
      )
      for dispatcher in list(active_dispatchers):
        dispatcher.resolve_result(correlation_id, outcome)

    async def pump_events() -> None:
      """Forward engine events to this client."""
      while True:
        event = await queue.get()
        await _send_json({"v": 1, **event})

    def _launch_chat_send(frame: dict) -> None:
      """Run `_handle_chat_send` as a tracked background task rather than
         awaiting it inline. `chat.send` handling can span many seconds
         (model latency, tool round trips) and, when the connected client
         itself provides tools (e.g. the VS Code extension), it depends on
         reading a `tool.call`'s matching `tool.result` back off THIS SAME
         connection's read loop. Awaiting it inline would block
         `pump_commands`'s `ws.receive_json()` for the whole turn, so that
         `tool.result` could never be read and the pending dispatch would
         hang until the model's inactivity timeout tore the turn down —
         losing the tool output and everything the model would have said
         after it, including the persistence step that would have saved
         them for the next `messages` reload.

         Running as its own task is also what makes `chat.cancel` possible: the
         read loop stays responsive, so a stop request can be read and the
         turn's task cancelled while it is still streaming."""
      task = asyncio.create_task(
        _handle_chat_send(
          _send_json, engine, frame, app.state.provider_table, active_dispatchers,
          cancelled_turns,
        )
      )
      chat_tasks.add(task)
      corr = frame.get("id")
      if isinstance(corr, str):
        chat_tasks_by_corr[corr] = task
        task.add_done_callback(lambda _t, c=corr: chat_tasks_by_corr.pop(c, None))
      task.add_done_callback(chat_tasks.discard)

    def _handle_chat_cancel(frame: dict) -> None:
      """Stop an in-flight turn at the user's request ("stop" button).

         `data.turn_id` names the `chat.send` correlation id to stop; when it is
         omitted every turn on this connection is stopped. The correlation id is
         recorded in `cancelled_turns` FIRST so the turn's own cancellation handler
         knows this was a deliberate stop and reports it as a normal (cancelled)
         completion rather than an error. Cancelling the task unwinds the agent run,
         and `_handle_chat_send` still persists whatever the model had already said
         and always emits a terminal frame, so the client's stop button reliably
         returns the UI to idle."""
      turn_id = (frame.get("data") or {}).get("turn_id") or frame.get("id")
      targets = (
        [(turn_id, chat_tasks_by_corr.get(turn_id))]
        if isinstance(turn_id, str) and turn_id in chat_tasks_by_corr
        else list(chat_tasks_by_corr.items())
      )
      for corr, task in targets:
        if task is None or task.done():
          continue
        if corr in cancelled_turns:
          # Already stopping. Cancelling again could interrupt the turn's own
          # cancellation handler mid-await, before it manages to persist the partial
          # reply and send the terminal frame the client needs to return to idle.
          continue
        cancelled_turns.add(corr)
        task.cancel()

    async def pump_commands() -> None:
      """Handle commands sent by the client."""
      while True:
        frame = await ws.receive_json()
        frame_type = frame.get("type")
        if frame_type == "client.hello":
          if not await _handle_client_hello(frame):
            return
          continue
        if frame_type == "profile.set":
          # `profile.set` carries workspace identity, not a
          # handshake-gated identity-requiring frame; record the advertised
          # Execution_Profile root for Claim attribution (Requirement 3.2).
          _handle_profile_set(frame)
          continue
        if frame_type == "tool.result":
          # A Provider's response to a `tool.call` this Engine dispatched
          # during a `chat.send` turn. Match it back to its pending dispatch
          # by correlation id (Requirement 8.4). Reading this frame is only
          # possible because `chat.send` no longer blocks this loop (see
          # `_launch_chat_send`).
          _handle_tool_result(frame)
          continue
        if frame_type == "chat.cancel":
          # The user pressed stop. Handled regardless of handshake state: it only
          # ever cancels work this same connection started.
          _handle_chat_cancel(frame)
          continue
        if frame_type in ("tool.register", "chat.send"):
          if not should_accept_frame(handshake_complete):
            # Dropped: this connection hasn't completed its handshake yet
            # (Requirement 1.1).
            continue
          if frame_type == "chat.send":
            _launch_chat_send(frame)
          elif frame_type == "tool.register":
            # For LOCAL_CLIENT connections record the advertised tool ids in
            # the Provider_Table; for REMOTE_CLIENT connections reject
            # without any table mutation (Requirement 2.1-2.5, 4.2, 4.3).
            await _handle_tool_register(frame)

    pump_task = asyncio.create_task(pump_events())
    cmd_task = asyncio.create_task(pump_commands())
    try:
      await asyncio.wait({pump_task, cmd_task}, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
      pass
    finally:
      engine.events.unsubscribe(queue)
      for t in (pump_task, cmd_task):
        t.cancel()
      # Requirement 5.4: no turn's Master_Engine/routing state outlives its
      # connection — cancel any `chat.send` turns still running for it.
      for t in list(chat_tasks):
        t.cancel()
      # Requirement 1.6: remove every Provider_Table entry attributed to
      # this connection's Client_ID, and free the id for reuse. Mark the
      # connection unavailable first so any in-flight dispatch does not treat
      # this Provider as reachable.
      if local_connection is not None:
        local_connection.mark_unavailable()
      if client_id is not None:
        app.state.handshake_service.release(client_id)
        app.state.provider_table.remove(client_id)

  return app


async def _resolve_routing_key(engine: Engine, workspace_id: Optional[int]) -> Optional[str]:
  """Resolve the Routing_Key for a turn: the active Thread's execution-context
     workspace root, compared as a plain string against each candidate
     Provider's Execution_Profile root during Claim resolution (Requirement
     3.2). It is derived from the Thread → Workspace relationship the same way
     the engine resolves `workspace_id` today; we use the workspace's first
     attached directory as its root. Returns None when no root is known, in
     which case Claim resolution falls back to earliest-registration order
     (Requirement 3.5) — and single-Provider tool ids skip Claim evaluation
     entirely (Requirement 3.1, 9.2)."""
  if not workspace_id:
    return None
  try:
    async with engine.db.get_session() as session:
      ws_row = await session.get(Workspace, workspace_id)
      if ws_row is None or not ws_row.directories:
        return None
      import json

      dirs = json.loads(ws_row.directories)
      if isinstance(dirs, list) and dirs:
        return str(dirs[0])
  except Exception:  # never let Routing_Key resolution break a turn
    logger.debug("routing key resolution failed", exc_info=True)
  return None


def _is_placeholder_thread_title(title: Optional[str]) -> bool:
  """ Whether a thread's current title is an unset placeholder ("New Thread" or
      empty/None) rather than a name the user or a prior auto-naming pass chose.
      Only placeholder-titled threads are eligible for auto-naming on their first
      message, so a user-renamed (or already auto-named) thread is never
      overwritten. """
  return not title or title.strip().lower() == "new thread"


def _derive_thread_title(content: str) -> str:
  """ Derive a short thread title from the first few words of a message, mirroring
      `Engine.get_or_create_thread`'s own placeholder-title derivation.

      Any attached file/folder context is stripped first: it is inlined ahead of the
      user's own words, so naming a thread from the raw text produced useless titles
      like "[Context - File." instead of describing what the user actually asked. Just
      as attachments don't show up in the transcript, they don't name the thread. """
  words = strip_attached_context(content).split()
  title = " ".join(words[:6])
  if len(words) > 6:
    title += "…"
  return title or "New Thread"


async def _autoname_thread(engine: Engine, thread_id: int, workspace_id: int, content: str) -> None:
  """ Rename a placeholder-titled thread from its first message and publish
      `thread.updated` so every connected client refreshes its thread list/label
      live (mirrors the desktop UI's "title may have been updated" refresh).
      Best-effort: a failure here must not abort the chat turn. """
  title = _derive_thread_title(content)
  try:
    async with engine.db.get_session() as session:
      th = await session.get(Thread, thread_id)
      if th is None or not _is_placeholder_thread_title(th.title):
        return
      th.title = title  # type: ignore[assignment]
      await session.commit()
      ws = await session.get(Workspace, workspace_id)
      workspace_uuid = ws.uuid if ws else None
      dto = ThreadDTO(
        uuid=th.uuid, workspace_uuid=workspace_uuid or "", title=th.title,
        description=th.description, default_model_id=th.default_model_id,
        created_at=th.created_at, updated_at=th.updated_at,
      )
    await engine.events.publish({"type": "thread.updated", "data": dto.model_dump(mode="json")})
  except Exception:
    logger.exception("failed to auto-name thread %s", thread_id)


def _make_tool_dispatch(dispatcher: ToolDispatcher, routing_key: Optional[str]):
  """Build the per-turn `async (tool_id, input) -> ToolResult-dict` callable the
     agent loop uses to route a tool call to exactly one Provider via
     `ToolDispatcher.call` (Requirement 3.7, 8.2). A `NoProviderError` — raised
     when no Provider currently offers the tool id — is converted into the
     `no-provider` `ToolError` kind in the unchanged `tool.result` result shape
     (`{ok: false, error: {...}}`) rather than propagating (Requirement 10.3,
     9.3)."""

  async def dispatch_tool_call(tool_id: str, tool_input: object) -> dict:
    try:
      outcome = await dispatcher.call(tool_id, tool_input, routing_key)
    except NoProviderError:
      return {
        "ok": False,
        "error": {
          "kind": "no-provider",
          "message": f"No provider is currently registered for tool '{tool_id}'.",
        },
      }
    if outcome.ok:
      return {"ok": True, "output": outcome.output}
    return {"ok": False, "error": outcome.error}

  return dispatch_tool_call


async def _handle_chat_send(
  send_json: Callable[[dict], Awaitable[None]],
  engine: Engine,
  frame: dict,
  provider_table: ProviderTable,
  active_dispatchers: set,
  cancelled_turns: Optional[set] = None,
) -> None:
  """ Persist the user message, stream the assistant reply as `chat.delta` frames,
      persist the assistant message, then emit `chat.done`. The `message.created`
      events for both messages are published by engine.save_message and reach all
      connected clients (including this one) via the event pump.

      The Engine receiving this `chat.send` is the Master_Engine for this turn
      (Requirement 5.1). Tool routing for the turn uses a fresh
      `ProviderTable.snapshot()` taken here — a turn-scoped view that is never
      persisted, never merged with another Engine's table, and never carried to
      a subsequent turn (Requirement 5.2, 5.4, 5.5). No hand-off with any other
      Engine precedes this (Requirement 5.3).

      Runs as a background task per `chat.send` (see `_launch_chat_send` in
      `events_ws`), so `send_json` is a lock-guarded write helper shared with
      the connection's other concurrent writers rather than the raw
      `WebSocket.send_json`, and the connection's own read loop stays free to
      receive this turn's `tool.result` frames while it runs.
  """
  corr = frame.get("id")
  data = frame.get("data", {})
  thread_uuid = data.get("thread_uuid")
  content = (data.get("content") or "").strip()
  # Client-resolved context attached to this message (the VS Code extension's "#"
  # file/folder/terminal mentions, already rendered as text). Sent separately from
  # `content` so it can augment the MODEL PROMPT without becoming part of the
  # persisted user message — otherwise the chat transcript shows the entire inlined
  # contents of every attachment instead of what the user actually typed.
  raw_context = data.get("context_blocks") or []
  context_blocks = [b for b in raw_context if isinstance(b, str) and b.strip()]
  # Optional model config selection. When omitted, stream_chat falls back to the
  # engine's best/default model. A supplied-but-unknown id is an error.
  model_id = data.get("model_id")
  if not thread_uuid or not content:
    await send_json({"v": 1, "type": "chat.error", "id": corr, "data": {"error": "thread_uuid and content required"}})
    return

  model_cfg = None
  if model_id:
    model_cfg = engine.agent_manager.get_model_cfg(model_id)
    if model_cfg is None:
      await send_json({"v": 1, "type": "chat.error", "id": corr, "data": {"error": f"unknown model_id: {model_id}"}})
      return

  # Resolve uuids → local ids
  async with engine.db.get_session() as session:
    th = await session.scalar(select(Thread).where(Thread.uuid == thread_uuid))
    if not th:
      await send_json({"v": 1, "type": "chat.error", "id": corr, "data": {"error": "thread not found"}})
      return
    thread_id, workspace_id = th.id, th.workspace_id
    current_title = th.title

  # Auto-name the thread from its first message: a thread created without an
  # explicit title (e.g. the VS Code extension's "New Thread" button, or the
  # desktop UI) otherwise keeps that placeholder forever, since nothing else ever
  # renames it. Deriving the name here — on the first turn only — mirrors what
  # `get_or_create_thread`'s own title derivation already does for threads created
  # directly from a message, and publishes `thread.updated` so every connected
  # client (VS Code extension, desktop UI) refreshes its thread list/label live.
  if _is_placeholder_thread_title(current_title):
    await _autoname_thread(engine, thread_id, workspace_id, content)

  # Master_Engine scoping (Requirement 5): take a fresh, turn-scoped snapshot of
  # the live Provider_Table and build a per-turn ToolDispatcher over it. The
  # dispatcher replaces the legacy direct single-client tool-call path — the
  # agent loop routes every tool call through `ToolDispatcher.call`, which
  # resolves exactly one Provider (Requirement 3.7) and short-circuits to a
  # `no-provider` error when none is registered (Requirement 10.3). The
  # dispatcher is registered so inbound `tool.result` frames can be matched
  # back to it, and unregistered when the turn completes so no routing state
  # survives the turn (Requirement 5.4).
  turn_table = provider_table.snapshot()
  dispatcher = ToolDispatcher(turn_table)
  routing_key = await _resolve_routing_key(engine, workspace_id)
  tool_dispatch = _make_tool_dispatch(dispatcher, routing_key)
  active_dispatchers.add(dispatcher)

  await engine.save_message(thread_id, "user", content)

  # Stream the STRUCTURED event feed (like the desktop UI) so tool calls are
  # surfaced to the client as `role="tool"` messages rather than being flattened
  # into text. Narration is streamed as `chat.delta` and flushed to an
  # `assistant` message before each tool result, so the persisted transcript
  # interleaves text blocks and tool cards in order. Tools are auto-approved on
  # this loopback path (the extension has its own approval model for its tools).
  current_text = ""
  pending_args: dict = {}
  # Turn cost/duration, reported on the terminal frame so the client can footnote the
  # turn with what it spent and how long it took.
  turn_started = time.monotonic()
  usage: Optional[TurnUsage] = None
  # Every exit path from here on MUST send exactly one terminal frame (`chat.done`
  # or `chat.error`). The client uses it to clear its "working" indicator, so a turn
  # that dies without one leaves the UI spinning forever with no way to recover —
  # which is exactly what used to happen when the inactivity timeout tore the turn
  # down via cancellation (a `CancelledError` is a BaseException, so it slipped past
  # the `except Exception` handler and no frame was ever sent).
  terminal_sent = False

  async def _send_terminal(error: Optional[str] = None, cancelled: bool = False, timeout_error: bool = False) -> None:
    nonlocal terminal_sent
    if terminal_sent:
      return
    terminal_sent = True
    # `thread_uuid` is echoed on every streaming/terminal frame so a client only
    # renders frames belonging to the thread it currently has open.
    data: dict = {
      "thread_uuid": thread_uuid,
      "duration_ms": int((time.monotonic() - turn_started) * 1000),
    }
    if usage is not None:
      data["usage"] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "requests": usage.requests,
      }
    if error is not None:
      data["error"] = error
      if timeout_error:
        data["timeout"] = True
      await send_json({"v": 1, "type": "chat.error", "id": corr, "data": data})
      return
    if cancelled:
      data["cancelled"] = True
    await send_json({"v": 1, "type": "chat.done", "id": corr, "data": data})

  try:
    async for event in engine.stream_chat_events(
      content, thread_id, workspace_id=workspace_id, model_cfg=model_cfg,
      auto_approve=True, tool_dispatch=tool_dispatch, provider_table=turn_table,
      context_blocks=context_blocks,
    ):
      if isinstance(event, TextDelta):
        current_text += event.content
        await send_json(
          {
            "v": 1, "type": "chat.delta", "id": corr,
            "data": {"delta": event.content, "thread_uuid": thread_uuid},
          }
        )
      elif isinstance(event, TurnUsage):
        usage = event
      elif isinstance(event, ToolCallStarted):
        # Capture the args now; the matching result arrives as a later event.
        pending_args[event.tool_call_id] = event.args
      elif isinstance(event, ToolCallResult):
        # Flush any narration that preceded this tool call as its own message,
        # then persist the tool call+result as a `role="tool"` row (which
        # publishes a message.created event the client renders as a tool card).
        if current_text.strip():
          await engine.save_message(thread_id, "assistant", current_text)
          current_text = ""
        tool_json = tool_block_to_json(
          tool_name=event.tool_name,
          args=pending_args.pop(event.tool_call_id, None),
          output=event.content,
          tool_call_id=event.tool_call_id,
          outcome=event.outcome,
        )
        await engine.save_message(thread_id, "tool", tool_json)

    # Normal completion: persist any trailing narration, then report done.
    if current_text.strip():
      await engine.save_message(thread_id, "assistant", current_text)
    await _send_terminal()
  except asyncio.CancelledError:
    # The turn was cancelled: either the user pressed stop (`chat.cancel` recorded
    # this correlation id) or something tore it down (inactivity timeout, connection
    # teardown, engine shutdown). Preserve whatever the model had already said, tell
    # the client the turn is over so it stops showing "working", then let the
    # cancellation continue to propagate — never swallow it.
    user_stopped = bool(cancelled_turns and corr in cancelled_turns)
    logger.info("chat.send turn cancelled (user_stopped=%s)", user_stopped)
    try:
      if current_text.strip():
        await engine.save_message(thread_id, "assistant", current_text)
      if user_stopped:
        # A deliberate stop is a normal end to the turn, not an error.
        await _send_terminal(cancelled=True)
      else:
        # An unhandled cancellation (likely a timeout) is treated as an error.
        await _send_terminal("The turn was cancelled before it completed.")
    except Exception:
      logger.debug("failed to notify client of cancelled turn", exc_info=True)
    raise
  except TimeoutError as exc:  # Model inactivity timeout - handle specially
    logger.exception("chat.send stream failed due to timeout")
    try:
      if current_text.strip():
        await engine.save_message(thread_id, "assistant", current_text)
    except Exception:
      logger.debug("failed to persist partial reply", exc_info=True)
    await _send_terminal(str(exc), timeout_error=True)
  except Exception as exc:  # surface model/config errors to the client
    logger.exception("chat.send stream failed")
    # Keep the partial reply: a turn that fails midway (e.g. the model stalled) has
    # usually already streamed useful text into the client's view, and dropping it
    # here would make that text vanish on the next thread reload.
    try:
      if current_text.strip():
        await engine.save_message(thread_id, "assistant", current_text)
    except Exception:
      logger.debug("failed to persist partial reply", exc_info=True)
    await _send_terminal(str(exc))
  finally:
    # The Master_Engine role and its routing state are turn-scoped: drop the
    # per-turn dispatcher so nothing designates this Engine as Master_Engine
    # for a later turn on the same Thread (Requirement 5.4).
    active_dispatchers.discard(dispatcher)
    if cancelled_turns is not None and isinstance(corr, str):
      cancelled_turns.discard(corr)
