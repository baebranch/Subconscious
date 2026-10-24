"""Unit tests for the provider-tool bridge (dispatch/provider_tools.py).

These cover the gap that made a connected Local_Client's registered tools
(e.g. the VS Code extension's workspace-scoped `create_file`/`edit_file`)
usable by the agent at all:

  - `build_provider_tools` builds one agent-callable tool per Provider-advertised
    tool id, using the advertised descriptor for its name/description/schema.
  - Invoking a built tool routes through `EngineContext.tool_dispatch` (the
    per-turn `ToolDispatcher`-backed callable) — i.e. the existing tool
    dispatch is actually called — returning the output on success and raising
    a retryable `ModelRetry` on a routed failure or when no client is connected.
  - `filter_local_tools_for_providers` gives a connected client's tools
    precedence over the engine's own same-named local built-ins, so an IDE
    client's sandboxed tool wins over the engine's unsandboxed equivalent.

Validates: connected-client tool bridging + precedence (single-engine spec).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pydantic_ai import ModelRetry

from subconscious.dispatch import (
  ProviderTable,
  build_provider_tools,
  filter_local_tools_for_providers,
)


class _FakeConnection:
  async def send_tool_call(self, correlation_id: str, tool_id: str, input: object) -> None:  # noqa: A002
    return None

  @property
  def is_available(self) -> bool:
    return True


def _table_with_create_file() -> ProviderTable:
  table = ProviderTable()
  table.upsert(
    provider_id="client-1",
    kind="local_client",
    connection=_FakeConnection(),
    tool_ids=frozenset({"create_file"}),
    execution_profile_root="C:/ws",
    tool_descriptors={
      "create_file": {
        "description": "Create a workspace file",
        "inputSchema": {
          "type": "object",
          "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
          "required": ["path", "content"],
        },
      }
    },
  )
  return table


def _ctx_with_dispatch(dispatch):
  return SimpleNamespace(deps=SimpleNamespace(tool_dispatch=dispatch))


class TestBuildProviderTools:
  def test_builds_one_tool_per_advertised_id_using_descriptor(self) -> None:
    tools = build_provider_tools(_table_with_create_file())
    assert [t.name for t in tools] == ["create_file"]
    assert tools[0].description == "Create a workspace file"

  def test_empty_table_yields_no_tools(self) -> None:
    assert build_provider_tools(ProviderTable()) == []

  @pytest.mark.asyncio
  async def test_invoking_tool_calls_tool_dispatch_and_returns_output(self) -> None:
    tool = build_provider_tools(_table_with_create_file())[0]
    calls: list[tuple[str, object]] = []

    async def dispatch(tool_id: str, tool_input: object) -> dict:
      calls.append((tool_id, tool_input))
      return {"ok": True, "output": f"created {tool_input['path']}"}

    out = await tool.function(_ctx_with_dispatch(dispatch), path="C:/ws/a.txt", content="hi")

    assert calls == [("create_file", {"path": "C:/ws/a.txt", "content": "hi"})]
    assert out == "created C:/ws/a.txt"

  @pytest.mark.asyncio
  async def test_routed_failure_raises_model_retry_with_kind(self) -> None:
    tool = build_provider_tools(_table_with_create_file())[0]

    async def dispatch(tool_id: str, tool_input: object) -> dict:
      return {"ok": False, "error": {"kind": "rejected", "message": "declined"}}

    with pytest.raises(ModelRetry) as excinfo:
      await tool.function(_ctx_with_dispatch(dispatch), path="C:/ws/a.txt", content="hi")
    assert "rejected" in str(excinfo.value)
    assert "declined" in str(excinfo.value)

  @pytest.mark.asyncio
  async def test_missing_dispatch_raises_model_retry(self) -> None:
    tool = build_provider_tools(_table_with_create_file())[0]
    ctx = SimpleNamespace(deps=SimpleNamespace(tool_dispatch=None))
    with pytest.raises(ModelRetry):
      await tool.function(ctx, path="C:/ws/a.txt", content="hi")


def _local_create_file(ctx, path, content):  # noqa: ANN001, ANN202 - local built-in stand-in
  return "LOCAL"
_local_create_file.__name__ = "create_file"


def _local_replace_in_file(ctx, path, old, new):  # noqa: ANN001, ANN202
  return "LOCAL"
_local_replace_in_file.__name__ = "replace_in_file"


def _local_run_terminal_command(ctx, command, cwd="~"):  # noqa: ANN001, ANN202
  return "LOCAL"
_local_run_terminal_command.__name__ = "run_terminal_command"


def _local_calculate(ctx, expression):  # noqa: ANN001, ANN202
  return "LOCAL"
_local_calculate.__name__ = "calculate"


class TestLocalToolPrecedence:
  def test_coding_client_suppresses_all_engine_coding_tools_but_keeps_utilities(self) -> None:
    # A coding client advertises `create_file`, so the engine's own
    # unsandboxed filesystem/terminal tools all defer to it — including
    # `replace_in_file`/`run_terminal_command`, which don't exactly match a
    # provider id — while a utility tool like `calculate` is retained.
    table = _table_with_create_file()
    kept = filter_local_tools_for_providers(
      [_local_create_file, _local_replace_in_file, _local_run_terminal_command, _local_calculate],
      table,
    )
    assert [f.__name__ for f in kept] == ["calculate"]

  def test_non_coding_client_only_drops_exact_name_collisions(self) -> None:
    # A connected client that advertises only a non-coding tool id must not
    # strip the engine's coding tools; only an exact-name collision is dropped.
    table = ProviderTable()
    table.upsert(
      provider_id="c1",
      kind="local_client",
      connection=_FakeConnection(),
      tool_ids=frozenset({"get_weather"}),
      tool_descriptors={"get_weather": {"description": "w", "inputSchema": {"type": "object"}}},
    )

    def get_weather(ctx, city):  # noqa: ANN001, ANN202
      return "LOCAL"

    kept = filter_local_tools_for_providers(
      [get_weather, _local_replace_in_file, _local_calculate], table
    )
    # get_weather collides by name → dropped; coding tools retained (no coding client).
    assert sorted(f.__name__ for f in kept) == ["calculate", "replace_in_file"]

  def test_no_table_keeps_all_local_tools(self) -> None:
    kept = filter_local_tools_for_providers([_local_create_file], None)
    assert [f.__name__ for f in kept] == ["create_file"]

  def test_empty_table_keeps_all_local_tools(self) -> None:
    kept = filter_local_tools_for_providers([_local_create_file], ProviderTable())
    assert [f.__name__ for f in kept] == ["create_file"]
