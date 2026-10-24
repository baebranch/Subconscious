"""Provider-tool bridge: turns a turn's Provider_Table entries into
pydantic-ai `Tool` objects the agent can actually call.

Without this module, `ToolDispatcher`/`ProviderTable` only *route* a tool
call once the model has already decided to call a tool id — but nothing
ever told the model those tool ids exist, because the agent's `tools=`
list was built exclusively from the engine's own local `desktop_tools`
callables (`tools/__init__.py`, `desktop_tools/__init__.py`). A connected
Local_Client (e.g. the VS Code extension) could complete its handshake,
register `create_file`/`edit_file`/`run_terminal`/etc., and those
registrations would sit in the Provider_Table unused: the model never saw
them as callable tools, so every file/terminal operation fell through to
the engine's own unsandboxed `desktop_tools.filesystem`/`terminal`
functions instead.

`build_provider_tools` closes that gap: for every tool id currently
advertised by at least one Provider in a turn's `ProviderTable` snapshot,
it builds one `pydantic_ai.tools.Tool` (via `Tool.from_schema`) whose
function body calls the turn's `tool_dispatch` callable — the same
`async (tool_id, input) -> {"ok": ..., "output"/"error": ...}` bridge
already built by `_make_tool_dispatch` in `api/app.py` and threaded through
`EngineContext.tool_dispatch`. The tool's name, description, and input
schema come from the `RegisteredToolDescriptor` the Provider advertised in
its `tool.register` frame (`ProviderTable.descriptor_for`), so the model
sees the *real* contract (e.g. the extension's workspace-scoped
`create_file`) rather than a generic passthrough.

Local-tool precedence
----------------------
When a connected Provider (typically the IDE extension) advertises a tool
id that collides with one of the engine's own local `desktop_tools`
callables (e.g. `create_file`, `list_directory`, `run_terminal_command`),
the Provider's tool wins: local coding-oriented tools exist mainly as a
fallback for turns with no connected IDE client, and a connected client's
tools are workspace-scoped/sandboxed in a way the engine's own tools are
not (see `pathGuard.ts` vs `desktop_tools/filesystem.py`'s unsandboxed
`_resolve_path`).

`filter_local_tools_for_providers` implements this precedence in two tiers:

1. **Exact-name collisions** — any local callable whose `__name__` equals a
   Provider-advertised tool id is dropped, so there is exactly one tool per
   id (the Provider's bridge tool).
2. **Coding-category deferral** — when the connected Provider is a *coding*
   client (it advertises any recognised coding tool id such as
   `create_file`/`edit_file`/`run_terminal`), the engine's own unsandboxed
   filesystem/terminal callables are dropped wholesale — even the ones whose
   names don't exactly match a Provider id (e.g. `replace_in_file`,
   `run_terminal_command`, `move_to_trash`, `get_directory_tree`,
   `find_symbol`). This closes the hole where the model could still reach an
   unsandboxed engine file/terminal tool (which defaults its path/cwd to the
   home directory) instead of the client's workspace-scoped equivalent.

Non-coding utility tools (calculator, weather, memory, notes, time,
contacts, knowledge, clipboard, images, system info, ...) are always kept,
and turns with no connected client — or with a non-coding client — retain
the engine's own tools unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from pydantic_ai.tools import RunContext, Tool

from .provider_table import ProviderTable

logger = logging.getLogger("subconscious")

# Fallback schema used when a Provider's `tool.register` frame omitted (or has
# not yet supplied) an `inputSchema` for one of its advertised tool ids. An
# empty permissive object schema still lets the model pass arbitrary
# arguments through; the Provider itself is the source of truth for
# validation once the call is dispatched.
_FALLBACK_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": True}

# Tool ids that mark a connected Provider as a *coding* client (the VS Code
# extension's built-in Coding_Tools, see `subconscious-code/src/tools/registry.ts`).
# When any of these is advertised, the engine's own unsandboxed filesystem/terminal
# tools defer to the client for the turn.
#
# `grep_search` is intentionally absent: it was removed from the extension's tool
# catalogue (a hand-rolled recursive file-content walk isn't cross-platform
# reliable) in favor of `run_terminal` and `search_files`.
_CODING_CLIENT_TOOL_IDS: frozenset[str] = frozenset({
  "create_file",
  "edit_file",
  "read_file",
  "list_directory",
  "search_files",
  "run_terminal",
  "run_tests",
  "run_task",
  "get_diagnostics",
  "list_code_usages",
  "get_changes",
  "get_workspace_context",
  "start_debug",
})

# The engine's own local filesystem/terminal callables (function `__name__`s from
# `desktop_tools/filesystem.py` and `desktop_tools/terminal.py`). These are
# unsandboxed (`_resolve_path` follows `~`/absolute paths anywhere) and are
# suppressed for a turn when a coding client is connected so all file/terminal
# work is routed through the client's workspace-scoped tools. Read-only host-info
# tools (`get_env_var`, `get_system_info`) are intentionally NOT included so
# system-aware answers still work alongside a connected client.
_ENGINE_CODING_TOOL_NAMES: frozenset[str] = frozenset({
  # filesystem.py
  "read_file",
  "read_range",
  "search_in_file",
  "search_files",
  "list_directory",
  "create_file",
  "get_file_info",
  "move_to_trash",
  "replace_in_file",
  "run_terminal_command",
  "get_directory_tree",
  "find_symbol",
  # terminal.py
  "run_command",
  "open_terminal_session",
  "run_in_session",
  "close_terminal_session",
})


def _make_provider_tool_function(tool_id: str) -> Callable[..., Any]:
  """Build the async function backing a single Provider-bridged Tool.

  The returned function ignores its keyword arguments' individual shape (the
  model already validated them against the advertised schema on pydantic-ai's
  side) and forwards the whole kwargs dict as the tool's `input`, via the
  turn's `EngineContext.tool_dispatch` callable. Raises `ModelRetry` — which
  pydantic-ai surfaces back to the model as a retryable tool error rather than
  crashing the run — when no `tool_dispatch` is wired (a local/CLI turn with
  no connected Provider) or when the dispatched call fails.
  """

  async def _call_provider_tool(ctx: RunContext[Any], **kwargs: Any) -> Any:
    from pydantic_ai import ModelRetry  # local import: keep module import light

    dispatch = getattr(ctx.deps, "tool_dispatch", None)
    if dispatch is None:
      raise ModelRetry(
        f"Tool '{tool_id}' is provided by a connected client, but no client is "
        "connected for this turn. Reconnect the client (e.g. the IDE extension) "
        "and try again."
      )

    result = await dispatch(tool_id, kwargs)
    if not isinstance(result, dict):
      return result

    if result.get("ok"):
      return result.get("output")

    error = result.get("error") or {}
    message = error.get("message") or f"Tool '{tool_id}' failed."
    kind = error.get("kind")
    # Surface the typed error kind so the model can distinguish, e.g., a
    # declined confirmation ("rejected") from a genuine failure.
    raise ModelRetry(f"[{kind}] {message}" if kind else message)

  _call_provider_tool.__name__ = f"provider_{tool_id}"
  return _call_provider_tool


def build_provider_tools(provider_table: ProviderTable) -> list[Tool]:
  """Build one agent-callable `Tool` per tool id currently advertised by any
  Provider in `provider_table` (a turn's `ProviderTable.snapshot()`).

  Each tool's name is the bare tool id (e.g. `"create_file"`), its
  description/input schema come from the earliest-registered Provider's
  advertised descriptor for that id (`ProviderTable.descriptor_for`), and
  invoking it routes through `EngineContext.tool_dispatch` — the per-turn
  `ToolDispatcher`-backed callable — so the *routing* decision (which
  Provider actually executes it) is still made independently at call time,
  exactly as designed by `ToolDispatcher`/`resolve_provider`. Returns an
  empty list when no Provider currently offers any tools (e.g. no client is
  connected), in which case the agent falls back to the engine's own local
  tools.
  """
  tools: list[Tool] = []
  for tool_id in sorted(provider_table.known_tool_ids()):
    descriptor = provider_table.descriptor_for(tool_id) or {}
    description = descriptor.get("description") or f"Tool '{tool_id}' provided by a connected client."
    schema = descriptor.get("inputSchema") or _FALLBACK_SCHEMA
    try:
      tool = Tool.from_schema(
        function=_make_provider_tool_function(tool_id),
        name=tool_id,
        description=description,
        json_schema=schema,
        takes_ctx=True,
      )
    except Exception:
      # A malformed schema from a misbehaving client must not break the
      # whole turn's tool list; skip just that one tool id.
      logger.warning("Failed to build bridge tool for '%s'; skipping", tool_id, exc_info=True)
      continue
    tools.append(tool)
  return tools


def filter_local_tools_for_providers(
  local_tools: list[Callable[..., Any]],
  provider_table: Optional[ProviderTable],
) -> list[Callable[..., Any]]:
  """Give a connected client's tools precedence over the engine's own local
  built-ins for this turn (see module docstring for the full policy).

  Tier 1 — always drop any local callable whose `__name__` collides with a
  tool id currently advertised by a connected Provider, so there is exactly
  one tool per id (the Provider's bridge tool).

  Tier 2 — when the connected Provider is a coding client (it advertises any
  id in `_CODING_CLIENT_TOOL_IDS`), additionally drop the engine's own
  unsandboxed filesystem/terminal callables (`_ENGINE_CODING_TOOL_NAMES`),
  even those without an exact-name Provider match, so the model cannot reach
  a home-directory-defaulting engine tool instead of the client's
  workspace-scoped equivalent.

  Local tools are returned unchanged when no client is connected, and only
  exact-name collisions are dropped for a connected non-coding client, so
  single-client, no-client, and non-coding-client turns keep the engine's
  utility tools intact.
  """
  if provider_table is None:
    return local_tools
  provider_ids = provider_table.known_tool_ids()
  if not provider_ids:
    return local_tools

  drop: set[str] = set(provider_ids)
  if provider_ids & _CODING_CLIENT_TOOL_IDS:
    drop |= _ENGINE_CODING_TOOL_NAMES

  return [fn for fn in local_tools if getattr(fn, "__name__", None) not in drop]
