# Base cross-platform tool registry for Subconscious.
# Contains tools that work on all platforms (desktop, web, mobile, server).
# Platform-specific registries (desktop_tools, mobile_tools, server_tools)
# extend this by importing and adding their additional tools.

from typing import Callable, Any
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Tool operation classification (query vs mutation)
# ---------------------------------------------------------------------------
# Every tool is roughly classified as a "query" (reads/derives data with no
# side effects) or a "mutation" (creates, updates, or deletes data / has side
# effects). This drives human-in-the-loop approval: a workspace/thread can
# independently require approval for queries and/or mutations. Unknown tools
# default to "mutation" so a newly-added tool is gated (approval-required) by
# default rather than silently running unapproved.

QUERY = "query"
MUTATION = "mutation"

# Explicit classification for every known built-in and desktop tool.
_MUTATION_TOOLS = frozenset({
  # todo
  "add_todo", "update_todo", "complete_todo", "delete_todo",
  # memory
  "remember", "forget", "forget_all",
  # notes
  "save_note", "delete_note",
  # contacts
  "add_contact", "update_contact", "delete_contact",
  # terminal
  "run_command", "run_terminal_command", "open_terminal_session",
  "run_in_session", "close_terminal_session",
  # settings
  "update_app_setting", "set_theme_mode",
  # images (write output files)
  "optimize_image", "convert_image", "batch_optimize_images",
  "batch_convert_image", "resize_image", "batch_resize_images",
  "images_to_pdf", "pdf_to_images",
  # filesystem
  "create_file", "move_to_trash", "replace_in_file",
  # clipboard
  "write_clipboard",
  # desktop automation (control the local machine's input devices)
  "move_mouse", "click_mouse", "double_click_mouse", "drag_mouse",
  "scroll_mouse", "type_text", "press_key", "press_hotkey",
})

_QUERY_TOOLS = frozenset({
  # time
  "get_current_time", "get_current_date", "convert_timezone", "list_common_timezones",
  # calculator
  "calculate", "convert_units", "list_supported_units",
  # weather
  "get_weather", "get_forecast",
  # todo / memory / notes / contacts reads
  "list_todos", "recall", "list_memories", "list_notes", "get_note",
  "list_contacts", "find_contact",
  # web
  "fetch_page", "search_web", "check_connectivity", "speed_test",
  # knowledge retrieval (RAG / GraphRAG) — read-only
  "search_knowledge", "search_knowledge_graph",
  # terminal reads
  "get_env_var", "get_system_info",
  # settings reads
  "get_app_setting",
  # search / filesystem reads
  "search_fs", "read_file", "read_range", "search_in_file", "search_files",
  "list_directory", "get_file_info", "get_directory_tree", "find_symbol",
  # clipboard reads
  "read_clipboard",
  # desktop automation reads (screen inspection, no side effects)
  "get_screen_size", "get_mouse_position", "get_pixel_color",
  "capture_screenshot", "locate_on_screen",
})

# Name-prefix heuristic for tools not in the explicit maps (e.g. user-added).
_QUERY_PREFIXES = (
  "get_", "list_", "read_", "find_", "search_", "fetch_", "check_",
  "view_", "show_", "describe_", "lookup_", "recall",
)


def classify_operation(tool_name: str) -> str:
  """Return ``"query"`` or ``"mutation"`` for a tool name.

  Explicit classifications win; otherwise a read-oriented name prefix marks a
  query and everything else defaults to ``"mutation"`` (approval-gated).
  """
  if tool_name in _MUTATION_TOOLS:
    return MUTATION
  if tool_name in _QUERY_TOOLS:
    return QUERY
  for prefix in _QUERY_PREFIXES:
    if tool_name.startswith(prefix):
      return QUERY
  return MUTATION


@dataclass
class EngineContext:
  """
  Dependency object injected into every tool via pydantic-ai RunContext.
  Gives tools access to the DB session factory, current workspace/thread
  identifiers, and the data directory path (for filesystem scoping).
  """
  db: Any                  # Database instance (db.session.Database)
  workspace_id: int
  thread_id: int
  engine: Any = None       # Subconscious Engine instance
  data_dir: str = ""
  # Resolved human-in-the-loop approval policy for this run:
  # {"query": bool, "mutation": bool} where True == approval required.
  approval_config: dict = field(default_factory=lambda: {"query": True, "mutation": True})
  # Per-turn Provider_Table-backed tool dispatcher, injected by the WebSocket
  # `chat.send` handler (single-engine-architecture spec, task 11.2). When set,
  # it is an ``async (tool_id, input) -> ToolResult-dict`` callable that routes
  # a tool call through the Master_Engine's fresh ``ProviderTable.snapshot()``
  # via ``ToolDispatcher.call`` — replacing the legacy direct single-client
  # tool-call path. ``None`` on in-process (desktop/CLI) turns that have no
  # connected Provider clients.
  tool_dispatch: Any = None


# ---------------------------------------------------------------------------
# Base Registry
# ---------------------------------------------------------------------------

class BaseToolRegistry:
  """
  Maps tool slugs to lists of pydantic-ai compatible tool callables.
  Contains only cross-platform tools safe to use on all platforms.

  Platform-specific registries subclass this and call super().__init__()
  then add their own tools.

  Usage
  -----
  registry = BaseToolRegistry()
  tools = registry.get_tools(["time", "calculator"])
  agent = Agent(model=..., tools=tools, deps_type=EngineContext)
  """

  def __init__(self):
    self._registry: dict[str, list[Callable]] = {}
    self._load_base_tools()

  def _load_base_tools(self):
    """Load tools that are safe on every platform (no OS/desktop APIs)."""
    from . import time_tools, calculator, weather
    from . import todo, memory, notes, contacts, knowledge

    modules = {
      "time":       time_tools,
      "calculator": calculator,
      "weather":    weather,
      "todo":       todo,
      "memory":     memory,
      "notes":      notes,
      "contacts":   contacts,
      "knowledge":  knowledge,
    }

    for slug, module in modules.items():
      self._registry[slug] = getattr(module, "TOOLS", [])

  def register(self, slug: str, tools: list[Callable]) -> None:
    """Register an external or user-defined tool set under a slug."""
    self._registry[slug] = tools

  def get_tools(self, slugs: list[str]) -> list[Callable]:
    """Return the flat list of callables for the requested slugs."""
    result = []
    for slug in slugs:
      result.extend(self._registry.get(slug, []))
    return result

  def all_slugs(self) -> list[str]:
    return list(self._registry.keys())

  def catalog(self) -> dict[str, list[dict]]:
    """
    Return a hierarchy of the built-in tools as:
      { slug: [ {"name": <callable name>, "doc": <first docstring line>}, ... ] }

    Used by the UI to render the per-workspace / per-thread tool toggle tree
    (top-level slug toggle + an individual toggle per tool callable).
    """
    result: dict[str, list[dict]] = {}
    for slug, tools in self._registry.items():
      entries: list[dict] = []
      for fn in tools:
        doc = (getattr(fn, "__doc__", "") or "").strip()
        first_line = doc.split("\n", 1)[0].strip() if doc else ""
        entries.append({"name": getattr(fn, "__name__", str(fn)), "doc": first_line})
      result[slug] = entries
    return result

  def get_tools_for_config(self, config: dict) -> list[Callable]:
    """
    Resolve enabled tool callables from a tools_config dict of the form:

      {"builtin": {slug: {"enabled": bool, "tools": {name: bool}}}}

    A missing slug or tool entry defaults to enabled (True), preserving the
    legacy "all tools" behaviour for unconfigured workspaces/threads.
    When a slug's top-level "enabled" is False, all of its tools are skipped.
    When the "builtin_enabled" master flag is False, no built-in tools are
    returned at all.
    """
    config = config or {}
    if not config.get("builtin_enabled", True):
      return []
    builtin = config.get("builtin", {})
    result: list[Callable] = []
    for slug, tools in self._registry.items():
      slug_cfg = builtin.get(slug, {})
      if not slug_cfg.get("enabled", True):
        continue
      tool_states = slug_cfg.get("tools", {})
      for fn in tools:
        name = getattr(fn, "__name__", "")
        if tool_states.get(name, True):
          result.append(fn)
    return result


# Convenience alias — the base registry is also usable as ToolRegistry
ToolRegistry = BaseToolRegistry

# Module-level singleton for convenience
registry = BaseToolRegistry()
