"""
Structured streaming events emitted by ``Engine.stream_chat_events``.

The desktop chat UI consumes these so a single agent turn can be rendered as
multiple message bubbles
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class TextDelta:
  """A chunk of model-generated narration text for the current text block."""
  content: str


@dataclass
class ToolCallStarted:
  """The model has requested a tool call.

  ``args`` is whatever pydantic-ai reports for the call — either a decoded
  ``dict`` or a raw JSON string — captured before the tool executes.
  """
  tool_name: str
  args: Any
  tool_call_id: str


@dataclass
class ToolCallResult:
  """The result of a previously-started tool call.

  ``outcome`` mirrors pydantic-ai's ``ToolReturnPart.outcome`` and is one of
  ``"success"``, ``"failed"`` or ``"denied"``.
  """
  tool_name: str
  content: Any
  tool_call_id: str
  outcome: str = "success"


@dataclass
class ApprovalRequest:
  """The model wants to run a tool that the policy has gated for approval.

  The stream pauses on this event until the UI resolves the decision (via
  ``Engine.resolve_approval``). ``operation`` is ``"query"`` or ``"mutation"``
  so the UI can explain why approval is being asked.
  """
  tool_name: str
  args: Any
  tool_call_id: str
  operation: str = "mutation"


@dataclass
class ApprovalResolved:
  """Emitted after the user approves/denies an ``ApprovalRequest`` so the UI
  can update the corresponding tool bubble."""
  tool_call_id: str
  approved: bool


@dataclass
class TurnUsage:
  """Token usage accumulated over a whole agent turn.

  Emitted once, last, so a client can render an end-of-turn footnote ("12.3k
  tokens"). Counts cover every model request the turn made, including the extra
  round trips a tool-calling turn needs. Any field the provider didn't report is
  ``0``.
  """
  input_tokens: int = 0
  output_tokens: int = 0
  total_tokens: int = 0
  requests: int = 0


# Discriminated union of everything the event stream can yield.
StreamEvent = Union[
  TextDelta, ToolCallStarted, ToolCallResult, ApprovalRequest, ApprovalResolved, TurnUsage
]


# ---------------------------------------------------------------------------
# Attached-context stripping
# ---------------------------------------------------------------------------

# Preambles used when client-attached context is folded into a model prompt (see
# ``Engine._with_context_blocks`` / ``Engine._build_prompt_with_attachments``).
_CONTEXT_PREAMBLES = (
  "The user attached the following context to this message.",
  "The user has attached the following files and folders.",
)

# Separator placed between the attached context and the user's own message.
_CONTEXT_SEPARATOR = "\n\n---\n\n"

# A single client-resolved mention block, e.g. "[Context — File: src/main.ts]".
_CONTEXT_BLOCK_PREFIX = "[Context"


def strip_attached_context(text: str) -> str:
  """Return *text* with any attached file/folder/terminal context removed, leaving
  just what the user actually wrote.

  Used for deriving a thread's automatic name: attached context is inlined at the
  START of the text, so a naive "first few words" title on a message with an
  attachment produced useless names like ``"[Context - File."`` instead of naming
  the user's actual request.

  Two forms are handled, in order:

  1. A whole-prompt preamble ("The user attached the following context...") followed
     by the ``---`` separator — everything up to and including the last separator is
     dropped.
  2. Leading ``[Context ...]`` blocks (the per-mention form), skipped line-wise while
     respecting fenced code blocks so a fence's blank lines don't end the skip early.
  3. Inline mentions like ``#filename`` at the start of the message are also stripped
     since they don't contribute to the user's actual request.

  Returns the original text when it carries no recognisable attached context, and
  falls back to the original when stripping would leave nothing behind.
  """
  if not text:
    return text

  working = text.lstrip()

  if working.startswith(_CONTEXT_PREAMBLES) and _CONTEXT_SEPARATOR in working:
    working = working.rsplit(_CONTEXT_SEPARATOR, 1)[1]

  lines = working.split("\n")
  index = 0
  in_fence = False
  skipping = False
  while index < len(lines):
    line = lines[index]
    stripped_line = line.strip()

    if skipping:
      if stripped_line.startswith("```"):
        in_fence = not in_fence
        index += 1
        continue
      if not in_fence and stripped_line == "":
        skipping = False
      index += 1
      continue

    if stripped_line == "":
      index += 1
      continue
    if stripped_line.startswith(_CONTEXT_BLOCK_PREFIX):
      skipping = True
      index += 1
      continue
    break

  remainder = "\n".join(lines[index:]).strip()
  
  # Also strip leading mentions like #filename from the start of the remainder
  # These are inserted inline in the composer as "#filename " and don't contribute
  # to the user's actual request content
  while remainder and remainder.lstrip().startswith("#"):
    # Find the end of the mention token (ends at whitespace or end of string)
    match = re.match(r'^(#\S*\s*)(.*)', remainder.lstrip(), re.DOTALL)
    if match:
      remainder = match.group(2).strip()
    else:
      break

  return remainder if remainder else text.strip()


def _coerce_jsonable(value: Any) -> Any:
  """Best-effort convert an arbitrary tool arg/result into JSON-friendly data.

  Tool args arrive as a ``dict`` or a JSON string; results can be strings,
  dicts, lists, or richer objects. We normalize to something ``json.dumps``
  can render so the UI can pretty-print it, falling back to ``str`` for
  anything exotic.
  """
  if value is None or isinstance(value, (dict, list, int, float, bool)):
    return value
  if isinstance(value, str):
    # A JSON string (common for tool args) is decoded so it renders as a
    # structured block rather than an escaped one-liner.
    text = value.strip()
    if text and text[0] in "[{":
      try:
        return json.loads(text)
      except (ValueError, TypeError):
        return value
    return value
  # Dataclasses / pydantic models / other objects: try their dict form, else str.
  for attr in ("model_dump", "dict"):
    method = getattr(value, attr, None)
    if callable(method):
      try:
        return method()
      except Exception:
        break
  return str(value)


def tool_block_to_json(
  tool_name: str,
  args: Any,
  output: Any,
  tool_call_id: str = "",
  outcome: str = "success",
) -> str:
  """Serialize a tool call+result into the JSON stored on a ``role="tool"`` row.

  This is the persisted/display form consumed by ``ToolMessage`` in the UI.
  """
  document = {
    "tool_name": tool_name,
    "tool_call_id": tool_call_id,
    "outcome": outcome,
    "input": _coerce_jsonable(args),
    "output": _coerce_jsonable(output),
  }
  return json.dumps(document, default=str)
