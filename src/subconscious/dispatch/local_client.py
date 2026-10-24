"""LocalClientConnection: the `ProviderConnection` backing a Local_Client over
its loopback WebSocket.

A Local_Client that has completed its handshake and sent a `tool.register`
frame becomes a Provider in the Engine's `ProviderTable` (Requirement 2). When
the `ToolDispatcher` resolves a tool call to such a Provider it invokes
`send_tool_call`, which serialises the call as the existing, unchanged
`tool.call` frame (`subconscious-code/src/tools/types.ts`) and writes it on the
client's own socket (Requirement 8.2, 8.3, 9.3). The matching `tool.result`
frame is correlated back to the pending call by the `ToolDispatcher`.

This adapter is deliberately thin: it holds only the socket's JSON-send
callable and an availability flag toggled on disconnect, so the dispatch
layer never needs to know it is talking to a WebSocket rather than a future
engine-to-engine transport (Requirement 8).
"""

from __future__ import annotations

from typing import Awaitable, Callable


class LocalClientConnection:
  """`ProviderConnection` for a loopback-connected Local_Client.

     `send_json` is the connection's own JSON-frame writer (e.g.
     `WebSocket.send_json`); it is the only capability the dispatch layer
     needs to reach this Provider."""

  def __init__(self, send_json: Callable[[dict], Awaitable[None]]) -> None:
    self._send_json = send_json
    self._available = True

  async def send_tool_call(self, correlation_id: str, tool_id: str, input: object) -> None:
    """Serialise the resolved tool call as the unchanged `tool.call` frame and
       write it on this Local_Client's socket (Requirement 8.2, 9.3). The
       correlation id is echoed back on the client's `tool.result`."""
    await self._send_json(
      {
        "v": 1,
        "type": "tool.call",
        "id": correlation_id,
        "data": {"tool": tool_id, "input": input},
      }
    )

  @property
  def is_available(self) -> bool:
    return self._available

  def mark_unavailable(self) -> None:
    """Called when the underlying socket disconnects so a Provider that has
       not yet been removed from the table is not treated as dispatchable."""
    self._available = False
