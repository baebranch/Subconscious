# Engine dispatch package for the single-engine architecture (Provider_Table
# based tool routing). See design.md ("Single Engine Architecture" spec) for
# the full component topology. Modules are added incrementally per the
# implementation plan's task list; this package currently exposes the
# `Provider` dataclass, its connection Protocols, the `ProviderTable`
# aggregation, Claim resolution, connection classification, peer approval
# enforcement, and the `ToolDispatcher` send/await lifecycle.

from .provider import Provider, ProviderConnection, PeerEngineTransport
from .local_client import LocalClientConnection
from .provider_table import ProviderTable
from .claim import ResolutionResult, resolve_provider
from .classify import ConnectionKind, classify_connection, should_reject_tool_register
from .peer_approval import PeerApprovalSettings, PeerApprovalGate, ApprovalDeclinedError
from .handshake import HandshakeResult, HandshakeService, should_accept_frame
from .dispatcher import DispatchOutcome, NoProviderError, ToolDispatcher
from .provider_tools import build_provider_tools, filter_local_tools_for_providers

__all__ = [
  "Provider",
  "ProviderConnection",
  "PeerEngineTransport",
  "LocalClientConnection",
  "ProviderTable",
  "ResolutionResult",
  "resolve_provider",
  "ConnectionKind",
  "classify_connection",
  "should_reject_tool_register",
  "PeerApprovalSettings",
  "PeerApprovalGate",
  "ApprovalDeclinedError",
  "HandshakeResult",
  "HandshakeService",
  "should_accept_frame",
  "DispatchOutcome",
  "NoProviderError",
  "ToolDispatcher",
  "build_provider_tools",
  "filter_local_tools_for_providers",
]
