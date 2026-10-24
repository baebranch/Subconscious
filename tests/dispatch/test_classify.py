"""Unit tests for connection classification (dispatch/classify.py).

Covers classify_connection's loopback/remote determination and
should_reject_tool_register's pure predicate over ConnectionKind.

Validates: Requirements 4.2, 4.3, 4.4
"""

from subconscious.dispatch.classify import (
    ConnectionKind,
    classify_connection,
    should_reject_tool_register,
)


class TestClassifyConnection:
    def test_ipv4_loopback_is_local_client(self) -> None:
        assert classify_connection("127.0.0.1") == ConnectionKind.LOCAL_CLIENT

    def test_ipv6_loopback_is_local_client(self) -> None:
        assert classify_connection("::1") == ConnectionKind.LOCAL_CLIENT

    def test_non_loopback_address_is_remote_client(self) -> None:
        assert classify_connection("203.0.113.5") == ConnectionKind.REMOTE_CLIENT

    def test_hostname_is_remote_client(self) -> None:
        assert classify_connection("some-host.example.com") == ConnectionKind.REMOTE_CLIENT

    def test_empty_host_is_remote_client(self) -> None:
        assert classify_connection("") == ConnectionKind.REMOTE_CLIENT


class TestShouldRejectToolRegister:
    def test_remote_client_is_rejected(self) -> None:
        assert should_reject_tool_register(ConnectionKind.REMOTE_CLIENT) is True

    def test_local_client_is_not_rejected(self) -> None:
        assert should_reject_tool_register(ConnectionKind.LOCAL_CLIENT) is False
