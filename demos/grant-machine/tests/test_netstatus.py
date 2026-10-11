from __future__ import annotations

import socket

from app.netstatus import classify_host, count_app_public_connections, probe_internet


def test_classify_loopback():
    assert classify_host("127.0.0.1") == "loopback"
    assert classify_host("::1") == "loopback"


def test_classify_private_rfc1918():
    assert classify_host("192.168.1.5") == "private"
    assert classify_host("10.0.0.1") == "private"
    assert classify_host("172.16.0.1") == "private"


def test_classify_cgnat_treated_as_private():
    assert classify_host("100.64.0.1") == "private"
    assert classify_host("100.100.50.1") == "private"
    assert classify_host("100.127.255.255") == "private"


def test_classify_public():
    assert classify_host("8.8.8.8") == "public"


def test_classify_unresolvable_hostname_defaults_public():
    assert classify_host("this-host-does-not-resolve.invalid") == "public"


def test_probe_internet_false_for_unroutable_host():
    # TEST-NET-1 (RFC 5737), guaranteed non-routable: no server should ever answer.
    assert probe_internet(["192.0.2.1:443"], timeout_s=0.5) is False


def test_probe_internet_true_when_any_host_reachable():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        # unreachable host listed first: probe must still find the reachable one
        assert probe_internet(["192.0.2.1:443", f"127.0.0.1:{port}"], timeout_s=0.5) is True
    finally:
        listener.close()


def test_count_app_public_connections_returns_nonnegative_int():
    count = count_app_public_connections()
    assert isinstance(count, int)
    assert count >= 0
