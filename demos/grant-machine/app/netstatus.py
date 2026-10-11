"""Internet reachability + endpoint classification (Spec I)."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import psutil

from app.settings import Settings

_CGNAT_START = ipaddress.ip_address("100.64.0.0")
_CGNAT_END = ipaddress.ip_address("100.127.255.255")


def classify_host(host: str) -> str:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(host))
        except (OSError, ValueError):
            return "public"
    if ip.is_loopback:
        return "loopback"
    if ip.is_private:
        return "private"
    if _CGNAT_START <= ip <= _CGNAT_END:
        return "private"
    return "public"


def probe_internet(hosts: list[str], timeout_s: float) -> bool:
    for hostport in hosts:
        host, _, port = hostport.partition(":")
        try:
            with socket.create_connection((host, int(port) if port else 443), timeout=timeout_s):
                return True
        except OSError:
            continue
    return False


def _is_local_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or (_CGNAT_START <= ip <= _CGNAT_END)


def count_app_public_connections() -> int:
    """ESTABLISHED connections from this process whose remote IP is not
    loopback/private/CGNAT. Says nothing about the model server process."""
    try:
        conns = psutil.Process().net_connections(kind="inet")
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        try:
            conns = psutil.Process().connections(kind="inet")  # psutil < 6.0 name
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return 0
    count = 0
    for c in conns:
        if c.status != psutil.CONN_ESTABLISHED or not c.raddr:
            continue
        if not _is_local_ip(c.raddr.ip):
            count += 1
    return count


def get_status(settings: Settings) -> dict[str, object]:
    probe_cfg = settings.demo.connectivity_probe
    online = probe_internet(probe_cfg.hosts, probe_cfg.timeout_s)
    llm_host = urlparse(settings.llm.base_url).hostname or ""
    return {
        "internet": "online" if online else "offline",
        "llm_endpoint_class": classify_host(llm_host),
        "app_public_connections": count_app_public_connections(),
    }
