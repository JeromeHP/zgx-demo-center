from __future__ import annotations

import copy
import threading
import time
from typing import Any

import pytest
import uvicorn
import yaml

from app.settings import REPO_ROOT, Settings
from tests.fake_llm_server import create_app

DEFAULT_CONFIG_TEXT = (REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")


def deep_merge_into(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge_into(base[key], value)
        else:
            base[key] = value


@pytest.fixture
def default_config_dict() -> dict[str, Any]:
    return copy.deepcopy(yaml.safe_load(DEFAULT_CONFIG_TEXT))


@pytest.fixture
def make_settings(default_config_dict):
    """make_settings(overrides={"llm": {"base_url": "..."}}) -> Settings"""

    def _make(overrides: dict[str, Any] | None = None) -> Settings:
        data = copy.deepcopy(default_config_dict)
        if overrides:
            deep_merge_into(data, overrides)
        return Settings.model_validate(data)

    return _make


@pytest.fixture
def tmp_repo(tmp_path):
    """A throwaway repo root with a real config/default.yaml, for settings-loading tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
    return tmp_path


@pytest.fixture
def fake_llm_app():
    return create_app()


class _UvicornThreadServer:
    """Runs a FastAPI app on a real loopback socket in a background thread.

    LLMClient uses the synchronous openai.OpenAI client, and httpx.ASGITransport
    only implements the async transport interface, so an in-process ASGI
    transport can't be used here -- a real socket is the simplest fix and
    exercises the actual HTTP/SSE code path anyway.
    """

    def __init__(self, app):
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        self.server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        while not self.server.started:
            time.sleep(0.01)

    @property
    def base_url(self) -> str:
        sock = self.server.servers[0].sockets[0]
        host, port = sock.getsockname()[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=5)


@pytest.fixture
def fake_llm_server(fake_llm_app):
    server = _UvicornThreadServer(fake_llm_app)
    server.start()
    yield server
    server.stop()
