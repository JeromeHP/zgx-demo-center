from __future__ import annotations

import pytest

from app.settings import SettingsError, load_settings, resolve_path


def test_defaults_load(tmp_repo):
    settings = load_settings(repo_root=tmp_repo, env={})
    assert settings.llm.base_url == "http://127.0.0.1:8080/v1"
    assert settings.server.port == 8095
    assert settings.demo.cloud_pricing.input_per_million_usd is None


def test_local_override_merges(tmp_repo):
    (tmp_repo / "config" / "local.yaml").write_text(
        "llm:\n  base_url: \"http://127.0.0.1:8081/v1\"\ndemo:\n  device_label: \"Test Rig\"\n",
        encoding="utf-8",
    )
    settings = load_settings(repo_root=tmp_repo, env={})
    assert settings.llm.base_url == "http://127.0.0.1:8081/v1"
    assert settings.demo.device_label == "Test Rig"
    # untouched sibling keys from default.yaml must survive the merge
    assert settings.llm.model == "qwen3.6-35b-a3b"
    assert settings.server.port == 8095


def test_gm_config_env_points_to_alternate_override_file(tmp_repo):
    alt = tmp_repo / "config" / "alt.yaml"
    alt.write_text("server:\n  port: 9999\n", encoding="utf-8")
    settings = load_settings(repo_root=tmp_repo, env={"GM_CONFIG": "config/alt.yaml"})
    assert settings.server.port == 9999


def test_env_overrides_apply_with_correct_types(tmp_repo):
    settings = load_settings(
        repo_root=tmp_repo,
        env={"GM_LLM__BASE_URL": "http://127.0.0.1:9090/v1", "GM_SERVER__PORT": "9091"},
    )
    assert settings.llm.base_url == "http://127.0.0.1:9090/v1"
    assert settings.server.port == 9091
    assert isinstance(settings.server.port, int)


def test_invalid_value_fails_with_readable_error(tmp_repo):
    with pytest.raises(SettingsError) as exc_info:
        load_settings(repo_root=tmp_repo, env={"GM_SERVER__PORT": "not-a-number"})
    message = str(exc_info.value)
    assert "server" in message
    assert "port" in message


def test_missing_default_config_fails_readably(tmp_path):
    with pytest.raises(SettingsError) as exc_info:
        load_settings(repo_root=tmp_path, env={})
    assert "default.yaml" in str(exc_info.value)


def test_resolve_path_relative_to_repo_root(tmp_repo):
    resolved = resolve_path("data/vault/manifest.json", repo_root=tmp_repo)
    assert resolved == tmp_repo / "data" / "vault" / "manifest.json"
    assert resolved.is_absolute()


def test_resolve_path_leaves_absolute_paths_alone(tmp_repo):
    resolved = resolve_path("/etc/hosts", repo_root=tmp_repo)
    assert str(resolved) == "/etc/hosts"
