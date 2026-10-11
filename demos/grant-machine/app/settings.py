"""Config loading: YAML defaults + local overrides + GM_ env overrides.

Load order (Spec A):
  1. config/default.yaml (required)
  2. deep-merge config/local.yaml if present, or the file named by GM_CONFIG
     if that env var is set (relative to REPO_ROOT)
  3. apply GM_<SECTION>__<KEY> env overrides (values parsed as YAML scalars)
  4. validate the merged dict with pydantic; fail fast with a readable error

All relative paths in config resolve against REPO_ROOT, not the process cwd.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a possibly-relative config path against repo_root."""
    p = Path(value)
    return p if p.is_absolute() else (repo_root / p)


class SettingsError(Exception):
    """Raised when config fails to load or validate, with a readable message."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMProfile(StrictModel):
    enable_thinking: bool
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    max_tokens: int


class LLMSettings(StrictModel):
    base_url: str
    api_key: str
    model: str
    request_timeout_s: float
    reasoning_mode: Literal["auto", "field", "inline_tags"]
    # How to ask the server for (or suppress) thinking mode. llama-server reads
    # `chat_template_kwargs.enable_thinking`; Ollama ignores that entirely and
    # only honours `reasoning_effort` ("none" to disable). Defaults to the
    # llama.cpp form so the original endpoint behaviour is unchanged.
    thinking_control: Literal["chat_template", "reasoning_effort"] = "chat_template"
    context_tokens: int
    profiles: dict[str, LLMProfile]


class ServerSettings(StrictModel):
    host: str
    port: int


class ScopeSettings(StrictModel):
    sections: list[str]
    required_tools: list[str]


class AgentSettings(StrictModel):
    max_steps: int
    max_revisions_per_section: int
    review_pass_threshold: float
    max_tool_result_chars: int
    scopes: dict[str, ScopeSettings]


class BudgetSettings(StrictModel):
    track: str
    equipment_threshold_usd: float


class UploadSettings(StrictModel):
    max_bytes: int
    max_pdf_pages: int
    allowed_extensions: list[str]


class HumanBaselineMinutes(StrictModel):
    compliance_matrix: float | None = None
    budget_review: float | None = None
    section_draft: float | None = None


class CloudPricing(StrictModel):
    input_per_million_usd: float | None = None
    output_per_million_usd: float | None = None
    label: str | None = None


class ConnectivityProbe(StrictModel):
    hosts: list[str]
    interval_s: float
    timeout_s: float


class EgressSettings(StrictModel):
    """The outbound boundary (app/egress.py).

    Defaults are deliberately conservative: an operator who never touches this block
    gets the strict boundary, and loosening it is an explicit, visible edit.
    """

    enabled: bool = True
    vocabulary_dir: str = "data/vocabulary"
    max_requests_per_run: int = 40
    timeout_s: float = 30.0
    max_response_bytes: int = 5_000_000
    max_term_words: int = 6
    ngram_len: int = 3
    # Self-hosted SearXNG for general web search. Empty = no provider, and search_web
    # reports itself unavailable rather than failing: opportunity search does not
    # depend on it. Deliberately no commercial default -- an API key on a demo box,
    # and queries routed through a third party, would both undercut the point.
    searxng_url: str = ""
    max_page_chars: int = 40000


class DemoSettings(StrictModel):
    device_label: str
    human_baseline_minutes: HumanBaselineMinutes
    cloud_pricing: CloudPricing
    connectivity_probe: ConnectivityProbe


class ModelServerSettings(BaseModel):
    """Read only by scripts/*. The app loads but never uses this block."""

    model_config = ConfigDict(extra="allow")


class Settings(StrictModel):
    llm: LLMSettings
    server: ServerSettings
    agent: AgentSettings
    budget: BudgetSettings
    uploads: UploadSettings
    demo: DemoSettings
    egress: EgressSettings = EgressSettings()
    model_server: ModelServerSettings | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


_NON_OVERRIDE_ENV_KEYS = {"GM_CONFIG"}  # consumed earlier in load_settings, not a settings path


def _apply_env_overrides(data: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    merged = copy.deepcopy(data)
    for raw_key, raw_value in env.items():
        if not raw_key.startswith("GM_") or raw_key in _NON_OVERRIDE_ENV_KEYS:
            continue
        path = [segment.lower() for segment in raw_key[len("GM_") :].split("__") if segment]
        if not path:
            continue
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise SettingsError(f"env override {raw_key} is not valid YAML: {exc}") from exc

        cursor = merged
        for segment in path[:-1]:
            existing = cursor.get(segment)
            if not isinstance(existing, dict):
                existing = {}
                cursor[segment] = existing
            cursor = existing
        cursor[path[-1]] = value
    return merged


def load_settings(
    repo_root: Path = REPO_ROOT,
    config_override: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    env = os.environ if env is None else env

    default_path = repo_root / "config" / "default.yaml"
    if not default_path.is_file():
        raise SettingsError(f"missing required config file: {default_path}")
    with default_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if config_override is not None:
        local_path: Path | None = resolve_path(config_override, repo_root)
    elif "GM_CONFIG" in env:
        local_path = resolve_path(env["GM_CONFIG"], repo_root)
    else:
        candidate = repo_root / "config" / "local.yaml"
        local_path = candidate if candidate.is_file() else None

    if local_path is not None:
        if not local_path.is_file():
            raise SettingsError(f"configured local config file not found: {local_path}")
        with local_path.open("r", encoding="utf-8") as f:
            local_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, local_data)

    data = _apply_env_overrides(data, env)

    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        raise SettingsError(f"invalid configuration:\n{exc}") from exc


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
