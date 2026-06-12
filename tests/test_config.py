from argparse import Namespace
import os
from pathlib import Path

import pytest

from core.config import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    default_max_tokens_for_model,
    load_app_config,
    resolve_model,
)


def _args(**overrides):
    values = {
        "prompt": None,
        "print": False,
        "auto_approve": False,
        "config": None,
        "provider": None,
        "api_key": None,
        "base_url": None,
        "model": None,
        "max_tokens": None,
        "effort": None,
        "buddy_model": None,
        "memory_dir": None,
        "no_auto_dream": False,
        "use_gpu": None,
        "online_evolution": None,
        "acp_agent": None,
        "acp_cwd": None,
        "acp_session": None,
        "acp_command": None,
        "acp_timeout": None,
        "acp_approve_all": None,
        "acp_model": None,
        "dream_interval": None,
        "dream_min_sessions": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_resolve_model_keeps_full_model_name():
    assert resolve_model("claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"


def test_default_max_tokens_follow_model_family():
    # Matches official getModelMaxOutputTokens() in context.ts
    assert default_max_tokens_for_model("claude-sonnet-4", provider="anthropic") == 32000
    assert default_max_tokens_for_model("claude-opus-4-6", provider="anthropic") == 64000
    assert default_max_tokens_for_model("claude-opus-4-1-20250805", provider="anthropic") == 32000
    assert default_max_tokens_for_model("claude-3-5-haiku-20241022", provider="anthropic") == 8192


def test_load_app_config_reads_anthropic_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("AUTODS_API_KEY", raising=False)
    monkeypatch.delenv("AUTODS_BASE_URL", raising=False)
    monkeypatch.delenv("AUTODS_MODEL", raising=False)
    monkeypatch.delenv("AUTODS_MAX_TOKENS", raising=False)

    config_path = tmp_path / "autods.toml"
    config_path.write_text(
        '[anthropic]\n'
        'api_key = "config-key"\n'
        'base_url = "https://example.test"\n'
        'model = "claude-3.7-sonnet"\n',
        encoding="utf-8",
    )

    config = load_app_config(_args(config=str(config_path)))

    assert config.provider == "anthropic"
    assert config.api_key == "config-key"
    assert config.base_url == "https://example.test"
    assert config.model == "claude-3-7-sonnet"
    assert config.max_tokens == 32000  # 3-7-sonnet: 32k per official context.ts


def test_load_app_config_cli_overrides_env_and_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "autods.toml"
    config_path.write_text(
        'api_key = "file-key"\n'
        'base_url = "https://file.test"\n'
        'model = "claude-3-5-haiku"\n'
        'max_tokens = 2048\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env.test")
    monkeypatch.setenv("AUTODS_MODEL", "claude-opus-4")
    monkeypatch.setenv("AUTODS_MAX_TOKENS", "1234")

    config = load_app_config(
        _args(
            config=str(config_path),
            api_key="cli-key",
            base_url="https://cli.test",
            model="claude-sonnet-4",
            max_tokens=999,
        )
    )

    assert config.api_key == "cli-key"
    assert config.base_url == "https://cli.test"
    assert config.model == "claude-sonnet-4"
    assert config.max_tokens == 999


def test_load_app_config_uses_defaults_when_nothing_is_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("AUTODS_API_KEY", raising=False)
    monkeypatch.delenv("AUTODS_BASE_URL", raising=False)
    monkeypatch.delenv("AUTODS_PROVIDER", raising=False)
    monkeypatch.delenv("AUTODS_MODEL", raising=False)
    monkeypatch.delenv("AUTODS_MAX_TOKENS", raising=False)

    config = load_app_config(_args())

    assert config.provider == DEFAULT_PROVIDER
    assert config.api_key == DEFAULT_API_KEY
    assert config.base_url == DEFAULT_BASE_URL
    assert config.model == DEFAULT_MODEL
    assert config.max_tokens == default_max_tokens_for_model(DEFAULT_MODEL)


def test_load_app_config_rejects_invalid_max_tokens(tmp_path: Path):
    config_path = tmp_path / "autods.toml"
    config_path.write_text('max_tokens = "abc"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid max_tokens"):
        load_app_config(_args(config=str(config_path)))


def test_load_app_config_reads_openai_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AUTODS_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    config_path = tmp_path / "autods.toml"
    config_path.write_text(
        'provider = "openai"\n'
        '[openai]\n'
        'api_key = "openai-key"\n'
        'base_url = "https://openai.test"\n'
        'model = "gpt-4.1-mini"\n'
        'max_tokens = 4096\n'
        'effort = "low"\n'
        'buddy_model = "gpt-4.1-nano"\n',
        encoding="utf-8",
    )

    config = load_app_config(_args(config=str(config_path)))

    assert config.provider == "openai"
    assert config.api_key == "openai-key"
    assert config.base_url == "https://openai.test"
    assert config.model == "gpt-4.1-mini"
    assert config.max_tokens == 4096
    assert config.effort == "low"
    assert config.buddy_model == "gpt-4.1-nano"


def test_openai_env_wins_when_provider_is_openai(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTODS_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.env")
    monkeypatch.setenv("AUTODS_MODEL", "gpt-4.1")
    monkeypatch.setenv("AUTODS_BUDDY_MODEL", "gpt-4.1-mini")

    config = load_app_config(_args())

    assert config.provider == "openai"
    assert config.api_key == "openai-env-key"
    assert config.base_url == "https://openai.env"
    assert config.model == "gpt-4.1"
    assert config.buddy_model == "gpt-4.1-mini"


def test_load_app_config_exports_kaggle_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KGAT_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

    config_path = tmp_path / "autods.toml"
    config_path.write_text(
        '[kaggle]\n'
        'username = "kaggle-user"\n'
        'key = "kaggle-key"\n'
        'kgat_api_token = "gateway-token"\n',
        encoding="utf-8",
    )

    load_app_config(_args(config=str(config_path)))

    assert os.environ["KAGGLE_USERNAME"] == "kaggle-user"
    assert os.environ["KAGGLE_KEY"] == "kaggle-key"
    assert os.environ["KGAT_API_TOKEN"] == "gateway-token"
    assert os.environ["KAGGLE_API_TOKEN"] == "gateway-token"


def test_kaggle_section_overrides_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "env-user")
    monkeypatch.setenv("KAGGLE_KEY", "env-key")
    monkeypatch.setenv("KGAT_API_TOKEN", "old-gateway")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "old-gateway")

    config_path = tmp_path / "autods.toml"
    config_path.write_text(
        '[kaggle]\n'
        'username = "file-user"\n'
        'key = "file-key"\n',
        encoding="utf-8",
    )

    load_app_config(_args(config=str(config_path)))

    assert os.environ["KAGGLE_USERNAME"] == "file-user"
    assert os.environ["KAGGLE_KEY"] == "file-key"
    assert "KGAT_API_TOKEN" not in os.environ
    assert "KAGGLE_API_TOKEN" not in os.environ


def test_kaggle_key_with_kgat_prefix_is_gateway_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KGAT_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

    config_path = tmp_path / "autods.toml"
    config_path.write_text(
        '[kaggle]\n'
        'username = "kaggle-user"\n'
        'key = "KGAT_test_gateway_token"\n',
        encoding="utf-8",
    )

    load_app_config(_args(config=str(config_path)))

    assert os.environ["KAGGLE_USERNAME"] == "kaggle-user"
    assert "KAGGLE_KEY" not in os.environ
    assert os.environ["KGAT_API_TOKEN"] == "KGAT_test_gateway_token"
    assert os.environ["KAGGLE_API_TOKEN"] == "KGAT_test_gateway_token"


def test_load_app_config_reads_use_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AUTODS_USE_GPU", raising=False)
    config_path = tmp_path / "autods.toml"
    config_path.write_text("use_gpu = true\n", encoding="utf-8")

    config = load_app_config(_args(config=str(config_path)))

    assert config.use_gpu is True


def test_use_gpu_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTODS_USE_GPU", "true")
    config_path = tmp_path / "autods.toml"
    config_path.write_text("use_gpu = false\n", encoding="utf-8")

    config = load_app_config(_args(config=str(config_path)))

    assert config.use_gpu is True


def test_load_app_config_reads_online_evolution_hyphen_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AUTODS_ONLINE_EVOLUTION", raising=False)
    config_path = tmp_path / "autods.toml"
    config_path.write_text("online-evolution = true\n", encoding="utf-8")

    config = load_app_config(_args(config=str(config_path)))

    assert config.online_evolution is True


def test_online_evolution_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTODS_ONLINE_EVOLUTION", "true")
    config_path = tmp_path / "autods.toml"
    config_path.write_text("online_evolution = false\n", encoding="utf-8")

    config = load_app_config(_args(config=str(config_path)))

    assert config.online_evolution is True


def test_load_app_config_reads_acp_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AUTODS_ACP_AGENT", raising=False)
    config_path = tmp_path / "autods.toml"
    config_path.write_text(
        'provider = "acp"\n'
        '[acp]\n'
        'agent = "claude"\n'
        'cwd = "."\n'
        'session = "kaggle-titanic"\n'
        'command = "npx acpx@latest"\n'
        'timeout = 2400\n'
        'approve_all = true\n'
        'model = "claude-sonnet-4"\n',
        encoding="utf-8",
    )

    config = load_app_config(_args(config=str(config_path)))

    assert config.provider == "acp"
    assert config.api_key is None
    assert config.base_url is None
    assert config.acp.agent == "claude"
    assert config.acp.cwd == "."
    assert config.acp.session == "kaggle-titanic"
    assert config.acp.command == "npx acpx@latest"
    assert config.acp.timeout == 2400
    assert config.acp.approve_all is True
    assert config.acp.model == "claude-sonnet-4"


def test_acp_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTODS_ACP_AGENT", "codex")
    monkeypatch.setenv("AUTODS_ACP_SESSION", "env-session")
    config_path = tmp_path / "autods.toml"
    config_path.write_text(
        'provider = "acp"\n'
        '[acp]\n'
        'agent = "claude"\n'
        'session = "file-session"\n',
        encoding="utf-8",
    )

    config = load_app_config(_args(config=str(config_path)))

    assert config.acp.agent == "codex"
    assert config.acp.session == "env-session"


def test_load_app_config_infers_acp_provider_from_section(tmp_path: Path):
    config_path = tmp_path / "autods.toml"
    config_path.write_text(
        '[acp]\n'
        'agent = "codex"\n',
        encoding="utf-8",
    )

    config = load_app_config(_args(config=str(config_path)))

    assert config.provider == "acp"
