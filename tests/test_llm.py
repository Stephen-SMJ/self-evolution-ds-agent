from core.llm import (
    ACPConfig,
    _ACPXStream,
    _extract_acp_stop_reason,
    _extract_acp_text_chunk,
    _to_acp_prompt,
    _to_openai_messages,
    _tool_schema_to_openai,
    build_acpx_command,
    default_companion_model,
    supports_reasoning_effort,
)


def test_to_openai_messages_maps_tool_roundtrip():
    messages = [
        {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "call_1",
                "name": "Echo",
                "input": {"message": "hello"},
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": "Echo: hello",
                "is_error": False,
            }],
        },
    ]

    converted = _to_openai_messages("system prompt", messages)

    assert converted[0] == {"role": "system", "content": "system prompt"}
    assert converted[1]["role"] == "assistant"
    assert converted[1]["tool_calls"][0]["function"]["name"] == "Echo"
    assert converted[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Echo: hello",
    }


def test_to_openai_messages_maps_image_input():
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "abcd",
                },
            },
            {"type": "text", "text": "describe this"},
        ],
    }]

    converted = _to_openai_messages(None, messages)

    assert converted[0]["role"] == "user"
    assert converted[0]["content"][0]["type"] == "image_url"
    assert converted[0]["content"][0]["image_url"]["url"] == "data:image/png;base64,abcd"
    assert converted[0]["content"][1] == {"type": "text", "text": "describe this"}


def test_tool_schema_to_openai_wraps_function_schema():
    tool = {
        "name": "Read",
        "description": "Read a file",
        "input_schema": {"type": "object", "properties": {}},
    }

    converted = _tool_schema_to_openai(tool)

    assert converted["type"] == "function"
    assert converted["function"]["name"] == "Read"
    assert converted["function"]["parameters"] == {"type": "object", "properties": {}}


def test_openai_reasoning_effort_support():
    assert supports_reasoning_effort("openai", "gpt-5") is True
    assert supports_reasoning_effort("openai", "gpt-4.1-mini") is False
    assert supports_reasoning_effort("anthropic", "claude-sonnet-4") is False


def test_default_companion_model_uses_main_model_for_openai():
    assert default_companion_model("openai", "gpt-4.1-mini") == "gpt-4.1-mini"


def test_acp_prompt_includes_system_and_messages():
    prompt = _to_acp_prompt(
        "system instructions",
        [{"role": "user", "content": "hello"}],
    )

    assert "Mantis System Context" in prompt
    assert "system instructions" in prompt
    assert "## user" in prompt
    assert "hello" in prompt


def test_acp_extracts_text_chunk_from_update_shapes():
    direct = {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "hello"},
        },
    }
    nested = {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "world"},
            },
        },
    }

    assert _extract_acp_text_chunk(direct) == "hello"
    assert _extract_acp_text_chunk(nested) == "world"


def test_acp_extracts_stop_reason():
    assert _extract_acp_stop_reason({"result": {"stopReason": "end_turn"}}) == "end_turn"


def test_acpx_stream_builds_command():
    stream = _ACPXStream(
        config=ACPConfig(
            agent="claude",
            cwd="/tmp/project",
            session="autods-test",
            command="npx acpx@latest",
            timeout=120,
            approve_all=True,
            model="claude-sonnet-4",
        ),
        model="acp-agent",
        messages=[{"role": "user", "content": "hello"}],
        system=None,
    )

    cmd = stream._build_command("/tmp/prompt.md")

    assert cmd[:2] == ["npx", "acpx@latest"]
    assert "--cwd" in cmd
    assert "/tmp/project" in cmd
    assert "--approve-all" in cmd
    assert "--model" in cmd
    assert "claude-sonnet-4" in cmd
    assert cmd[-6:] == ["claude", "prompt", "-s", "autods-test", "--file", "/tmp/prompt.md"]


def test_build_acpx_command_uses_acp_model_and_session():
    cmd = build_acpx_command(
        ACPConfig(
            agent="codex",
            cwd=".",
            session="mantis",
            command="acpx",
            timeout=1800,
            approve_all=False,
            model="gpt-5.5[medium]",
        ),
        "acp-agent",
        "/tmp/prompt.md",
    )

    assert cmd[:10] == [
        "acpx",
        "--cwd",
        ".",
        "--format",
        "json",
        "--json-strict",
        "--timeout",
        "1800",
        "--approve-reads",
        "--model",
    ]
    assert "gpt-5.5[medium]" in cmd
    assert cmd[-6:] == ["codex", "prompt", "-s", "mantis", "--file", "/tmp/prompt.md"]
