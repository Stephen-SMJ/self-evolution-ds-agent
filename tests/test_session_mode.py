import json

from core.session import SessionStore, _sanitize_cwd
from commands import find_session, recent_conversation_preview


def test_session_store_persists_mode(tmp_path, monkeypatch):
    monkeypatch.setattr("core.session._SESSIONS_ROOT", tmp_path)

    store = SessionStore(
        cwd="/tmp/project",
        model="test-model",
        session_id="session-1",
        mode="coordinator",
    )
    store.append_message({"role": "user", "content": "hello"})

    meta_path = tmp_path / _sanitize_cwd("/tmp/project") / "session-1.meta.json"
    data = json.loads(meta_path.read_text())

    assert data["mode"] == "coordinator"

    sessions = SessionStore.list_sessions("/tmp/project")
    assert len(sessions) == 1
    assert sessions[0].mode == "coordinator"


def test_session_store_persists_kaggle_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("core.session._SESSIONS_ROOT", tmp_path)

    store = SessionStore(
        cwd="/tmp/project",
        model="test-model",
        session_id="session-2",
    )
    store.append_message({
        "role": "user",
        "content": "/kaggle https://www.kaggle.com/competitions/titanic/leaderboard",
    })

    meta_path = tmp_path / _sanitize_cwd("/tmp/project") / "session-2.meta.json"
    data = json.loads(meta_path.read_text())

    assert data["title"] == "Kaggle: titanic"
    assert data["workspace"] == "competitions/titanic"

    sessions = SessionStore.list_sessions("/tmp/project")
    assert sessions[0].workspace == "competitions/titanic"


def test_session_store_infers_workspace_for_old_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("core.session._SESSIONS_ROOT", tmp_path)
    session_dir = tmp_path / _sanitize_cwd("/tmp/project")
    session_dir.mkdir(parents=True)

    (session_dir / "session-3.meta.json").write_text(json.dumps({
        "session_id": "session-3",
        "title": "# Kaggle Competition Workflow",
        "cwd": "/tmp/project",
        "model": "test-model",
        "created_at": "2026-06-09T00:00:00+00:00",
        "updated_at": "2026-06-09T00:00:01+00:00",
        "message_count": 1,
    }))
    (session_dir / "session-3.jsonl").write_text(json.dumps({
        "role": "user",
        "content": "# Kaggle Competition Workflow\n\n## User Competition Input\n\nhttps://www.kaggle.com/competitions/house-prices-advanced-regression-techniques",
    }) + "\n")

    sessions = SessionStore.list_sessions("/tmp/project")
    assert sessions[0].workspace == "competitions/house-prices-advanced-regression-techniques"


def test_session_store_infers_workspace_from_kaggle_skill_slug(tmp_path, monkeypatch):
    monkeypatch.setattr("core.session._SESSIONS_ROOT", tmp_path)

    store = SessionStore(
        cwd="/tmp/project",
        model="test-model",
        session_id="session-5",
    )
    store.append_message({
        "role": "user",
        "content": "# Kaggle Competition Workflow\n\n## User Competition Input\n\nnlp-getting-started",
    })

    sessions = SessionStore.list_sessions("/tmp/project")
    assert sessions[0].title == "Kaggle: nlp-getting-started"
    assert sessions[0].workspace == "competitions/nlp-getting-started"


def test_find_session_matches_workspace_slug(tmp_path, monkeypatch):
    monkeypatch.setattr("core.session._SESSIONS_ROOT", tmp_path)

    store = SessionStore(
        cwd="/tmp/project",
        model="test-model",
        session_id="session-4",
    )
    store.append_message({
        "role": "user",
        "content": "/kaggle https://www.kaggle.com/competitions/spaceship-titanic",
    })

    sessions = SessionStore.list_sessions("/tmp/project")

    assert find_session(sessions, "spaceship-titanic").session_id == "session-4"
    assert find_session(sessions, "competitions/spaceship-titanic").session_id == "session-4"


def test_recent_conversation_preview_shows_last_five_turns_only():
    messages = []
    for i in range(7):
        messages.append({"role": "user", "content": f"user {i}"})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"assistant {i}"}]})

    rows = recent_conversation_preview(messages, turns=5)

    assert len(rows) == 10
    assert rows[0] == ("User", "user 2")
    assert rows[-1] == ("Mantis", "assistant 6")


def test_recent_conversation_preview_skips_tool_results_and_truncates():
    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "content": "x" * 1000}],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "run"}}],
        },
        {
            "role": "user",
            "content": "y" * 300,
        },
    ]

    rows = recent_conversation_preview(messages, turns=5)

    assert rows[0] == ("Mantis", "[used tools: Bash]")
    assert rows[1][0] == "User"
    assert len(rows[1][1]) == 220
    assert rows[1][1].endswith("…")
