from unittest.mock import patch, MagicMock
import subprocess
from core.context import build_system_prompt, _get_git_section, _get_autods_md_section


def test_build_system_prompt_contains_base_instructions():
    prompt = build_system_prompt(cwd="/tmp")
    assert "data science" in prompt
    assert "tools" in prompt.lower()
    assert "Kaggle competition workflow" in prompt
    assert "self-evolving" in prompt


def test_build_system_prompt_contains_env_info():
    prompt = build_system_prompt(cwd="/tmp")
    assert "Primary working directory: /tmp" in prompt
    assert "Platform:" in prompt
    assert "Shell:" in prompt
    assert "Kaggle auth:" in prompt


def test_build_system_prompt_reports_official_kaggle_auth(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")
    monkeypatch.delenv("KGAT_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

    prompt = build_system_prompt(cwd="/tmp")

    assert "Kaggle auth: official API key configured" in prompt


def test_build_system_prompt_includes_gpu_info_when_enabled():
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "0, NVIDIA A100, 40960 MiB, 550.54\n"
    fake_result.stderr = ""

    with patch("core.context.subprocess.run", return_value=fake_result):
        prompt = build_system_prompt(cwd="/tmp", use_gpu=True)

    assert "GPU Environment" in prompt
    assert "GPU use configured: true" in prompt
    assert "NVIDIA A100" in prompt


def test_build_system_prompt_omits_gpu_info_by_default():
    prompt = build_system_prompt(cwd="/tmp")
    assert "GPU Environment" not in prompt


def test_build_system_prompt_marks_online_evolution_disabled_by_default():
    prompt = build_system_prompt(cwd="/tmp")
    assert "Online evolution configured: false" in prompt


def test_build_system_prompt_includes_online_evolution_when_enabled():
    prompt = build_system_prompt(cwd="/tmp", online_evolution=True)
    assert "Online evolution configured: true" in prompt
    assert "promotion_ledger.jsonl" in prompt
    assert "at least two distinct competitions" in prompt


def test_build_system_prompt_contains_working_directory():
    prompt = build_system_prompt(cwd="/some/test/dir")
    assert "/some/test/dir" in prompt


def test_build_system_prompt_includes_git_status_when_available():
    fake_result = MagicMock()
    fake_result.stdout = "main"

    with patch("core.context.subprocess.run", return_value=fake_result):
        prompt = build_system_prompt(cwd="/tmp")
    assert "Git Status" in prompt
    assert "main" in prompt


def test_build_system_prompt_includes_mantis_md(tmp_path):
    mantis_md = tmp_path / "MANTIS.md"
    mantis_md.write_text("# Test Project\nSome instructions here.")

    prompt = build_system_prompt(cwd=str(tmp_path))
    assert "MANTIS.md" in prompt
    assert "Test Project" in prompt


def test_build_system_prompt_without_mantis_md(tmp_path):
    prompt = build_system_prompt(cwd=str(tmp_path))
    # Should not have the MANTIS.md section header (beyond the base prompt)
    assert "# Test Project" not in prompt


def test_get_git_section_returns_branch_and_log(tmp_path):
    def fake_run(cmd, **kwargs):
        result = MagicMock()
        if "branch" in cmd:
            result.stdout = "feature-branch"
        elif "status" in cmd:
            result.stdout = " M file.py"
        elif "log" in cmd:
            result.stdout = "abc1234 some commit"
        else:
            result.stdout = ""
        return result

    with patch("core.context.subprocess.run", side_effect=fake_run):
        status = _get_git_section(str(tmp_path))

    assert "feature-branch" in status
    assert "M file.py" in status
    assert "abc1234" in status


def test_get_git_section_returns_empty_on_non_git_dir():
    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.stdout = ""
        return result

    with patch("core.context.subprocess.run", side_effect=fake_run):
        status = _get_git_section("/tmp/not-a-git-repo")
    assert status == ""


def test_get_git_section_returns_empty_on_exception():
    with patch("core.context.subprocess.run", side_effect=OSError("fail")):
        status = _get_git_section("/tmp")
    assert status == ""


def test_get_autods_md_section_reads_mantis_file(tmp_path):
    mantis_md = tmp_path / "MANTIS.md"
    mantis_md.write_text("hello world")

    result = _get_autods_md_section(str(tmp_path))
    assert "hello world" in result
    assert "MANTIS.md" in result


def test_get_autods_md_section_reads_legacy_file(tmp_path):
    autods_md = tmp_path / "AUTODS.md"
    autods_md.write_text("legacy hello")

    result = _get_autods_md_section(str(tmp_path))
    assert "legacy hello" in result
    assert "AUTODS.md (legacy)" in result


def test_get_autods_md_section_returns_empty_when_missing(tmp_path):
    result = _get_autods_md_section(str(tmp_path))
    assert result == ""


def test_get_autods_md_section_truncates_large_file(tmp_path):
    mantis_md = tmp_path / "MANTIS.md"
    mantis_md.write_text("x" * 20_000)

    result = _get_autods_md_section(str(tmp_path))
    # Section includes header, so content is truncated to fit within 10k chars
    assert len(result) <= 10_100  # Allow some margin for the header
