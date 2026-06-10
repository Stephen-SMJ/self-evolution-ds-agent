"""Online evolution artifacts for Kaggle competition runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9-]*$")
_URL_RE = re.compile(r"kaggle\.com/competitions/([a-zA-Z0-9][a-zA-Z0-9-]*)")


@dataclass
class EvolutionRefreshResult:
    slug: str
    workspace: Path
    run_count: int
    ledger_records: int


def normalize_competition_ref(ref: str) -> str:
    raw = ref.strip().strip("/")
    if not raw:
        raise ValueError("competition slug or folder is required")
    url_match = _URL_RE.search(raw)
    if url_match:
        return url_match.group(1)
    if raw.startswith("competitions/"):
        raw = raw.split("/", 2)[1]
    if not _SLUG_RE.fullmatch(raw):
        raise ValueError(f"invalid competition slug: {ref!r}")
    return raw


def refresh_online_evolution(root: Path, competition_ref: str) -> EvolutionRefreshResult:
    slug = normalize_competition_ref(competition_ref)
    workspace = root / "competitions" / slug
    evolution_dir = workspace / "evolution"
    online_dir = root / ".autods" / "online_evolution"
    evolution_dir.mkdir(parents=True, exist_ok=True)
    online_dir.mkdir(parents=True, exist_ok=True)

    runs_path = evolution_dir / "runs.jsonl"
    if not runs_path.exists():
        runs_path.write_text("", encoding="utf-8")

    runs = _read_runs(runs_path)
    _write_score_trends(evolution_dir / "score_trends.md", slug, runs)
    _write_lessons(evolution_dir / "lessons.md", slug, runs)
    _write_hypotheses(evolution_dir / "hypotheses.json", slug, runs)
    ledger_records = _append_promotion_candidates(
        online_dir / "promotion_ledger.jsonl",
        slug,
        runs,
    )
    proposals = online_dir / "skill_patch_proposals.md"
    if not proposals.exists():
        proposals.write_text(
            "# Skill Patch Proposals\n\n"
            "Global skill patches require cross-competition evidence. Add proposals here only after checking `promotion_ledger.jsonl`.\n",
            encoding="utf-8",
        )

    return EvolutionRefreshResult(
        slug=slug,
        workspace=workspace,
        run_count=len(runs),
        ledger_records=ledger_records,
    )


def _read_runs(path: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            runs.append(obj)
    return runs


def _score_value(run: dict[str, Any], group: str, key: str) -> Any:
    value = run.get(group)
    if isinstance(value, dict):
        return value.get(key)
    return None


def _write_score_trends(path: Path, slug: str, runs: list[dict[str, Any]]) -> None:
    lines = [
        f"# Score Trends: {slug}",
        "",
        "| run | changes | OOF | public LB | rank | status | decision |",
        "|-----|---------|-----|-----------|------|--------|----------|",
    ]
    for run in runs:
        changes = ", ".join(str(x) for x in run.get("changes", [])) or "-"
        lines.append(
            "| {run_id} | {changes} | {oof} | {public} | {rank} | {status} | {lesson} |".format(
                run_id=run.get("run_id", "-"),
                changes=changes.replace("|", "/")[:80],
                oof=_score_value(run, "validation", "oof") or "-",
                public=_score_value(run, "leaderboard", "public") or "-",
                rank=_score_value(run, "leaderboard", "rank") or "-",
                status=run.get("status", "-"),
                lesson=str(run.get("lesson", "-")).replace("|", "/")[:100],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_lessons(path: Path, slug: str, runs: list[dict[str, Any]]) -> None:
    groups = {
        "promoted": [],
        "rejected": [],
        "inconclusive": [],
        "validation-risk": [],
    }
    for run in runs:
        status = str(run.get("status", "inconclusive")).lower()
        bucket = status if status in groups else "inconclusive"
        lesson = str(run.get("lesson", "")).strip()
        if lesson:
            groups[bucket].append((run.get("run_id", "-"), lesson))

    lines = [f"# Online Lessons: {slug}", ""]
    for bucket, items in groups.items():
        lines.extend([f"## {bucket.title()}", ""])
        if not items:
            lines.append("- None yet.")
        else:
            lines.extend(f"- `{run_id}`: {lesson}" for run_id, lesson in items)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_hypotheses(path: Path, slug: str, runs: list[dict[str, Any]]) -> None:
    hypotheses = {
        "competition": slug,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "active": [],
        "stop_rules": [
            "Stop or change strategy after three meaningful experiments without CV or LB improvement.",
            "Repair validation before adding complexity when CV and LB disagree strongly.",
        ],
    }
    if runs:
        last = runs[-1]
        next_step = str(last.get("next", "")).strip()
        if next_step:
            hypotheses["active"].append({
                "source_run": last.get("run_id"),
                "hypothesis": next_step,
                "evidence": last.get("lesson", ""),
                "next_test": next_step,
            })
    path.write_text(json.dumps(hypotheses, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_promotion_candidates(path: Path, slug: str, runs: list[dict[str, Any]]) -> int:
    existing = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing.add((obj.get("competition"), obj.get("run_id"), obj.get("tactic_key")))

    appended = 0
    with path.open("a", encoding="utf-8") as fh:
        for run in runs:
            if str(run.get("status", "")).lower() != "promoted":
                continue
            for tactic_key in run.get("tactic_keys", []) or []:
                marker = (slug, run.get("run_id"), tactic_key)
                if marker in existing:
                    continue
                record = {
                    "tactic_key": tactic_key,
                    "domain": run.get("domain"),
                    "competition": slug,
                    "run_id": run.get("run_id"),
                    "metric": run.get("metric"),
                    "evidence": run.get("delta_vs_previous", {}),
                    "verdict": "domain_candidate",
                    "risk": "medium",
                    "notes": run.get("lesson", ""),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                appended += 1
    return appended
