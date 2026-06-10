import json

from features.online_evolution import normalize_competition_ref, refresh_online_evolution


def test_normalize_competition_ref_accepts_url_folder_and_slug():
    assert normalize_competition_ref("titanic") == "titanic"
    assert normalize_competition_ref("competitions/titanic") == "titanic"
    assert normalize_competition_ref("https://www.kaggle.com/competitions/titanic/leaderboard") == "titanic"


def test_refresh_online_evolution_creates_artifacts_and_ledger(tmp_path):
    evolution_dir = tmp_path / "competitions" / "titanic" / "evolution"
    evolution_dir.mkdir(parents=True)
    run = {
        "run_id": "v03",
        "competition": "titanic",
        "domain": "tabular",
        "metric": "accuracy",
        "changes": ["CatBoost depth 8"],
        "tactic_keys": ["tabular/catboost-depth-lr-tuning"],
        "validation": {"oof": 0.8169},
        "leaderboard": {"public": 0.80664},
        "delta_vs_previous": {"oof": 0.0013, "public": 0.00211},
        "status": "promoted",
        "lesson": "CatBoost depth/lr improved both OOF and public LB.",
        "next": "Ablate noisy features.",
    }
    (evolution_dir / "runs.jsonl").write_text(json.dumps(run) + "\n", encoding="utf-8")

    result = refresh_online_evolution(tmp_path, "titanic")

    assert result.slug == "titanic"
    assert result.run_count == 1
    assert result.ledger_records == 1
    assert "CatBoost depth 8" in (evolution_dir / "score_trends.md").read_text()
    assert "CatBoost depth/lr improved" in (evolution_dir / "lessons.md").read_text()
    assert "Ablate noisy features" in (evolution_dir / "hypotheses.json").read_text()

    ledger = tmp_path / ".autods" / "online_evolution" / "promotion_ledger.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert records[0]["tactic_key"] == "tabular/catboost-depth-lr-tuning"
