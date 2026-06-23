# Online Evolution Protocol

Use this protocol only when the system prompt says `Online evolution configured:
true`.

The goal is to convert online competition attempts into reusable evidence. Do
not treat every score increase as a skill. A skill patch needs repeated,
transferable evidence.

## Per-Competition Files

For each competition, maintain:

```text
competitions/<slug>/evolution/
  runs.jsonl
  score_trends.md
  lessons.md
  hypotheses.json
```

`runs.jsonl` is the source of truth. Append one JSON object per meaningful
experiment:

```json
{
  "run_id": "v03",
  "competition": "<slug>",
  "domain": "tabular",
  "metric": "accuracy",
  "direction": "higher",
  "changes": ["CatBoost depth 8", "lr 0.05"],
  "tactic_keys": ["tabular/catboost-depth-lr-tuning"],
  "feature_changes": ["added ticket frequency"],
  "model_changes": ["CatBoost"],
  "validation": {"oof": 0.8169, "folds": 5},
  "leaderboard": {"public": 0.80664, "rank": null, "percentile": null},
  "delta_vs_previous": {"oof": 0.0013, "public": 0.00211},
  "runtime_seconds": null,
  "status": "promoted",
  "lesson": "CatBoost depth/lr improved both OOF and public LB.",
  "next": "Ablate noisy engineered features."
}
```

Use stable `tactic_keys` so evidence can aggregate across competitions:

```text
<domain>/<short-action>
tabular/categorical-boosting-baseline
tabular/group-count-frequency-features
tabular/oof-threshold-tuning
timeseries/grouped-lag-validation
cv/stratified-kfold-oof-ensemble
nlp/clean-text-tfidf-linear-baseline
```

## Reflection Triggers

Run reflection after:

- every submitted Kaggle result
- every three local experiments without submission
- any new best local validation or leaderboard score
- two consecutive regressions
- major CV/LB disagreement

Reflection must update:

- `score_trends.md`: table of run, change, OOF, LB, rank/percentile, delta, decision.
- `lessons.md`: promoted, rejected, inconclusive, and validation-risk lessons.
- `hypotheses.json`: active hypotheses with evidence, next test, and stop rule.

## Global Evidence Files

Maintain:

```text
.mantis/online_evolution/
  promotion_ledger.jsonl
  skill_patch_proposals.md
```

Each ledger record should contain:

```json
{
  "tactic_key": "tabular/group-count-frequency-features",
  "domain": "tabular",
  "competition": "porto-seguro-safe-driver-prediction",
  "metric": "normalized-gini",
  "evidence": {
    "validation_delta": 0.0021,
    "leaderboard_delta": 0.0014,
    "rank_delta": null
  },
  "verdict": "domain_candidate",
  "risk": "low",
  "notes": "Improved both OOF and public LB; no leakage indicators."
}
```

## Promotion Gates

Local lesson:

- One competition is enough.
- Write to `competitions/<slug>/evolution/lessons.md` and `MANTIS.md`.

Domain candidate:

- One strong competition result is enough if validation and LB agree.
- Write to `.mantis/online_evolution/promotion_ledger.jsonl`.
- Do not patch global skills yet.

Global skill patch proposal:

- Requires the same `tactic_key` to have positive evidence in at least two
  distinct competitions in the same domain, or one online competition plus
  matching offline V4 evidence and no contradictory online evidence.
- Write a proposal to `.mantis/online_evolution/skill_patch_proposals.md`.

Global skill patch:

- Patch `.mantis/skills/mantis-kaggle-distilled/` only when the proposal names:
  supporting competitions, score deltas, validation behavior, known failures,
  and when not to use the tactic.
- The patch should change the smallest relevant reference file, usually
  `references/domain-playbooks.md` or a dedicated online evolution reference.

## Rejection Rules

Do not promote:

- a tactic that improves public LB while validation strongly regresses
- results from known leaderboard probing, hardcoded labels, or rule-violating leakage
- single-competition tricks that depend on IDs, test-set quirks, or hand-coded
  public leaderboard reverse engineering
- expensive methods whose gain is smaller than noise and whose cost blocks
  better experiments

When evidence is mixed, mark the tactic as `inconclusive` and design the next
experiment to isolate the cause.
