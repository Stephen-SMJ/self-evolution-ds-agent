# V3 Semantic Distillation

V3 is the intelligent distillation layer. The current usable V3 is a Codex semantic pass: Codex directly compares five rank buckets for each competition and infers the core improvements from weaker solutions to stronger/gold-level solutions.

Note: V4 deep-read distillation now supersedes V3 for planning. Use V3 as a supporting reference when V4 lacks detail or when you need the older competition-level semantic trace.

Primary report:

- `offline/distillation/v3/codex_semantic_distillation_v3.md`

This should be preferred over V2 rule-based traces when planning competition strategy, but V4 should be preferred over V3 when available.

Script:

- `offline/tools/distill_v3_llm_semantic.py`

Expected outputs:

- `offline/distillation/v3/competition_traces/<domain>/<competition>.json`
- `offline/distillation/v3/domain_recipes/<domain>.json`
- `offline/distillation/v3/semantic_distillation_v3.md`
- `offline/distillation/v3/summary.json`

## Use Priority

When V4 deep-read distillation exists:

1. Read `references/v4-deep-read-distillation.md`.
2. If needed, read `offline/distillation/v4/deep_distillation_v4.md`.
3. Use V3 only as supporting context or fallback.

When Codex V3 semantic report exists and V4 is not available:

1. Read `codex_semantic_distillation_v3.md`.
2. Use its domain recipe and competition-level semantic traces to plan the baseline and upgrades.
3. Fall back to V2 JSON traces for evidence lookup and file paths.

When external-LLM V3 outputs exist:

1. Read the target domain recipe from `distillation/v3/domain_recipes/`.
2. Read 1-3 similar competition traces from `distillation/v3/competition_traces/`.
3. Use the semantic `core_evolution` steps as the upgrade plan.
4. Fall back to V2 traces only for evidence inspection or missing V3 coverage.

When V3 outputs do not exist:

1. Use V2 rule-based traces.
2. Inspect raw notebooks for ambiguous or surprising signals.
3. Do not claim semantic distillation has been completed.

## Current Status

Codex V3 semantic pass is available at:

- `offline/distillation/v3/codex_semantic_distillation_v3.md`

External-LLM V3 infrastructure is implemented, but the first smoke test returned `401 invalid_key` from the configured OpenAI-style endpoint. Full API-driven V3 semantic distillation requires a valid `MANTIS_API_KEY`/`OPENAI_API_KEY` and base URL.

## Run Command

```bash
cd <repo-root>/offline
python3 tools/distill_v3_llm_semantic.py --root .
```

## Expected Semantic Output

Each competition trace should include:

- bucket summaries for `70pct`, `40pct`, `20pct`, `10pct`, `1st`
- core upgrades between adjacent buckets
- why each upgrade likely improves score
- implementation hints
- gold-level principles
- validation guardrails
- failure modes and avoid-list
