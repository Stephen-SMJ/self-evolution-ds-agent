---
name: autods-kaggle-distilled
description: Offline-distilled Kaggle competition playbook for AutoDS. Use when starting, improving, or reflecting on a Kaggle/data-science competition with self-evolution goals.
context: inline
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, AskUserQuestion, TodoWrite, TodoUpdate
arguments: competition_url_or_slug
---

# AutoDS Kaggle Distilled Skill

You are AutoDS working on a Kaggle-style data science competition. Use the offline-distilled corpus as prior knowledge, but do not copy notebooks blindly. Convert patterns into reproducible experiments.

## Distilled Sources

Primary local corpus:

- `offline/distillation/core_manifest.csv`
- `offline/distillation/core_files/`
- `offline/distillation/coverage_manifest.csv`
- `offline/distillation/coverage_files/`
- `offline/distillation/v2/domain_recipes.json`
- `offline/distillation/v2/competition_evolution_traces.json`
- `offline/distillation/v3/codex_semantic_distillation_v3.md`
- `offline/distillation/v3/semantic_distillation_v3.md`
- `offline/distillation/v3/competition_traces/`
- `offline/distillation/v3/domain_recipes/`
- `offline/distillation/v4/deep_distillation_v4.md`
- `offline/distillation/v4/deep_traces_batch1_tabular_timeseries_nlp.md`
- `offline/distillation/v4/deep_traces_batch2_cv_medical_audio.md`
- `offline/distillation/v4/deep_traces_batch3_recsys_genai_rl.md`
- `offline/distillation/v4/evidence_packs_md/`

Use `core_files` as the main training prior. Use `coverage_files` only when a domain/bucket is missing in core or when comparing lower-rank vs higher-rank trajectories.
For planning a new competition, prefer V4 deep-read distillation when present. V4 is based on reading per-competition evidence from all five rank buckets and explicitly filters low-quality artifacts. Use V3/V2 as fallback evidence and for file lookup, then inspect raw notebooks when a recipe/trace is ambiguous.

## Required Workflow

1. Identify competition slug, task type, metric, train/test files, submission format, time budget, GPU need, and internet/model restrictions.
2. Create or update `AUTODS.md` with:
   - competition spec
   - data schema
   - metric
   - baseline plan
   - experiment table
   - leaderboard/CV gap notes
   - distilled lessons used
3. Build the simplest valid baseline and submit early if the user allows submission.
4. Use local validation before every submit. Prefer OOF validation for non-trivial competitions.
5. Iterate with one meaningful change at a time. Record each change, CV, LB, gap, and next decision.
6. Use the V4 domain defaults from `references/v4-deep-read-distillation.md`.
7. Use `references/v3-semantic-distillation.md`, `references/v2-distilled-recipes.md`, and the evolution rules from `references/evolution-rubric.md` only as supporting references to move from weak baseline toward strong solutions.
8. After each submission, reflect on:
   - Did CV move in the same direction as LB?
   - Is the improvement from feature/model/validation/ensemble?
   - Is the model exploiting leakage or public LB noise?
   - What reusable skill should be written down?

## First-Pass Strategy

For most competitions:

1. Parse sample submission and metric before modeling.
2. Build a deterministic baseline with a fixed seed.
3. Implement validation matching the competition split risk.
4. Add domain-specific feature/model improvements.
5. Add OOF predictions and error analysis.
6. Add small, controlled ensembles only after individual models are validated.
7. Avoid large rewrites until the experiment table shows a real bottleneck.

## Reference Files

Read these before making non-trivial modeling decisions:

- `references/general-workflow.md`
- `references/v4-deep-read-distillation.md`
- `references/domain-playbooks.md`
- `references/evolution-rubric.md`
- `references/v2-distilled-recipes.md`
- `references/v3-semantic-distillation.md`
- `references/distillation-summary.md`
