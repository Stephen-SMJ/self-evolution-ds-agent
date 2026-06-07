# AutoDS Offline Distillation V1

Date: 2026-06-06

## Goal

Convert the offline Kaggle notebook corpus into reusable AutoDS competition skills for the first stage of self-evolution.

This is not model fine-tuning. It is the first symbolic distillation pass: extracting workflow, validation, feature, modeling, and reflection patterns into an AutoDS skill that can guide future competition runs.

## Input Corpus

Prepared corpus:

- `distillation/core_manifest.csv`
- `distillation/core_files/`
- `distillation/coverage_manifest.csv`
- `distillation/coverage_files/`

Corpus statistics:

- Scored files: 365
- Core files: 185
- Core complete competitions: 37
- Coverage files: 225
- Coverage complete competitions: 45
- Coverage domains: 9 domains x 5 complete competitions

## Distillation Output

AutoDS skill:

- `/home/ubuntu/proj/autods/AutoDS/.autods/skills/autods-kaggle-distilled/SKILL.md`

References:

- `references/general-workflow.md`
- `references/domain-playbooks.md`
- `references/evolution-rubric.md`
- `references/distillation-summary.md`

## Main Distilled Principles

1. Validation is the main lever.
   Strong notebooks repeatedly use folds, OOF predictions, group/time-aware splits, or domain-specific validation. Weak notebooks often jump directly to submission.

2. Domain representation beats blind model scaling.
   Tabular/time-series/RecSys solutions rely on feature families and aggregates. CV/Medical/Audio/NLP solutions rely on pretrained representation plus task-specific preprocessing.

3. Strong solutions evolve through experiment logs.
   Reusable improvement comes from recording CV, leaderboard, gap, and decision after each change.

4. Ensembling is late-stage.
   It is common in top buckets, but should come after strong single models and trustworthy validation.

5. Low-rank notebooks are still useful as contrast.
   They often reveal valid submission format and common mistakes, but should not receive equal distillation weight.

## Recommended Use

Use `core_files` for main distillation and future skill extraction.

Use `coverage_files` only for:

- maintaining complete domain/rank-bucket coverage
- comparing lower-rank and higher-rank trajectories
- contrastive lessons such as "what not to do"

Do not train equally on fallback weak/reject files.

## Next Distillation Pass

V2 should extract structured JSON skills from each domain:

- validation recipes
- feature recipes
- model recipes
- postprocessing recipes
- ensemble recipes
- failure modes

Then AutoDS can retrieve domain-specific recipes during `/kaggle` runs.

