# Offline Evolution Rubric

Use this to convert rank-bucket observations into improvement steps.

## Rank Bucket Interpretation

70pct bucket:

- Often valid but naive.
- Useful for detecting minimum submission format, starter preprocessing, and common failure modes.
- Do not copy as a target solution.

40pct bucket:

- Usually has a real model and basic features.
- Good source for simple baselines and starter validation.

20pct bucket:

- Often contains task-specific features, stronger preprocessing, and improved validation.
- Use to identify high-ROI feature families.

10pct bucket:

- Often has robust validation, better models, OOF logic, thresholding/calibration, or first ensembles.
- Use as the first serious target.

1st / gold-near bucket:

- Often includes ensembling, careful leakage control, tuned validation, inference tricks, and postprocessing.
- Distill principles, not brittle code.

## Upgrade Ladder

Move through this ladder unless evidence says otherwise:

1. Valid submission.
2. Deterministic baseline.
3. Metric-aligned validation.
4. Domain-specific preprocessing.
5. Strong single model.
6. OOF error analysis.
7. Feature/model family expansion.
8. Calibration/threshold/postprocessing.
9. Controlled ensemble.
10. Reflection and skill update.

## Self-Evolution Log Template

Append this to `AUTODS.md` for every meaningful attempt:

```markdown
| id | change | local CV | public LB | gap | decision | lesson |
|----|--------|----------|-----------|-----|----------|--------|
| E001 | baseline LightGBM | 0.812 | 0.790 | -0.022 | keep | validation roughly aligned |
```

## Distillation Weights

Recommended training/use weights:

- `core_manifest.csv`, quality strong: 1.0
- `core_manifest.csv`, quality usable: 0.75
- `core_manifest.csv`, quality weak: 0.35
- `coverage_manifest.csv` weak/reject fallback: 0.10 to 0.20, use mainly for contrastive "what not to do" analysis
- `rejected_or_unused.csv`: do not train on by default

## Reflection Questions

After each experiment:

- What changed relative to the previous rank bucket?
- Did the change improve validation, leaderboard, both, or neither?
- Is the gain likely robust private-LB gain or public-LB noise?
- Which reusable skill should be extracted?
- Which assumption should be tested next?

