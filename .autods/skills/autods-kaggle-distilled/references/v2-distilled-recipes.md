# V2 Distilled Recipes

V2 adds structured evolution traces over the full coverage set:

- 45 competitions
- 225 rank-bucket samples
- 9 domains
- 5 competitions per domain
- 5 buckets per competition: `70pct`, `40pct`, `20pct`, `10pct`, `1st`

Source files:

- `/home/ubuntu/proj/autods/self-evolution-ds-agent/distillation/v2/domain_recipes.json`
- `/home/ubuntu/proj/autods/self-evolution-ds-agent/distillation/v2/domain_recipes.md`
- `/home/ubuntu/proj/autods/self-evolution-ds-agent/distillation/v2/competition_evolution_traces.json`
- `/home/ubuntu/proj/autods/self-evolution-ds-agent/distillation/v2/competition_trace_index.md`

## How To Use V2

1. Detect the target competition domain.
2. Read the matching domain recipe from `domain_recipes.md` or `domain_recipes.json`.
3. Select 1-3 same-domain competition traces from `competition_trace_index.md`.
4. Use the `70pct` and `40pct` buckets to build a valid baseline and first serious model.
5. Use the `20pct`, `10pct`, and `1st` buckets to choose upgrade steps.
6. Record the chosen trace and recipe in `AUTODS.md`.

## Interpretation

V2 extraction is heuristic and static. Treat traces as retrieval guides:

- trust repeated domain-level patterns more than single notebook signals
- inspect raw notebooks when a signal looks surprising
- use fallback weak/reject files only as contrastive evidence
- prefer validation and representation improvements before ensembling

## Default Upgrade Template

```markdown
### Distilled Prior Used
- domain recipe:
- similar competition traces:
- baseline bucket used:
- upgrade bucket used:
- hypothesis:
- experiment id:
```

