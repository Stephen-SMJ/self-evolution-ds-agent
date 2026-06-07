# AutoDS Offline Distillation V2

Date: 2026-06-07

## What V2 Adds

- Parses all 225 coverage bucket samples from 45 competitions.
- Builds one structured evolution trace per competition.
- Extracts validation, feature, model, training, postprocess, and risk signals per rank bucket.
- Generates domain-level recipes from repeated patterns.
- Keeps low-rank and fallback samples as contrastive evidence rather than equal-quality training data.

## Output Files

- `distillation/v2/competition_evolution_traces.json`
- `distillation/v2/domain_recipes.json`
- `distillation/v2/domain_recipes.md`
- `distillation/v2/competition_trace_index.md`
- `distillation/v2/summary.json`

## Coverage

- competitions: 45
- bucket files: 225
- domains: Audio, CV, GenAI, Medical, NLP, RL, RecSys, Tabular, Time-Series

## How AutoDS Should Use This

1. Detect the target competition domain.
2. Read the matching domain recipe.
3. Retrieve 1-3 similar competition traces from the same domain.
4. Build a baseline using the 70pct/40pct lessons.
5. Plan upgrades using 20pct/10pct/1st transitions.
6. Record which trace/recipe influenced each experiment in `AUTODS.md`.

## Limitations

- V2 is heuristic/static extraction, not full semantic reasoning over every notebook cell.
- Some keyword-level false positives may remain, so JSON traces should guide inspection rather than replace it.
- V3 should add LLM-assisted per-notebook summaries and structured recipe validation against held-out offline competitions.
