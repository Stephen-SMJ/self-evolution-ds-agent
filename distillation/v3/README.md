# AutoDS V3 LLM Semantic Distillation

V3 is the first intelligent distillation pass. Unlike V1/V2 rule extraction, it calls an OpenAI-style LLM to compare five rank buckets for each competition and infer the real upgrade path.

## Inputs

- `distillation/coverage_manifest.csv`
- `distillation/v2/competition_evolution_traces.json`
- raw notebooks/code referenced by the manifest

## Outputs

When a valid LLM API key is configured, the script writes:

- `distillation/v3/competition_traces/<domain>/<competition>.json`
- `distillation/v3/domain_recipes/<domain>.json`
- `distillation/v3/semantic_distillation_v3.md`
- `distillation/v3/summary.json`

## Run

```bash
cd /home/ubuntu/proj/autods
AutoDS/.venv/bin/python -u self-evolution-ds-agent/tools/distill_v3_llm_semantic.py \
  --root self-evolution-ds-agent
```

For a smoke test:

```bash
AutoDS/.venv/bin/python -u self-evolution-ds-agent/tools/distill_v3_llm_semantic.py \
  --root self-evolution-ds-agent \
  --limit 1 \
  --skip-domain-synthesis
```

## Required API Config

The script reads, in priority order:

- `AUTODS_API_KEY` or `OPENAI_API_KEY`
- `AUTODS_BASE_URL` or `OPENAI_BASE_URL`
- AutoDS defaults from `AutoDS/src/core/config.py`

Current smoke-test status: the default key/baseurl returned `401 invalid_key`, so full V3 semantic distillation cannot run until a valid OpenAI-style key is configured.

## What The LLM Extracts

For each competition:

- bucket summaries for `70pct`, `40pct`, `20pct`, `10pct`, `1st`
- core upgrades between adjacent rank buckets
- why each upgrade matters
- implementation hints for AutoDS
- gold-level principles
- validation guardrails
- failure modes and avoid-list

For each domain:

- recurring upgrade path
- baseline recipe
- validation recipe
- feature recipe
- model recipe
- postprocess/ensemble recipe
- common failure modes
- AutoDS decision policy

