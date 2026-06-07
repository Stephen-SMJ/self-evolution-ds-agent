# First Distillation Summary

Corpus prepared from:

- `/home/ubuntu/proj/autods/self-evolution-ds-agent/distillation/core_manifest.csv`
- `/home/ubuntu/proj/autods/self-evolution-ds-agent/distillation/coverage_manifest.csv`

## Prepared Sets

Core set:

- 185 files
- 37 complete competitions
- quality counts: 115 strong, 57 usable, 13 weak
- recommended for main distillation/use

Coverage set:

- 225 files
- 45 complete competitions
- exactly 5 competitions per domain
- exactly 5 rank buckets per selected competition
- includes 13 fallback reject files and 16 weak files for coverage completeness

## Coverage By Domain

Coverage set has 5 complete competitions in each domain:

- Audio
- CV
- GenAI
- Medical
- NLP
- RL
- RecSys
- Tabular
- Time-Series

## Corpus-Level Signals

Frequent high-quality patterns:

- metric-aligned validation
- OOF predictions
- feature engineering by domain
- pretrained models for unstructured data
- GBM families for structured/time-series/ranking tasks
- controlled ensembling
- submission validation and experiment logging

The strongest cross-domain lesson is that competition performance improves through better validation and task-specific representation before blind model scaling.

