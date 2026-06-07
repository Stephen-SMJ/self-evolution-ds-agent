# Self-Evolution DS Agent Corpus

This repository is the offline distillation workspace for building **AutoDS**, a self-evolving data science agent focused on Kaggle-style competitions.

The goal is not to copy public notebooks. The goal is to study many competitions across domains and rank levels, then distill the upgrade mechanisms that move a weak baseline toward medal-level behavior: better metric alignment, leakage-safe validation, representation construction, domain-specific features, OOF diagnostics, postprocessing, and controlled ensembling.

## Project Goal

AutoDS is intended to become an interactive agent that can take a Kaggle competition link, inspect the task, download data, understand the metric and submission format, build baselines, run experiments, submit results, read scores, reflect on failures, and evolve its strategy.

This repository supports the first phase:

**Offline self-evolution distillation**

- Collect public Kaggle notebooks across many domains.
- Organize them by competition and approximate rank bucket.
- Prefer competitions with five rank buckets: `70pct`, `40pct`, `20pct`, `10pct`, `1st`.
- Analyze how stronger notebooks differ from weaker notebooks.
- Convert those differences into reusable domain playbooks for AutoDS.

The second phase, online competition self-evolution, is intentionally not the main focus of this repo yet.

## Current Corpus

Current local corpus summary:

- 9 domains
- 71 collected competitions in the raw directory tree
- 375 rank-bucket directories in the raw tree
- 45 competitions selected for full V4 evidence distillation
- 225 selected rank-bucket notebook/code files for full five-bucket analysis

The V4-selected competitions are balanced across domains:

| Domain | Selected Competitions |
| --- | ---: |
| Audio | 5 |
| CV | 5 |
| GenAI | 5 |
| Medical | 5 |
| NLP | 5 |
| RL | 5 |
| RecSys | 5 |
| Tabular | 5 |
| Time-Series | 5 |

## Repository Layout

```text
.
├── Audio/
├── CV/
├── GenAI/
├── Medical/
├── NLP/
├── RL/
├── RecSys/
├── Tabular/
├── Time-Series/
├── distillation/
│   ├── reports/
│   │   ├── first_distillation_v1.md
│   │   └── offline_distillation_v2.md
│   ├── v2/
│   │   ├── competition_evolution_traces.json
│   │   ├── competition_trace_index.md
│   │   ├── domain_recipes.json
│   │   ├── domain_recipes.md
│   │   └── summary.json
│   ├── v3/
│   │   ├── README.md
│   │   └── codex_semantic_distillation_v3.md
│   ├── v4/
│   │   ├── deep_distillation_v4.md
│   │   ├── deep_traces_batch1_tabular_timeseries_nlp.md
│   │   ├── deep_traces_batch2_cv_medical_audio.md
│   │   ├── deep_traces_batch3_recsys_genai_rl.md
│   │   ├── evidence_packs/
│   │   ├── evidence_packs_md/
│   │   └── summary.json
│   ├── all_scored.csv
│   ├── core_manifest.csv
│   ├── coverage_manifest.csv
│   ├── distillation_ready_manifest.csv
│   ├── rejected_or_unused.csv
│   └── summary.json
├── tools/
│   ├── prepare_distillation_corpus.py
│   ├── distill_v2_evolution_traces.py
│   ├── distill_v3_llm_semantic.py
│   └── build_v4_evidence_packs.py
└── download_notebooks.py
```

Each raw competition directory follows this pattern:

```text
<Domain>/<competition-slug>/rank<rank>_<bucket>/
```

Example:

```text
Tabular/titanic/rankXXXX_70pct/
Tabular/titanic/rankXXXX_40pct/
Tabular/titanic/rankXXXX_20pct/
Tabular/titanic/rankXXXX_10pct/
Tabular/titanic/rankXXXX_1st/
```

## Distillation Versions

### V1: First Manual Distillation

Primary output:

- `distillation/reports/first_distillation_v1.md`

V1 was the first human-readable distillation pass. It summarized the initial corpus and extracted broad Kaggle competition lessons. It was useful for orienting the project, but it was not yet systematic enough to drive AutoDS behavior directly.

V1 characteristics:

- High-level summaries.
- Early domain observations.
- Initial hypothesis about what AutoDS should learn.
- Limited evidence traceability.

Use V1 for historical context only.

### V2: Rule-Based Corpus and Evolution Traces

Primary outputs:

- `distillation/reports/offline_distillation_v2.md`
- `distillation/v2/domain_recipes.md`
- `distillation/v2/domain_recipes.json`
- `distillation/v2/competition_evolution_traces.json`
- `distillation/v2/competition_trace_index.md`
- `distillation/v2/summary.json`

Primary scripts:

- `tools/prepare_distillation_corpus.py`
- `tools/distill_v2_evolution_traces.py`

V2 made the corpus operational. It scored notebook/code files, selected usable files, rejected low-value artifacts, and generated structured traces using pattern detection.

V2 extracted signals such as:

- validation patterns: KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, OOF
- feature patterns: aggregation, lagging, categorical encoding, text/image/audio transforms
- model patterns: GBM, CNN, transformer, linear/tree baselines
- training patterns: early stopping, scheduling, augmentation, imbalance handling
- postprocessing patterns: thresholds, clipping, ranking, ensembling
- risk patterns: random submissions, downloader-only notebooks, thin artifacts

V2 limitations:

- It was mostly rule-based.
- It could detect that a notebook contained `KFold` or `LightGBM`, but not always why that mattered.
- It could overvalue high-rank artifacts that were only downloaders, sample submissions, or leaderboard analysis.

Use V2 as a structured index and fallback evidence layer.

### V3: Semantic Distillation

Primary outputs:

- `distillation/v3/README.md`
- `distillation/v3/codex_semantic_distillation_v3.md`

Primary script:

- `tools/distill_v3_llm_semantic.py`

V3 introduced semantic analysis. The intent was to compare five rank buckets for each competition and infer the core upgrades from weaker solutions to stronger/gold-level solutions.

The current usable V3 output is:

- `distillation/v3/codex_semantic_distillation_v3.md`

V3 characteristics:

- More intelligent than V2 pattern counting.
- Focuses on the transition from lower-rank to higher-rank solutions.
- Extracts domain recipes and competition-level improvement logic.

V3 limitation:

- The external-LLM automation path was implemented, but one API smoke test failed with `401 invalid_key` in the local environment. The committed V3 Codex semantic report remains useful, but V4 supersedes it for planning.

Use V3 as semantic support when V4 needs more context or when comparing against older recipes.

### V4: Deep-Read Distillation

Primary outputs:

- `distillation/v4/deep_distillation_v4.md`
- `distillation/v4/deep_traces_batch1_tabular_timeseries_nlp.md`
- `distillation/v4/deep_traces_batch2_cv_medical_audio.md`
- `distillation/v4/deep_traces_batch3_recsys_genai_rl.md`
- `distillation/v4/evidence_packs/`
- `distillation/v4/evidence_packs_md/`
- `distillation/v4/summary.json`

Primary script:

- `tools/build_v4_evidence_packs.py`

V4 is the current planning-quality distillation layer.

V4 reads per-competition evidence from all five rank buckets and then manually distills the actual improvement mechanisms. It explicitly filters bad evidence instead of blindly trusting rank labels.

V4 covers 45 selected competitions:

- 5 Audio
- 5 CV
- 5 GenAI
- 5 Medical
- 5 NLP
- 5 RL
- 5 RecSys
- 5 Tabular
- 5 Time-Series

V4 key conclusion:

The repeated gold-level pattern is not simply "use a stronger model." The real upgrade path is:

1. decode the competition metric and submission mechanics
2. build leakage-safe validation that resembles hidden scoring
3. transform raw competition data into the right supervised representation
4. add domain-specific features or pretrained representations
5. generate OOF predictions and error diagnostics
6. specialize model, threshold, postprocessing, or confidence to the metric
7. ensemble only after individual components are validated

V4 also marks low-quality notebook types:

- downloader-only notebooks
- random/sample submissions
- leaderboard-only analysis
- incomplete artifacts
- write-ups with no transferable implementation
- high-rank buckets that are not actually good modeling evidence

Use V4 as the primary reference for AutoDS competition planning.

## Domain Lessons from V4

Tabular:

- Human semantic features, missingness semantics, metric alignment, categorical handling, grouped aggregation, OOF, and conservative GBM/CatBoost/linear comparisons matter more than blindly increasing model size.

Time-Series:

- Decode horizon and submission IDs first. Use time-safe validation, lag/rolling/calendar features, target transforms, and entity/regime-aware modeling.

NLP:

- Validate label construction and text context. Use transformer folds after simple sanity baselines. For QA, retrieval recall is often more important than reader strength.

CV:

- Identify the actual task: classification, segmentation, retrieval, or WSI. Use pretrained backbones with folds, augmentation, OOF, task metrics, TTA, and threshold/ranking logic.

Medical:

- Patient/group-aware validation is mandatory. Use modality-specific preprocessing such as DICOM windowing or WSI tiling. Metadata-only, image-only, and combined models should all be tested.

Audio:

- Define clip/window/label alignment. Use log-mel or pretrained embeddings, multilabel/group folds, validation-inference parity, and metric threshold tuning.

RecSys:

- Candidate generation and recall come before reranking. Use time-window validation, recency/popularity/co-visitation/source/rank features, and query-grouped rankers.

GenAI:

- Build a generate-validate-score-select loop. Enforce hard output constraints and use proxy metrics/evaluators. For LLM knowledge tasks, retrieval comes before answer generation.

RL / Games / Optimization:

- Build the official environment, valid submission, and repeated tournament harness first. Use rules, imitation, search, or RL according to the problem type. Some "RL" tasks are better solved as optimization/search.

## How to Regenerate Artifacts

From the repository root:

```bash
python3 tools/prepare_distillation_corpus.py
python3 tools/distill_v2_evolution_traces.py
python3 tools/build_v4_evidence_packs.py --root . --manifest distillation/coverage_manifest.csv --out-dir distillation/v4
```

The V3 external-LLM path is available but not required for the current V4 output:

```bash
python3 tools/distill_v3_llm_semantic.py --root .
```

## How AutoDS Should Use This Repo

For a new competition, AutoDS should:

1. classify domain and task type
2. inspect rules, data, metric, sample submission, and scoring mechanics
3. implement exact local metric and submission validator
4. choose validation by leakage risk: stratified, grouped, temporal, patient/session-aware, or tournament-based
5. create the simplest valid baseline
6. apply the relevant V4 domain playbook
7. save OOF predictions, diagnostics, and experiment logs
8. compare CV and public leaderboard gaps
9. distill each successful delta into a reusable skill

The agent should not copy notebooks directly. It should convert notebook evidence into hypotheses, experiments, and reusable skills.

## Git Identity Used for This Repo

Commits should use:

```bash
git config user.name "Stephen-SMJ"
git config user.email "s23471160103@163.com"
```

