# V4 Deep-Read Distillation for AutoDS

Date: 2026-06-07

Scope:

- 45 competitions
- 9 domains
- 225 rank-bucket notebook/code files
- Evidence source: `distillation/v4/evidence_packs_md/` and JSON packs generated from raw notebook/code content
- Detailed traces:
  - `deep_traces_batch1_tabular_timeseries_nlp.md`
  - `deep_traces_batch2_cv_medical_audio.md`
  - `deep_traces_batch3_recsys_genai_rl.md`

This V4 pass is deliberately not a keyword/rule-count summary. It reads per-competition rank-bucket evidence and extracts the mechanisms that explain movement from low/mid notebooks toward gold-level behavior. Some rank buckets contain low-quality or irrelevant artifacts; those are marked rather than blindly treated as gold.

## Main Finding

The repeated gold-level pattern is not "use a stronger model." It is:

1. decode the competition metric and submission mechanics
2. build a leakage-safe validation that resembles scoring
3. transform raw competition data into the right supervised representation
4. add domain-specific features or representations
5. generate OOF predictions and error diagnostics
6. specialize model/threshold/postprocessing to the metric
7. ensemble only after component validation is trustworthy

AutoDS should therefore evolve by creating better competition-specific experiment plans, not by applying one universal modeling stack.

## Cross-Domain Upgrade Factors

### 1. Metric alignment beats model swapping

Seen in Prudential, Web Traffic, OSIC, RSNA ICH, Freesound, RFCX, H&M, OTTO and LLM/generative tasks.

AutoDS behavior:

- Parse metric before modeling.
- Implement local metric exactly.
- Add postprocessing for the metric: thresholds, clipping, ranking order, confidence, SMAPE transforms, MAP@k, lwlrap, weighted logloss.
- Record every experiment with CV metric and public score separately.

### 2. Validation design is the first high-ROI skill

Seen in Spaceship Titanic, AMP, Cassava, OSIC, SIIM-ISIC, BirdCLEF, H&M, OTTO and RL/game tasks.

AutoDS behavior:

- Use `StratifiedKFold` for class balance.
- Use `GroupKFold` or patient/session/group-aware splits when leakage is possible.
- Use time-window validation for forecasting/recsys.
- Make audio/image validation mirror inference segmentation.
- For games, run tournaments rather than judging one episode.

### 3. Representation construction often matters more than model family

Seen in Predict Future Sales, Elo, H&M, OTTO, Expedia, Prostate/PANDA, RFCX, TGS, LLM Science and Drawing with LLMs.

AutoDS behavior:

- Convert raw logs into panel/candidate/ranker datasets.
- Convert WSI into tiles and slide aggregation.
- Convert audio into log-mel windows and aggregate windows at inference.
- Convert QA into retrieval context plus answer scoring.
- Convert SVG/generative tasks into generate-validate-score-select loops.

### 4. Domain features are the bridge from bronze to silver/gold

Examples:

- Titanic: title, family, cabin, fare/class interactions.
- House Prices: missingness semantics, ordinal quality, log target, outliers.
- AMP: shifted future targets, visit-month parsing, scored-regime filtering.
- BirdCLEF: secondary labels, species data coverage, soundscape windows.
- RecSys: recency, co-visitation, candidate sources, customer/product deltas.
- Medical: DICOM windows, patient groups, metadata, uncertainty.

AutoDS behavior:

- Extract domain-specific assumptions into an explicit "feature hypothesis" list.
- Test each hypothesis with local CV and error slices.
- Keep high-signal features even when they are simple.

### 5. OOF is the unit of trustworthy self-evolution

Seen across Cassava, SIIM-ISIC, House Prices, Prudential, LightGBM tabular, H&M/OTTO rerankers and NLP folds.

AutoDS behavior:

- Save OOF predictions for every model.
- Use OOF to compare, ensemble, calibrate and diagnose.
- Never tune ensemble weights from public leaderboard alone.

### 6. Low-quality notebooks must be filtered, not averaged

Examples:

- iMaterialist: downloader-only notebooks.
- Spaceship Titanic: selected 1st bucket is sample-submission-like and less useful than 40pct/20pct evidence.
- TGS Salt: 1st bucket is leaderboard/opponent analysis, not a model.
- Freesound: random sample submission.
- G-Research: random/no-validation evidence should not become a recipe.
- Several RL buckets are evaluation/scraping tools rather than training solutions.

AutoDS behavior:

- Label each notebook as one of: modeling, feature engineering, validation, inference/submission, EDA, downloader, metric/evaluation, write-up, low-value.
- Distill only the role it actually supports.
- Prefer lower-rank but complete notebooks over high-rank artifacts with no transferable method.

## Domain Playbooks

### Tabular

Default path:

1. inspect metric, target distribution and leakage columns
2. build simple baseline and exact submission validator
3. add semantic missingness, categorical encoding, interactions and grouped aggregations
4. compare linear/tree/GBM/CatBoost under folds
5. add OOF ensemble only after component stability

High-value patterns:

- Small tabular: human semantic features and stratified CV.
- Skewed regression: log target, RMSLE/RMSE alignment, outlier policy.
- Ordinal: threshold/cut optimization for QWK.
- Transaction panel: lagged monthly matrix and grouped aggregates.

### Time-Series

Default path:

1. decode prediction horizon and submission ID format
2. choose time-safe validation
3. create lag/rolling/calendar/regime features
4. align target transform to metric
5. add entity-specific models when entity IDs exist

High-value patterns:

- Forecasting often starts from seasonal median/shift baselines.
- Clinical time-series needs patient timeline parsing and scored-regime filtering.
- Market data needs time-safe lagging, asset/entity models and regime detection.

### NLP

Default path:

1. inspect label construction and text context boundaries
2. start with TF-IDF/linear baseline for sanity
3. move to transformer with folds and OOF
4. add task-specific reconstruction: span, discourse, pairwise, QA retrieval
5. ensemble only after error slices show complementary behavior

High-value patterns:

- Educational NLP: DeBERTa plus essay/discourse context.
- Duplicate/question tasks: pair features and leakage-safe split.
- Knowledge QA: retrieval recall before reader model.

### CV

Default path:

1. identify task type: classification, segmentation, retrieval, WSI
2. create image/mask/id integrity checks
3. train pretrained backbone with fold/OoF discipline
4. add augmentation, TTA and thresholding
5. use task metric: IoU, MAP@k, AUC, accuracy

High-value patterns:

- Classification: pretrained backbones + folds + augmentation.
- Segmentation: U-Net/encoder-decoder + IoU threshold search.
- Retrieval: embeddings/triplet/Siamese + MAP@5.
- WSI: tiling, stain normalization, slide aggregation.

### Medical

Default path:

1. enforce patient/group-aware validation
2. use modality-specific preprocessing: DICOM windows, WSI tiles, metadata
3. implement competition metric and calibration/uncertainty
4. combine image and tabular/clinical features
5. inspect false positives/false negatives by patient or study

High-value patterns:

- Patient leakage control is mandatory.
- Metadata-only and image-only baselines should both be tested.
- Confidence/uncertainty can be part of the target, not just a display value.

### Audio

Default path:

1. define clip/window/label alignment
2. build log-mel or embedding pipeline
3. use stratified/multilabel/group folds
4. train spectrogram CNN/timm/EfficientNet/ResNest models
5. match validation slicing to inference aggregation
6. tune thresholds for F1/lwlrap/MAP-like metrics

High-value patterns:

- Bird tasks are strongly affected by external/pretraining data and secondary labels.
- Detection tasks require segment-level aggregation parity.
- Noisy labels require robust validation and augmentation.

### RecSys

Default path:

1. define objective: next item, cluster, product delta, ranking, regression
2. build time-window validation
3. generate candidates and measure recall
4. add source/rank/recency/popularity/co-visitation features
5. train query-grouped rankers or strong heuristic rankers
6. optimize MAP@k/recall/NDCG locally

High-value patterns:

- Candidate recall comes before reranker quality.
- Transaction aggregation and recency dominate many baselines.
- Suppress already-owned/already-seen items when objective is "new".

### GenAI

Default path:

1. inspect hard output constraints and evaluator behavior
2. create candidate generation loop
3. validate format and constraints automatically
4. score with proxy metrics or local evaluator
5. select/blend/merge candidates

High-value patterns:

- Generative SVG/image tasks need generate-validate-score-select.
- LLM judge tasks require prompt search and proxy scoring.
- Knowledge LLM tasks are retrieval problems before answer generation.

### RL / Games / Optimization

Default path:

1. run official environment locally
2. make a valid baseline bot/submission
3. build tournament/evaluation harness
4. inspect replays and action validity
5. add rules, imitation or search according to problem type
6. select by repeated match results, not a single score

High-value patterns:

- Imitation from strong agents is often more practical than pure RL.
- Rule fallbacks and valid-action masks are essential.
- Some "RL" competitions are search/optimization; lower bounds and local search can beat neural RL.

## AutoDS Self-Evolution Loop for Offline Distillation

For every competition in the offline corpus, AutoDS should create:

1. `SPEC.md`: metric, data, submission, validation, task type
2. `NOTEBOOK_TRACE.md`: ranked notebook role labels and transferable ideas
3. `EXPERIMENTS.md`: baseline to stronger attempts with CV/public score
4. `OOF/`: per-model OOF predictions and diagnostics
5. `SKILLS.md`: reusable domain skills extracted from successful deltas

Distillation target:

- Do not memorize notebook code.
- Extract transition rules: "what changed from lower to higher rank, why it helps, and when it applies."
- Mark uncertainty and bad evidence explicitly.

## Immediate Changes for AutoDS Agent Behavior

When `/kaggle <competition>` starts, AutoDS should:

1. classify domain/task type
2. build exact local metric and submission validator
3. choose validation split by leakage risk
4. create the simplest valid baseline
5. run domain playbook high-ROI feature/representation upgrades
6. save OOF and error analysis
7. only then attempt ensembling or leaderboard submission

This is the practical core distilled from the 45-competition V4 read.

