# V4 Deep-Read Kaggle Distillation

Date: 2026-06-07

This reference is a compact operational version of the V4 deep-read distillation from:

`offline/distillation/v4/deep_distillation_v4.md`

Use V4 before earlier rule-based summaries when planning a Kaggle workflow.

## Core Rule

Do not start by asking "which model is strongest?" Start by answering:

1. What is the exact metric and submission mechanic?
2. What validation split best matches hidden scoring and avoids leakage?
3. What representation converts raw data into the scored task?
4. What domain features or pretrained representations are likely decisive?
5. What OOF/error evidence proves an upgrade works?

## Domain Defaults

Tabular:

- small classification: semantic features, stratified CV, simple model comparison
- skewed regression: log target, missingness semantics, outlier policy, RMSLE alignment
- ordinal: metric-specific threshold/cut optimization
- transaction panel: group aggregates, lag features, target clipping

Time-series:

- parse horizon/submission IDs
- use time-safe validation
- add lag/rolling/calendar/regime/entity features
- align target transform to SMAPE/RMSE/correlation-style metric

NLP:

- validate label construction and context boundaries
- use TF-IDF/linear sanity baseline, then transformer folds
- add task-specific reconstruction: span, pair, discourse, retrieval
- for QA, retrieval recall comes before reader strength

CV:

- classify task: image classification, segmentation, retrieval, WSI
- use pretrained backbones with fold/OOF discipline
- add task-specific metric: IoU, MAP@k, AUC, accuracy
- segmentation needs threshold search; retrieval needs embeddings/ranking; WSI needs tiling/aggregation

Medical:

- patient/group-aware validation is mandatory
- use modality preprocessing: DICOM windows, WSI tiles, metadata
- test image-only, metadata-only and combined models
- include calibration/uncertainty when metric requires it

Audio:

- define clip/window/label alignment
- use log-mel or pretrained audio embeddings
- use multilabel/group folds when needed
- make validation slicing match inference aggregation
- tune thresholds for F1/lwlrap

RecSys:

- define objective and time-window validation
- generate candidates first and measure recall
- add recency/popularity/co-visitation/source/rank features
- train query-grouped rankers only after candidate recall is acceptable

GenAI:

- build generate-validate-score-select loop
- enforce hard format constraints
- use proxy metrics/evaluators for candidate selection
- for LLM QA, solve retrieval before answer generation

RL / games / optimization:

- run official environment locally
- create valid baseline submission
- build repeated tournament/evaluation harness
- use rules, imitation or search according to problem type
- keep valid-action masks and rule fallbacks

## Notebook Evidence Quality Filter

Label every notebook before distilling:

- modeling
- feature engineering
- validation/metric
- inference/submission
- EDA
- downloader/data ops
- evaluation/replay/scraping
- write-up
- low-value artifact

Do not treat a high-rank bucket as gold if it is only a downloader, sample submission, random guess, leaderboard analysis, or incomplete artifact.

## Required Experiment Artifacts

For every serious attempt, create:

- exact metric implementation
- submission validator
- fold or validation split definition
- OOF predictions when feasible
- error slices by class/group/time/entity
- experiment log with CV score, public score, gap and next hypothesis

