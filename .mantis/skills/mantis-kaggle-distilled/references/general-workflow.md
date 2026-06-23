# General Kaggle Workflow Distilled for Mantis

## Setup

- Always create `MANTIS.md` before or during the first baseline.
- Keep all competition files under `competitions/<slug>/` when possible.
- Keep submissions under `submissions/<slug>/` or `competitions/<slug>/submissions/`.
- Record every experiment. A forgotten experiment is not reusable self-evolution.

## Competition Parsing

Extract these before modeling:

- task type: regression, binary classification, multiclass, multilabel, ranking, retrieval, segmentation, generation, reinforcement learning, forecasting
- metric and direction
- train/test shape
- ID column and prediction columns
- group/time/entity leakage risks
- public/private split risk
- submission file requirements
- GPU/internet/model restrictions

## Baseline

The baseline is successful only if it:

- creates a valid submission
- has deterministic seeds
- has a local validation score
- stores predictions and code
- updates `MANTIS.md`

Use small models first:

- Tabular: LightGBM/CatBoost/XGBoost plus simple preprocessing.
- NLP: TF-IDF + linear model, or a small transformer inference/training baseline if feasible.
- CV/Medical/Audio: pretrained model inference/training baseline with small image/audio size.
- Time-Series: lag/rolling features plus GBM, or naive/statistical baseline if data is small.
- RecSys: popularity/co-occurrence baseline.
- RL: valid heuristic/rule baseline before learned policy.

## Validation

Validation quality dominates model choice.

- Use stratified folds for classification.
- Use group folds when identities/users/patients/sessions repeat.
- Use time-aware splits for forecasting and market tasks.
- Use OOF predictions for stacking, calibration, and error analysis.
- Never optimize only public LB unless CV is clearly broken and documented.

## Iteration

Prefer high-information experiments:

- one feature family at a time
- one model family at a time
- one validation change at a time
- one ensemble change at a time

Each experiment should answer a question:

- Is validation reliable?
- Which error slice dominates?
- Is the model underfitting or overfitting?
- Is the bottleneck data cleaning, representation, model capacity, leakage, or submission format?

## Submission Discipline

- Validate row count, ID order, column names, dtypes, and value ranges.
- Save each submission with experiment ID.
- Use short meaningful submit messages.
- After leaderboard score returns, update `MANTIS.md` with CV, LB, gap, and interpretation.

## When To Ensemble

Ensemble after at least two individual models have validated value.

Useful ensemble types:

- mean/rank averaging for regression and probabilities
- weighted average from OOF scores
- model family blend: GBM + neural, transformer + linear, CNN + transformer
- fold ensemble / seed ensemble

Do not use ensemble as a substitute for fixing broken validation.

