# Domain Playbooks

These are first-version distilled patterns from the local notebook corpus. Treat them as priors, not rules.

## Tabular

Strong patterns:

- robust validation, often stratified/group-aware
- LightGBM/XGBoost/CatBoost as first serious models
- categorical encoding, missingness indicators, count/frequency features
- target/log transforms for skewed regression targets
- OOF predictions and model blending

High-ROI sequence:

1. EDA for leakage, missingness, target skew, categorical cardinality.
2. Baseline GBM with correct metric.
3. Feature families: counts, interactions, aggregations, target/frequency encoding.
4. Tune regularization, depth/leaves, learning rate, early stopping.
5. Blend GBM families and seeds.

Avoid:

- public LB-only tuning
- target encoding without fold isolation
- dropping rare categories without checking signal

## Time-Series

Strong patterns:

- lag, rolling, expanding, calendar, event, entity aggregate features
- time-aware validation
- GBM baselines remain competitive
- ensembles across horizons/models

High-ROI sequence:

1. Define forecast horizon and validation cutoffs.
2. Build naive baseline and lag/rolling GBM.
3. Add calendar/event features.
4. Add entity-level aggregates and target transformations.
5. Evaluate horizon-level errors.
6. Blend specialized horizon/entity models if useful.

Avoid:

- random CV over time
- leakage from future aggregate features
- using test-period information not available at inference time

## NLP

Strong patterns:

- transformer models dominate strong entries
- TF-IDF/linear baselines are useful sanity checks
- cleaning is task dependent, not always beneficial
- OOF predictions support thresholding, stacking, and calibration

High-ROI sequence:

1. Build TF-IDF baseline.
2. Inspect label imbalance, length distribution, duplicate text, leakage.
3. Fine-tune a pretrained transformer if compute allows.
4. Use stratified/group folds.
5. Tune max length, pooling, thresholds, and loss.
6. Ensemble transformer checkpoints/seeds or blend with linear baseline.

Avoid:

- aggressive text cleaning before testing
- training without fold discipline
- relying only on public LB for threshold selection

## CV

Strong patterns:

- pretrained CNN/ViT backbones
- augmentation, TTA, fold/seed ensembles
- image-size and crop policy matter
- validation split must match source/entity leakage risk

High-ROI sequence:

1. Verify labels, file paths, image sizes, corrupt files.
2. Build pretrained baseline with modest resolution.
3. Add augmentations suitable for task.
4. Tune resolution, backbone, loss, class imbalance.
5. Add TTA and fold/seed ensembles.
6. For segmentation/detection, inspect masks/boxes visually.

Avoid:

- augmentations that break labels
- random splits when same subject/object appears multiple times
- heavy model before data path and validation are proven

## Medical

Strong patterns:

- subject/patient-level validation is critical
- pretrained image models plus careful preprocessing
- ensembling is common in strong solutions
- data leakage risk is high

High-ROI sequence:

1. Identify patient/study/group columns.
2. Use group-aware validation.
3. Inspect image/windowing/normalization/mask quality.
4. Build pretrained baseline.
5. Add domain preprocessing and calibration.
6. Ensemble folds/seeds/backbones only after validation is trusted.

Avoid:

- image-level random split when patient appears multiple times
- leaderboard chasing on small public split
- ignoring class imbalance and calibration

## Audio

Strong patterns:

- convert waveforms to spectrogram/log-mel features
- pretrained audio/CNN pipelines
- clip-level to recording-level aggregation
- augmentation and TTA can help

High-ROI sequence:

1. Validate sampling rate, duration, labels, and missing/corrupt audio.
2. Build spectrogram/log-mel baseline.
3. Add pretrained audio/CNN model.
4. Add clip sampling, mixup/noise/time/frequency masking.
5. Aggregate clip predictions carefully.
6. Tune thresholds for multilabel tasks.

Avoid:

- random or all-zero/all-one baselines as learning signal
- ignoring label noise and weak labels
- inconsistent train/test audio preprocessing

## RecSys

Strong patterns:

- popularity/co-occurrence baselines are strong sanity checks
- candidate generation plus ranking is the main architecture
- time-aware validation matters
- feature aggregation over user/item/session is high ROI

High-ROI sequence:

1. Build popularity and recent-item baseline.
2. Create candidate generators: popularity, co-visitation, item-item, user history.
3. Build ranker with user/item/session features.
4. Validate with time-aware split and target metric.
5. Tune candidate recall before ranker complexity.
6. Blend candidate sources and rankers.

Avoid:

- training only a ranker without candidate recall checks
- random validation over temporal interactions
- optimizing proxy metrics that do not correlate with competition metric

## GenAI / LLM

Strong patterns:

- retrieval, prompting, reranking, and postprocessing are often as important as model choice
- external model/data restrictions must be checked early
- ensembles of prompts/checkpoints/rerankers are common

High-ROI sequence:

1. Parse rules for internet, external models, generated data, and notebooks.
2. Build a deterministic prompt/retrieval baseline.
3. Add validation harness with exact competition metric.
4. Improve retrieval/context construction.
5. Add reranking or model ensemble if allowed.
6. Harden output parser and fallback behavior.

Avoid:

- invalid external assets
- non-deterministic output without logging
- optimizing prompt examples only on public LB

## RL / Simulation

Strong patterns:

- valid heuristic baseline first
- local simulator/evaluator is essential
- fast self-play/evaluation loop matters more than complex model at first
- strong solutions often combine rules, search, imitation, or lightweight learned policies

High-ROI sequence:

1. Run a valid sample agent locally.
2. Implement deterministic heuristic baseline.
3. Build evaluation harness with many seeds/opponents.
4. Profile bottlenecks.
5. Add search/rules/imitation incrementally.
6. Track win rate confidence intervals, not single games.

Avoid:

- trusting tiny evaluation samples
- changing many heuristics at once
- slow agents that fail runtime constraints

