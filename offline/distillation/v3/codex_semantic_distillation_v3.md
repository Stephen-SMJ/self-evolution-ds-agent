# AutoDS Offline Distillation V3 - Codex Semantic Pass

Date: 2026-06-07

This is the first intelligent distillation pass done directly by Codex, without calling an external LLM API.

Inputs used:

- `distillation/coverage_manifest.csv`
- `distillation/v2/competition_evolution_traces.json`
- `distillation/v2/domain_recipes.md`
- notebook names, rank buckets, quality scores, and extracted V2 evidence

Scope:

- 9 domains
- 45 competitions
- 225 rank-bucket notebook/code samples
- rank ladder: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

Important caveat:

This pass is semantic and judgment-based, but it is not yet a full cell-by-cell human-style reading of all 225 notebooks. It is already more intelligent than V2 rule extraction because it reasons about what the rank transitions mean and what AutoDS should learn, but V4 should still deepen individual notebook summaries.

## Core Cross-Domain Finding

The main upgrade from weak to gold-level Kaggle solutions is rarely "use a bigger model" by itself.

The repeated upgrade path is:

1. Make a valid submission.
2. Build a local validation protocol that reflects the leaderboard split.
3. Convert raw data into the right representation for the domain.
4. Train one strong, reproducible model with metric-aligned validation.
5. Analyze OOF/errors and fix the largest failure slice.
6. Add task-specific postprocessing.
7. Ensemble only after individual models and validation are trusted.
8. Keep a written experiment ledger so improvements become reusable skills.

The strongest notebook buckets tend to add one or more of:

- better validation split
- OOF predictions
- domain-specific feature representation
- pretrained backbone or strong model family
- fold/seed/model ensembling
- calibrated thresholds or postprocessing
- robust inference/submission handling

The weakest buckets often fail because they are:

- sample-submission-shaped only
- random or naive predictions
- downloader/preprocessing-only notebooks
- leaderboard-only tweaks without validation
- single model without error analysis
- useful demonstrations but not complete competition strategy

## Domain Recipes

### Tabular

Observed competitions:

- `titanic`
- `house-prices-advanced-regression-techniques`
- `spaceship-titanic`
- `competitive-data-science-predict-future-sales`
- `prudential-life-insurance-assessment`

Semantic upgrade path:

1. Low-rank solutions usually have a valid sklearn/GBM model with basic imputation and categorical handling.
2. Mid-rank solutions add EDA-driven features: family/group features, missingness indicators, target transforms, lag/aggregate features, and categorical encodings.
3. High-rank solutions improve validation and model diversity: KFold/stratified folds, OOF, LightGBM/XGBoost/CatBoost, ridge/linear blends, seed/fold ensembles.
4. Gold-level solutions are not just more features; they align validation with the metric and use conservative ensembling/postprocessing.

AutoDS recipe:

- Start with a deterministic baseline using LightGBM/CatBoost or sklearn fallback.
- Immediately identify leakage risks, target skew, categorical cardinality, and metric direction.
- Add features in families: missingness, counts/frequency, interactions, group aggregates, target/log transforms.
- Use fold-isolated target encoding only if validation is reliable.
- Tune regularization before adding complex ensembles.
- Blend GBM families only after each has an OOF score.

Core lesson:

Tabular gold performance comes from validation discipline plus feature representation, not blind hyperparameter tuning.

### Time-Series

Observed competitions:

- `amp-parkinsons-disease-progression-prediction`
- `g-research-crypto-forecasting`
- `tabular-playground-series-jan-2022`
- `ubiquant-market-prediction`
- `web-traffic-time-series-forecasting`

Semantic upgrade path:

1. Low-rank solutions often use naive regressors, random predictions, or simple tree pipelines.
2. Mid-rank solutions add lags, rolling windows, entity/time aggregates, holidays, regime features, or domain-specific time cutoffs.
3. High-rank solutions use time-aware validation, horizon-aware modeling, model ensembles, and postprocessing matched to the metric.
4. Gold-level solutions are careful about future leakage and public/private split instability.

AutoDS recipe:

- Never start with random CV for forecasting/market tasks.
- Define cutoff-based validation before modeling.
- Build a naive baseline, then lag/rolling/aggregate GBM.
- Add calendar/event/regime/entity features.
- Evaluate per horizon, per entity, and per time regime.
- Use ensembles across model classes/horizons only if validation is time-safe.

Core lesson:

Time-series improvement is mostly about controlling time leakage and representing temporal structure. Model choice is secondary to split design and feature timing.

### NLP

Observed competitions:

- `commonlitreadabilityprize`
- `feedback-prize-english-language-learning`
- `google-quest-challenge`
- `nlp-getting-started`
- `quora-question-pairs`

Semantic upgrade path:

1. Low-rank solutions often use simple notebooks, shallow text features, or incomplete high-rank references.
2. Mid-rank solutions add TF-IDF, cleaned text, classical ML, or basic BERT/transformer fine-tuning.
3. High-rank solutions use pretrained transformers with folds, max-length tuning, OOF predictions, seed/fold ensembles, and threshold/postprocess logic.
4. Gold-level solutions combine model quality with target/metric-specific postprocessing.

AutoDS recipe:

- Always build a TF-IDF/linear or simple transformer sanity baseline.
- Inspect duplicates, leakage, length distribution, label imbalance, and group/user splits.
- Prefer pretrained transformer fine-tuning with stratified/group folds.
- Track OOF predictions for thresholding and error slicing.
- Blend transformer checkpoints/seeds only after single-fold behavior is understood.
- Do not assume aggressive text cleaning helps.

Core lesson:

NLP upgrades are mainly representation upgrades plus validation/threshold discipline. The jump from bronze to gold is often OOF transformer reliability, not only a larger model.

### CV

Observed competitions:

- `cassava-leaf-disease-classification`
- `digit-recognizer`
- `humpback-whale-identification`
- `imaterialist-challenge-fashion-2018`
- `tgs-salt-identification-challenge`

Semantic upgrade path:

1. Low-rank solutions are often starter CNNs, simple preprocessing, or even downloader-only artifacts.
2. Mid-rank solutions add real pretrained backbones, stronger augmentations, segmentation/detection architecture where appropriate, and visual error analysis.
3. High-rank solutions tune image size/crops, TTA, fold/seed ensembles, and metric-specific postprocessing.
4. Gold-level solutions protect against identity/source leakage and use task-specific modeling: U-Net for segmentation, metric learning for retrieval, pretrained CNN/ViT for classification.

AutoDS recipe:

- First verify file paths, labels, corrupt images, class imbalance, and sample submission.
- Choose architecture by task: classifier, segmenter, detector, or retrieval/metric-learning model.
- Use pretrained backbone before custom architecture.
- Use augmentations that preserve labels.
- Use group-aware validation for subject/object identity tasks.
- Add TTA and fold/seed ensembles only after validation is trusted.

Core lesson:

CV gold performance comes from matching architecture and validation to the visual task. A generic CNN baseline is rarely the main differentiator.

### Medical

Observed competitions:

- `osic-pulmonary-fibrosis-progression`
- `prostate-cancer-grade-assessment`
- `rsna-breast-cancer-detection`
- `rsna-intracranial-hemorrhage-detection`
- `siim-isic-melanoma-classification`

Semantic upgrade path:

1. Low-rank solutions often show data loading, EDA, basic CNNs, or tabular/image baselines.
2. Mid-rank solutions add domain preprocessing: windowing/cropping, tiling, stain normalization, patient-level features, or adjacent slices.
3. High-rank solutions use patient/group-aware validation, pretrained medical/image backbones, OOF, calibration, and fold ensembles.
4. Gold-level solutions often combine multiple modalities or model families and heavily guard against patient leakage.

AutoDS recipe:

- Identify patient/study/group IDs before any split.
- Use group-aware validation by patient/study.
- Inspect domain preprocessing requirements: DICOM/windowing, tiling, stain normalization, slice context, image resolution.
- Build a strong pretrained baseline.
- Add calibration and thresholding because medical metrics often punish miscalibration.
- Ensemble folds/backbones only after group validation is stable.

Core lesson:

Medical competitions reward leakage-safe validation and domain preprocessing more than generic model scaling.

### Audio

Observed competitions:

- `birdclef-2021`
- `birdclef-2022`
- `birdclef-2023`
- `freesound-audio-tagging-2019`
- `rfcx-species-audio-detection`

Semantic upgrade path:

1. Low-rank solutions often produce valid predictions using simple spectrogram/CNN pipelines or naive/random baselines.
2. Mid-rank solutions improve audio representation: log-mel, STFT/inverse STFT, clip sampling, cleaning, metadata, and label-noise handling.
3. High-rank solutions use pretrained audio/CNN models, fold validation, augmentation, TTA, secondary labels, and clip-to-recording aggregation.
4. Gold-level solutions are often robust inference systems: pretraining, external/extra data where allowed, secondary-label strategy, threshold tuning, and stable aggregation.

AutoDS recipe:

- Standardize sampling rate, duration, clipping strategy, and spectrogram pipeline.
- Build log-mel/spectrogram baseline.
- Use pretrained audio/CNN model if available.
- Handle weak labels, secondary labels, and label noise explicitly.
- Aggregate clip predictions carefully into recording-level outputs.
- Tune thresholds for multilabel metrics.

Core lesson:

Audio improvement is representation and aggregation: how waveform becomes model input, and how clip predictions become competition predictions.

### RecSys

Observed competitions:

- `elo-merchant-category-recommendation`
- `expedia-hotel-recommendations`
- `h-and-m-personalized-fashion-recommendations`
- `otto-recommender-system`
- `santander-product-recommendation`

Semantic upgrade path:

1. Low-rank solutions often use popularity/recent-item rules or incomplete ranking demos.
2. Mid-rank solutions add time/user/item/session aggregates and co-occurrence/candidate generation.
3. High-rank solutions separate candidate generation from ranking and validate using the competition retrieval metric.
4. Gold-level solutions optimize recall first, then ranker quality, then blend candidate sources/rankers.

AutoDS recipe:

- Start with popularity and recent-history baseline.
- Build multiple candidate generators: popularity, recent items, co-visitation, item-item, user history.
- Measure candidate recall before training a ranker.
- Use time-aware validation.
- Train ranker with user/item/session/time features.
- Blend candidate sources and rankers only after recall bottleneck is addressed.

Core lesson:

RecSys gold performance is a two-stage system problem: candidate recall first, ranking second.

### GenAI / LLM

Observed competitions:

- `drawing-with-llms`
- `feedback-prize-2021`
- `feedback-prize-effectiveness`
- `kaggle-llm-science-exam`
- `llms-you-cant-please-them-all`

Semantic upgrade path:

1. Low-rank solutions often rely on simple prompts, single model inference, or basic transformer baselines.
2. Mid-rank solutions improve retrieval/context, reranking, prompt construction, or task-specific parser/postprocessor.
3. High-rank solutions combine model families, prompt ensembles, sentence/embedding features, and robust output formatting.
4. Gold-level solutions are systems: retrieval + model + reranker + parser + fallback + rule compliance.

AutoDS recipe:

- Parse rules for allowed external models/data/internet before modeling.
- Build deterministic baseline with exact metric.
- Add retrieval/context construction if the task is knowledge-heavy.
- Add reranking or verifier where output candidates exist.
- Harden output parser and fallback behavior.
- Use prompt/model ensembles only if deterministic validation supports them.

Core lesson:

GenAI competitions are not solved by prompting alone; gold solutions are controlled pipelines with retrieval, verification, and output discipline.

### RL / Simulation

Observed competitions:

- `google-football`
- `halite`
- `kore-2022`
- `lux-ai-2021`
- `santa-2022`

Semantic upgrade path:

1. Low-rank solutions often are starter agents, visualization, or simple heuristics.
2. Mid-rank solutions add rule improvements, search, imitation learning, or local evaluation utilities.
3. High-rank solutions emphasize fast self-play/evaluation, opponent diversity, runtime constraints, and robust heuristics.
4. Gold-level solutions are often hybrid systems: rules + search + learned policy + massive local evaluation.

AutoDS recipe:

- First make a valid agent and local simulator loop.
- Build a deterministic heuristic baseline.
- Create evaluation harness with many seeds/opponents.
- Track confidence intervals, not single-game wins.
- Profile runtime early.
- Add search/imitation/RL incrementally and compare against fixed benchmark agents.

Core lesson:

RL competition progress is evaluation-loop quality. Without a fast local simulator and opponent set, model complexity is mostly noise.

## Competition-Level Semantic Traces

These traces are concise. They identify the likely core upgrade path rather than reproducing every notebook.

### Audio

- `birdclef-2021`: starter identifier/data cleaning -> multi-year/pretraining data -> fold-based audio model -> secondary-label use. Core upgrade: leverage external/year-spanning bird audio priors and secondary labels.
- `birdclef-2022`: random/starter notebooks are weak; useful signal appears when moving to real training/inference and error analysis. Core upgrade: replace placeholder prediction with trained audio representation and robust inference.
- `birdclef-2023`: log-mel/TFRecord preprocessing -> baseline classifier -> pretrained bird embeddings/inference. Core upgrade: convert raw audio into reusable embeddings and aggregate predictions.
- `freesound-audio-tagging-2019`: random sample submission is useless; strong notebooks use 2D CNN/log-mel style pipelines and competition metric awareness. Core upgrade: spectrogram representation plus multilabel metric-aligned thresholding.
- `rfcx-species-audio-detection`: write-up/minimal/inverse-STFT/pretrained ResNet signals point to audio cleaning plus CNN representation. Core upgrade: robust denoising/spectrogram pipeline and CNN inference.

### CV

- `cassava-leaf-disease-classification`: simple Keras DenseNet -> EfficientNet/ResNeXt/ViT ensembles -> confusion-matrix/error analysis. Core upgrade: pretrained backbone selection plus model-level error diagnostics.
- `digit-recognizer`: simple CNN -> LeNet/scratch comparison -> public MNIST variants. Core upgrade: small-data augmentation/architecture discipline matters more than oversized models.
- `humpback-whale-identification`: generic image handling -> triplet/siamese metric learning -> ensemble. Core upgrade: treat as retrieval/identity matching, not plain classification.
- `imaterialist-challenge-fashion-2018`: selected notebooks are mostly downloaders, not modeling knowledge. Core lesson: exclude from training except as data-acquisition utility evidence.
- `tgs-salt-identification-challenge`: U-Net/data augmentation -> ResNet segmentation -> opponent/solution analysis. Core upgrade: segmentation-specific architecture and augmentation/postprocess.

### GenAI

- `drawing-with-llms`: simple LLM prompting -> Gemma/SDXL/vtracer pipelines -> winning SVG generation strategy. Core upgrade: combine generation model, format conversion, and output validation.
- `feedback-prize-2021`: BERT baseline -> sentence transformers/EDA -> LGB/train hybrids. Core upgrade: represent text spans/sentences well and blend neural embeddings with structured models.
- `feedback-prize-effectiveness`: BERT baseline -> DeBERTa train/inference -> cross-competition essay data. Core upgrade: stronger pretrained model and data reuse, but reject incomplete deletion notebook as gold evidence.
- `kaggle-llm-science-exam`: DeBERTa baseline -> Wikipedia RAG/Platypus -> retriever-reader sharding. Core upgrade: retrieval-augmented candidate evidence beats pure parametric guessing.
- `llms-you-cant-please-them-all`: simple prompt -> adversarial/diverse writing strategies -> final-place solution notes. Core upgrade: optimize against evaluator behavior with controlled prompt/style diversity.

### Medical

- `osic-pulmonary-fibrosis-progression`: environment/baseline -> dropout/loss tuning -> DNN + LGBM + NGBoost + ElasticNet. Core upgrade: multimodel tabular/clinical ensemble with uncertainty-aware regression.
- `prostate-cancer-grade-assessment`: starter/tile factory -> stain normalization/downsample tiling -> two-DL submission. Core upgrade: pathology-specific tiling and stain/resolution strategy.
- `rsna-breast-cancer-detection`: baseline inference/tabular -> KFold PyTorch Lightning -> rejection ensemble. Core upgrade: reduce false positives with ensemble/rejection and robust validation.
- `rsna-intracranial-hemorrhage-detection`: EDA/augmentation -> adjacent images/cropping -> final CNN. Core upgrade: exploit slice context and preprocessing before CNN inference.
- `siim-isic-melanoma-classification`: fastai baseline -> ResNet/XGBoost/OOf weighted ensemble. Core upgrade: combine image model outputs with tabular/meta features and calibration.

### NLP

- `commonlitreadabilityprize`: LSTM/baseline -> RoBERTa pretrained -> threshold/postprocessing/forked high-rank recipe. Core upgrade: transformer representation and fold-aware regression stability.
- `feedback-prize-english-language-learning`: baseline -> DeBERTa family inference/weight tuning. Core upgrade: transformer family ensemble and metric-aligned weighting; reject pip/install notebook as gold evidence.
- `google-quest-challenge`: spelling/EDA features -> end-to-end BERT -> postprocessing. Core upgrade: pretrained semantic model plus target-specific postprocessing.
- `nlp-getting-started`: weak notebook -> TF-IDF/BiLSTM/BERT comparison -> RoBERTa 5-fold. Core upgrade: systematic model comparison and fold ensemble.
- `quora-question-pairs`: weak/incomplete high-rank references; usable signal from XGB clean text and sentiment/feature notebooks. Core upgrade: duplicate-question features and clean validation, not the selected 1st bucket artifact.

### RL

- `google-football`: template bot -> GAIL/imitation -> episode scraping/visualization. Core upgrade: collect/evaluate trajectories and imitate stronger behavior.
- `halite`: explore/exploit heuristics -> aggressive starter -> PyTorch starter. Core upgrade: move from heuristic rules to learned policy only after simulator/evaluation tooling.
- `kore-2022`: understanding/evaluation notebooks -> model loading -> score trajectory analysis. Core upgrade: evaluation harness and model selection are more valuable than isolated agents.
- `lux-ai-2021`: rule-based agent -> imitate strategy -> fast IL in Keras. Core upgrade: imitation learning from stronger trajectories combined with baseline rules.
- `santa-2022`: greedy/hill-climbing/RL/MST bounds. Core upgrade: optimization/search framing and lower-bound reasoning, not generic RL alone.

### RecSys

- `elo-merchant-category-recommendation`: starter -> PCA/survival/AutoML variants. Core upgrade: temporal/cardholder feature representation and model diversity.
- `expedia-hotel-recommendations`: comments/MAP@K demos but weak gold artifact. Core upgrade: ranking metric and hotel candidate ordering; selected top bucket is not reliable.
- `h-and-m-personalized-fashion-recommendations`: recommender demo -> bronze ensemble -> Polars/GPU 1st-stage pipeline. Core upgrade: scalable candidate generation and fast feature pipeline.
- `otto-recommender-system`: easy pandas rec -> LightGBM/cudf -> matrix/co-visitation -> GBDT team solution. Core upgrade: co-visitation candidates plus GBDT ranker.
- `santander-product-recommendation`: EDA -> less-is-more scripts. Core upgrade: sparse product history features and conservative ranking logic.

### Tabular

- `competitive-data-science-predict-future-sales`: regression starter -> advanced regression -> automated ensembling. Core upgrade: lag/item/shop/month features and ensemble, but selected 20pct is weak.
- `house-prices-advanced-regression-techniques`: baseline -> EDA/feature engineering -> XGB/Cat/LGB blend -> two-stage model. Core upgrade: target transform, feature cleanup, and diverse model blend.
- `prudential-life-insurance-assessment`: R/xgboost/mlr snippets are incomplete. Core lesson: keep as low-confidence evidence only; do not train main skill from this competition.
- `spaceship-titanic`: baseline -> EDA/clean CV/CatBoost -> stronger tabular pipeline. Core upgrade: group/family/cabin-derived features and categorical handling.
- `titanic`: basic model -> EDA/ensemble -> high-rank ML notebook. Core upgrade: title/family/cabin/sex-class features plus conservative ensemble.

### Time-Series

- `amp-parkinsons-disease-progression-prediction`: simple UPDRS prediction -> 2-stage model/protein groups -> high-rank solution. Core upgrade: domain grouping and staged modeling.
- `g-research-crypto-forecasting`: random/candle analysis -> classic pipeline -> LGBM starter. Core upgrade: time-safe feature pipeline and GBM; random notebook is negative evidence.
- `tabular-playground-series-jan-2022`: XGB/basic -> holidays -> Prophet. Core upgrade: calendar/holiday structure and model matched to seasonality.
- `ubiquant-market-prediction`: N-BEATS beginner -> clustering/regime features -> aggregated features -> model ensemble. Core upgrade: regime-aware features and ensemble across market behavior.
- `web-traffic-time-series-forecasting`: SMAPE utility -> benchmarks -> out-of-box L2 regression. Core upgrade: metric-specific loss/evaluation and robust baseline; selected 70pct is utility-only.

## How AutoDS Should Use This V3

When starting a Kaggle competition:

1. Classify domain and metric.
2. Load this V3 domain recipe.
3. Pick 1-3 same-domain competition traces above.
4. Build baseline using the low/mid-rank path.
5. Plan the first serious upgrade using the 20pct/10pct lessons.
6. Only attempt gold push after validation, representation, and OOF/error analysis are in place.

Recommended `AUTODS.md` section:

```markdown
## Distilled Prior
- V3 domain:
- similar traces:
- baseline idea:
- first upgrade:
- gold-level hypothesis:
- validation guardrail:
```

