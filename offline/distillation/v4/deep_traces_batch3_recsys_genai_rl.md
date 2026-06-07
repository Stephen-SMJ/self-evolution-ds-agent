# V4 Deep Trace Batch 3: RecSys, GenAI, RL

Date: 2026-06-07

This batch is based on reading the generated evidence packs for 15 competitions:

- RecSys: 5 competitions
- GenAI: 5 competitions
- RL / games / optimization: 5 competitions

These domains contain many notebooks that are not normal train-predict pipelines. The distillation therefore separates modeling signal from operational/evaluation artifacts.

## RecSys

### Elo Merchant Category Recommendation

Evidence read:

- 70pct: EDA and transaction feature engineering.
- 40pct: historical transaction aggregations by `card_id`, LightGBM 5-fold OOF/RMSE, PCA/noise analysis.
- 20pct: survival/time-to-event educational notebook; weak direct evidence for Elo.
- 10pct: deep/AutoML-style modeling on prebuilt features.
- 1st: ETL around historical/new merchant transactions: `month_lag`, authorized flags, purchase amount aggregates, horizontal/vertical splits.

Core improvement:

The key skill is transaction-to-customer feature aggregation:

1. aggregate historical and new merchant transactions separately
2. group by `card_id`
3. encode `month_lag`, authorized flag, category, merchant and amount statistics
4. build OOF LightGBM/RMSE validation
5. add advanced models only after ETL is stable

AutoDS lesson:

For transaction recommendation/regression, data aggregation is the model. AutoDS should audit aggregation leakage and cardinality before tuning.

### Expedia Hotel Recommendations

Evidence read:

- 70pct: scoring dictionaries for `hotel_cluster` using user/location/destination/country/market keys and recency/booking weights.
- 40pct/10pct: MAP@5 metric demos.
- 20pct: `valium-1.py` style count dictionaries, popularity, date/time weights, `nlargest` ranking.
- 1st: unrelated/weak artifact; not solution evidence.

Core improvement:

This is a candidate ranking heuristic problem:

1. construct multiple key-specific popularity dictionaries
2. weight bookings higher than clicks
3. add recency/time decay
4. backfill with global popularity
5. optimize MAP@5 ordering

AutoDS lesson:

For sparse large-scale recommendation, AutoDS should first build a strong heuristic ranker and metric implementation. Full ML can come after candidate coverage is high.

### H&M Personalized Fashion Recommendations

Evidence read:

- 10pct: ensemble of strong notebooks.
- 1st: stage demo using Polars/GPU; validation cutoff and label window, history/global/age/category top candidates, candidate source weights/rank/count/recency, MAP and recall metrics; explicit note that stage-1 candidate set should feed a reranker.
- Lower buckets are weaker or narrower.

Core improvement:

H&M teaches two-stage recommender design:

1. define time-window validation
2. generate candidates from user history, global trends, age/category affinities and recency
3. measure candidate recall before ranking
4. create source/rank/count/recency features
5. train or ensemble a reranker after recall is sufficient

AutoDS lesson:

For large recommenders, AutoDS should optimize candidate recall first. A ranker cannot recover items that were never generated.

### OTTO Recommender System

Evidence read:

- 70pct: conceptual recommender overview; weak direct implementation.
- 40pct: LightGBM LambdaRank, group by session, train/validation query groups, `lambdarank` objective.
- 20pct: simple next-click candidate-list reasoning.
- 10pct: co-visitation matrices built from train only, chunked merge, click/time top candidates.
- 1st: 3rd-place final submission, ensemble of three XGBoost rerankers and blended submissions.

Core improvement:

The winning structure is candidate generation plus session reranking:

1. create co-visitation matrices without test leakage
2. generate session candidates for clicks/carts/orders
3. group training rows by session/query
4. train LGBM/XGB rankers
5. ensemble specialized rerankers by action type

AutoDS lesson:

For session recsys, AutoDS should build co-visitation candidates and query-grouped ranker data before any neural recommendation model.

### Santander Product Recommendation

Evidence read:

- 70pct: EDA.
- 40pct/20pct/1st: XGBoost scripts using product list targets, previous/lag product features, multiclass/multilabel ranking, sorted predictions.

Core improvement:

This task rewards modeling product changes, not existing holdings:

1. create lagged customer product holdings
2. predict newly added products
3. use customer demographic/account features
4. train XGBoost-style rankers/classifiers
5. suppress products already owned
6. output top-N ranked recommendations

AutoDS lesson:

For next-product recommendation, AutoDS must model deltas over time. Predicting current ownership is leakage-prone and not the objective.

## GenAI

### Drawing with LLMs

Evidence read:

- 70pct: Gemma/Gemma-style model generating SVG, helper validation with `svg_constraints`, path validation.
- 40pct: Stable Diffusion to SVG through `vtracer`, validates `href` and path constraints.
- 20pct: Gemma 2B direct SVG prompting.
- 10pct: SDXL Turbo, metric package, VQA/OCR/aesthetic scoring, SVG resize and feature extraction.
- 1st: Segmind SSD/SDXL + TAESD/CLIP/aesthetic model, dual GPU, deterministic seeds, FFT filtering, chooses best candidates by aesthetic score and penalties.

Core improvement:

The jump is from direct prompting to candidate generation plus proxy judging:

1. generate multiple SVG/image candidates
2. validate hard SVG constraints
3. use diffusion or vision models when direct SVG text is weak
4. convert/rasterize/vectorize where needed
5. score with VQA/OCR/aesthetic/proxy metrics
6. select the best compliant candidate

AutoDS lesson:

For generative competitions, AutoDS should build a generate-validate-score-select loop. Prompting alone is an entry-level baseline.

### Feedback Prize 2021

Evidence read:

- 70pct: BERT tutorial baseline.
- 40pct: corrected train CSV/data cleanup evidence.
- 20pct: QA-style inference artifact.
- 10pct: sentence-transformers/EDA style evidence.
- 1st: Feedback LGB train artifact.

Core improvement:

The available notebooks imply a hybrid span/structure approach:

1. clean labels and corrected training data
2. use transformer embeddings for discourse/text spans
3. add sentence/paragraph position and length features
4. train span classifiers or GBM over structured features
5. validate by competition-specific span overlap/F1 behavior

AutoDS lesson:

For discourse segmentation, AutoDS should not only fine-tune BERT. It needs label-quality checks and span reconstruction rules.

### Feedback Prize Effectiveness

Evidence read:

- Lower buckets show BERT-style baselines.
- Mid/high buckets include DeBERTa training, external/cross-competition data references, essay features and detectAI-like resources.
- The 1st bucket is not a complete high-quality modeling artifact and should be treated cautiously.

Core improvement:

Effectiveness classification benefits from strong pretrained encoders plus essay/context features:

1. start with DeBERTa-style encoder
2. preserve discourse text and essay context
3. add essay/discourse metadata features
4. use folds and OOF predictions
5. cautiously add related competition data if rules permit

AutoDS lesson:

For educational NLP, context around the span matters. AutoDS should test isolated discourse text versus discourse-plus-essay context.

### Kaggle LLM Science Exam

Evidence read:

- 70pct: DeBERTa-large inference.
- 40pct: Platypus2 70B plus Wikipedia RAG.
- 20pct: retriever-reader with BM25/Lucene, `rank_bm25`, chunk merging, sharded prompts, first-token scoring.
- 10pct: LongFormer/TFIDF retrieval, sentence-transformers, full paragraph context, multiple-choice classifier.
- 1st: wiki URL/category resource preparation.

Core improvement:

Retrieval quality dominates model choice:

1. build a corpus from allowed Wikipedia/resources
2. retrieve with BM25/TFIDF/dense embeddings
3. chunk and merge context around candidate passages
4. score options with reader/LLM/classifier
5. ensemble retrievers or rerankers
6. audit retrieval misses as the first error category

AutoDS lesson:

For knowledge QA competitions, AutoDS should inspect retrieval recall before changing the answer model. Bad context makes a strong LLM weak.

### LLMs - You Can't Please Them All

Evidence read:

- 70pct: simple Mistral prompt with sentence-transformer checks.
- 40pct: adversarial/polarizing/dynamic prompt, judge-manipulation and numerical-response behavior.
- 20pct: Phi/Gemma prompt mashups, suffix strategies, add-it-up/emergency prompts, postprocessing, similarity.
- 10pct: simple adjustments, longer text, prompt likelihood, fewer choices.
- 1st: selected batch/merge and similarity-score artifact.

Core improvement:

The high-level skill is evaluator-aware prompt search:

1. generate diverse prompt families
2. measure similarity/diversity/proxy judge behavior
3. add adversarial or polarizing suffixes when aligned with rules
4. postprocess outputs for format and target behavior
5. batch-select and merge best candidates

AutoDS lesson:

For LLM judge competitions, AutoDS should treat prompts as search candidates with proxy metrics, not as one final instruction.

## RL / Games / Optimization

### Google Football

Evidence read:

- 70pct: template bot.
- 40pct: GAIL/PyTorch RL, PPO-like policy/value/discriminator, expert states/actions, submission tarball.
- 20pct: tunable heuristic baseline bot.
- 10pct: visualization.
- 1st: episode scraper that filters top submissions by final rating.

Core improvement:

The useful pattern is imitation plus evaluation data:

1. create valid submission package
2. implement heuristic baseline
3. scrape/collect high-rated episodes if allowed
4. train imitation/GAIL policy from expert traces
5. evaluate policies against fixed opponents

AutoDS lesson:

For game competitions, AutoDS must build local evaluation and replay ingestion. Training without opponents/replays is blind.

### Halite

Evidence read:

- 70pct: explore/exploit heuristic.
- 40pct: leaderboard result visualization and cargo/halite curves.
- 20pct: aggressive attack bot.
- 10pct: PyTorch starter with input stack, action rankings and valid action matrix.
- 1st: win/loss counting by team and episode score filtering.

Core improvement:

The progression is rules to learned policy with action validity:

1. implement valid rule-based agent
2. instrument local match metrics
3. analyze episode outcomes and resource curves
4. add attack/defense heuristics
5. train policy models only with valid-action masking

AutoDS lesson:

For grid-resource games, action validity and evaluation instrumentation are more important than network depth.

### Kore 2022

Evidence read:

- 70pct: competition understanding and `attacker.py`, `env.run`.
- 40pct: visualization/integration.
- 20pct: loading another model.
- 10pct: win-rate evaluation/multiprocessing-style evidence.
- 1st: score trajectory analysis.

Core improvement:

Kore evidence is mostly evaluation and integration:

1. run the official environment locally
2. package bots correctly
3. test against fixed opponents
4. parallelize win-rate evaluation
5. select models by stable match results, not one replay

AutoDS lesson:

For RL/game tasks, AutoDS should first automate tournaments and score tracking. Model changes are second-order until evaluation is stable.

### Lux AI 2021

Evidence read:

- 70pct: delux-ai-zero style baseline.
- 40pct: imitation strategy with map feature tensor, output action map and custom loss.
- 20pct: rule-based city/resource cluster logic, expansion positions, attached city tiles, day/night and gate protection.
- 10pct: fast imitation learning in Keras, map coordinate shifts, Sequence pipeline, validation accuracy, load model, sorted policy actions.
- 1st: select agents/download matches; leaderboard episode scraping and high-score filtering above 1500.

Core improvement:

The strongest pattern is rules plus imitation from strong agents:

1. start from robust rule-based resource/city/day-night logic
2. scrape or collect strong-agent replays if allowed
3. convert maps into feature tensors
4. train imitation policy over valid actions
5. evaluate in local tournaments
6. keep rule fallbacks for invalid or low-confidence actions

AutoDS lesson:

For complex environment games, AutoDS should combine symbolic rules and imitation. Pure RL from scratch is usually too slow for competition iteration.

### Santa 2022

Evidence read:

- 70pct: problem framing around robotic arm path and color cost.
- 40pct: DDQN/LSTM RL attempt with high runtime.
- 20pct: baseline path script.
- 10pct: greedy/hill-climbing/random exploration with references.
- 1st: lower-bound / MST-style analysis.

Core improvement:

This competition behaves more like optimization/search than classic RL:

1. formalize objective and constraints
2. build a valid baseline path
3. use greedy, local search and hill-climbing
4. compute lower bounds to measure gap
5. only use RL if it beats search under runtime constraints

AutoDS lesson:

For optimization competitions, AutoDS should classify the problem type before choosing RL. Search and lower bounds are often the gold path.

