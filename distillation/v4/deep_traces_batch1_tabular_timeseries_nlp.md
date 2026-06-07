# V4 Deep Trace Batch 1: Tabular, Time-Series, NLP

Date: 2026-06-07

This batch is based on reading the generated evidence packs for 15 competitions:

- Tabular: 5 competitions
- Time-Series: 5 competitions
- NLP: 5 competitions

The evidence packs contain snippets from the five rank-bucket notebooks/code files for each competition. This batch focuses on the actual upgrade mechanics visible in notebook cells, not only file names or keyword counts.

## Tabular

### Titanic

Evidence read:

- 70pct: logistic regression / random forest / XGBoost, simple train/validation metrics, submission.
- 40pct: title-based age imputation and XGBoost.
- 20pct: sklearn pipeline, one-hot encoding, random forest, train/valid split, then `StratifiedKFold` + `cross_val_score`.
- 10pct: EDA-heavy pipeline, title medians, sex/age/fare/family analysis, `GridSearchCV`, `StratifiedKFold`, soft voting ensemble over linear/tree/GBM-style models.
- 1st: structured EDA/feature creation, processing/scaling, logistic regression, 5-fold CV, submission.

Core improvement:

The real upgrade is not simply "use XGBoost." The useful transition is:

1. valid model and submission
2. domain features such as title/family/cabin/fare/class/sex interactions
3. stratified validation
4. simple model family comparison
5. conservative ensemble or regularized linear model

AutoDS lesson:

For small tabular competitions, strong feature semantics and validation beat large model complexity. AutoDS should first extract human-meaningful features, then compare simple models under stable CV.

### House Prices

Evidence read:

- 70pct: random forest regression, label encoding, R2 validation, feature importance.
- 40pct: explicit competition metric awareness: RMSE on log-transformed SalePrice; EDA for missingness, outliers, correlations, `OverallQual`, `GrLivArea`, neighborhood, year effects; Ridge/Lasso/ElasticNet/RF/stacking/XGBoost imports.
- 20pct: ordinal encoding for quality columns, one-hot for nominal columns, Ridge pipeline, validation R2, Kaggle RMSLE table.
- 10pct: XGB/Cat/LGB blend notebook.
- 1st: two-stage model.

Core improvement:

The upgrade path is metric alignment plus target/domain preprocessing:

1. baseline RF on encoded raw columns
2. understand RMSLE/log target
3. treat missing values as semantic "not present" for garage/basement/pool
4. remove known outliers
5. encode ordinal quality correctly
6. blend regularized linear models and GBMs
7. two-stage or stacked modeling for residual structure

AutoDS lesson:

For skewed regression, AutoDS should inspect target transform and metric before modeling. House Prices teaches that missingness semantics, outlier policy, and ordinal encoding are core skills.

### Predict Future Sales

Evidence read:

- 70pct: monthly shop/item aggregation, lag features, linear/Ridge/Lasso/ElasticNet, validation MAE/R2, test lag merge.
- 40pct: explicit plan around monthly aggregates, lag features, rolling windows, item/shop/category enrichment, RMSE, target clipping 0-20.
- 20pct: weak but contains date_block/shop/item aggregation.
- 10pct: much richer matrix construction over month/shop/item, item first-sale month, label encoding, item/shop/category merges, group mean features, lag feature function, clipping lag features.
- 1st: automated weighted ensembling over prediction files.

Core improvement:

The decisive upgrade is transforming daily transaction logs into a supervised monthly panel:

1. aggregate by `date_block_num, shop_id, item_id`
2. build complete month-shop-item matrix
3. merge shop/item/category metadata
4. add lagged target and group-mean features
5. clip predictions/lag features to competition target range
6. ensemble only after good lagged panel models exist

AutoDS lesson:

For sales forecasting in tabular form, feature table construction is the model. AutoDS should spend most effort on panel construction and leakage-safe lagging before model tuning.

### Spaceship Titanic

Evidence read:

- 70pct: group/member count, CryoSleep analysis, deck/side/age/spend visual analysis, XGBoost pipeline.
- 40pct: very strong evidence: `StratifiedGroupKFold`, group-aware validation, CatBoost, OOF blending, fold-averaged probabilities, cabin structure, spending behavior, surname/group features, rule-based imputation audit.
- 20pct: high-accuracy ensemble, `StratifiedKFold`, RandomForest/ExtraTrees/XGB/LGBM/CatBoost/VotingClassifier, group size, CatBoost validation.
- 10pct: broad preprocessing, one-hot/ordinal scaling, stratified shuffle split, gradient boosting with early stopping.
- 1st: selected notebook is only a sample-submission-shaped output and is not useful.

Core improvement:

The best evidence is the 40pct/20pct notebooks, not the selected 1st bucket. The real upgrade is:

1. group/cabin/spend feature extraction
2. conservative rule-based imputation
3. group-aware validation to avoid passenger group leakage
4. CatBoost/GBM ensemble over categorical-heavy data
5. fold-averaged probability submission

AutoDS lesson:

AutoDS must not blindly trust rank bucket labels when the notebook content is bad. For Spaceship, the best distillation source is the group-aware CatBoost/OOF notebook.

### Prudential Life Insurance

Evidence read:

- 70pct: R XGBoost classifier with `merror`.
- 40pct/10pct/1st: same R `mlr3` notebook content; custom quadratic weighted kappa function, threshold cut optimization, custom measure.
- 20pct: Python XGBoost with quadratic weighted kappa wrapper, offset optimization, clipping/rounding predictions.

Core improvement:

This competition's useful lesson is metric-specific ordinal postprocessing:

1. train XGBoost/regression-like model
2. optimize thresholds/cuts for quadratic weighted kappa
3. clip and round predictions to valid ordinal response range
4. use custom metric wrappers inside model selection

AutoDS lesson:

For ordinal classification, metric-specific threshold optimization can matter more than model family. However this competition is low-confidence because several buckets duplicate the same notebook.

## Time-Series

### AMP Parkinson's Disease Progression

Evidence read:

- 70pct: minimal clinical data prediction loop.
- 40pct: SMAPE+1 function, clinical + supplemental data, visit-month medians, missing UPDRS fill, shifted future targets by `plus_month`, test iterator parsing.
- 20pct: explicitly reports CV/private score, two-stage model, shifted target creation, target-specific clipping, OOF SMAPE, proteins merge, patient forward-fill, XGB/stacking model.
- 10pct: protein NPX groups, patient-level protein feature fill, correlation/shift exploration, best constants.
- 1st: 4th-place solution; removes data before first blood work because not scored, creates visit-month booleans, standardizes 11 features, 5-fold MLP weights per target, fold inference.

Core improvement:

The upgrade is understanding the scoring mechanics and patient timeline:

1. parse prediction IDs into patient/visit/target/horizon
2. use visit-month medians as baseline
3. create shifted future target rows
4. add protein-derived patient features and forward fill
5. use OOF SMAPE+1 and target-specific clipping
6. filter to scored regime and train fold MLP/SVR-style models on standardized compact features

AutoDS lesson:

In clinical time-series, first decode the submission/scoring mechanics. The winning signal came from aligning the training rows with what Kaggle actually scores.

### G-Research Crypto Forecasting

Evidence read:

- 70pct: rolling beta-like features and LightGBM per regression target.
- 40pct: random Gaussian predictions; negative evidence.
- 20pct: candle pattern analysis, lag features, z-score metrics, asset-wise OHLC pattern exploration.
- 10pct: classic pipeline reading supplemental train, asset details, gap/previous features by `Asset_ID`.
- 1st: starter LGBM pipeline, one model per asset, basic OHLCV features, iterator submission; explicitly no validation.

Core improvement:

The reliable lesson is not "random" or "no validation"; it is:

1. maintain asset-specific models/features
2. create lag/gap/rolling features without future leakage
3. handle streaming iterator submission correctly
4. use LightGBM per asset as a strong baseline

AutoDS lesson:

For market competitions, AutoDS should treat public notebooks cautiously. A high-rank starter may teach valid inference mechanics, while robust improvement still requires time-safe validation.

### TPS Jan 2022

Evidence read:

- 70pct: random forest with one-hot encoding.
- 40pct: XGBRegressor after noting linear models fail on year-end spikes.
- 20pct: holiday kernel; 5th-place linear regression idea, holiday flags, Christmas-specific kernel, GDP merge, country/store/product factors.
- 10pct: PyCaret blend/finalize/predict.
- 1st: Prophet, GDP data, merge by date/country/store/product.

Core improvement:

The main upgrade is discovering the true generative structure:

1. baseline tree model on encoded categorical variables
2. observe systematic holiday/year-end spikes
3. encode country-specific holidays and Christmas effects
4. add GDP/country macro signal
5. use model class suited to seasonality/holiday components

AutoDS lesson:

When time-series is synthetic/seasonal, feature discovery can beat model complexity. AutoDS should inspect calendar effects before fitting generic GBMs.

### Ubiquant Market Prediction

Evidence read:

- 70pct: N-BEATS quick baseline; explicitly says no kfold, ensembling, LR tuning; suggests investment ID, KFold, ensembling.
- 40pct: XGBoost with clustering and separate models per cluster.
- 20pct: regime classification using aggregated features by `time_id`; KMeans/elbow to identify market regimes.
- 10pct: DNN model ensemble; investment ID input, 300 features, RMSE/correlation losses, GroupKFold/KFold references, loaded 10-fold weights, Pearson evaluation by `time_id`.
- 1st: full beginner-to-end notebook describing `time_id`, `investment_id`, target and 300 anonymized market features.

Core improvement:

The upgrade is from generic sequence modeling to market-regime/entity-aware modeling:

1. establish NN baseline
2. incorporate investment ID
3. cluster/regime classify by feature aggregates
4. train cluster/regime-specific models or ensembles
5. evaluate using correlation-like metric grouped by time
6. blend/fold DNNs with specialized losses

AutoDS lesson:

For obfuscated market tabular data, AutoDS should search for latent regimes and entity IDs early. Generic model scaling without regime validation is weak.

### Web Traffic Time Series

Evidence read:

- 70pct: SMAPE utility only; not a solution.
- 40pct: mean/median submissions, rolling-median outlier fill, SMAPE function.
- 20pct/10pct: top-50 silver benchmark, year shift of 364 days, weekly behavior via neighboring weekdays.
- 1st: transforms target for L2 regressors to better approximate SMAPE; validation split over last 64-day horizon; grid search transform hyperparameter.

Core improvement:

The upgrade is metric-aware target transformation plus seasonal baseline:

1. valid mean/median baseline
2. exploit weekly/yearly seasonality with 364-day shift
3. fill outliers with rolling medians
4. validate on the forecast horizon
5. transform target so ordinary L2 regressors better optimize SMAPE

AutoDS lesson:

For metric-specific forecasting, AutoDS should adapt the target/loss to the metric, not only change model family.

## NLP

### CommonLit Readability

Evidence read:

- 70pct: LSTM/torch dataset with train/valid split.
- 40pct: random forest baseline with validation MSE.
- 20pct: RoBERTa-base pretraining/MLM setup.
- 10pct: pretrained CLRP RoBERTa, KFold, deterministic seeds, tokenizer, weighted hidden-state pooling, scheduler, saved folds.
- 1st: large inference ensemble: transformer sequence models, bin postprocessing, Ridge over logits/preds, ensembler over fold predictions, multiple model directories.

Core improvement:

The upgrade is:

1. simple model with validation
2. use pretrained language model adapted to the domain
3. KFold and deterministic training
4. custom pooling/head
5. combine logits/predictions with ridge/bin postprocessing and model/fold ensembles

AutoDS lesson:

For text regression, the gold push is usually ensemble/postprocess over strong transformer embeddings, not a single transformer checkpoint.

### Feedback Prize English Language Learning

Evidence read:

- 70pct: TF/Keras text vectorization, KFold, RMSE, early stopping.
- 40pct: simple TextVectorization + Embedding + GlobalAveragePooling + RMSE.
- 20pct: classical models and `cross_validate`, Ridge/BayesianRidge/GradientBoosting exploration.
- 10pct: DeBERTa family inference with 10-fold configs, multiple model sizes, FGM variants, OOF pickles, RMSE/MCRMSE scoring, weight tuning.
- 1st: selected notebook is only pip/model loading style and should be rejected.

Core improvement:

The reliable upgrade is the 10pct notebook:

1. simple neural text baseline
2. classical baseline for sanity
3. transformer family models
4. 10-fold OOF inference
5. model family/size/adversarial variant weighting
6. MCRMSE-aligned scoring

AutoDS lesson:

When target is multi-output text regression, AutoDS should build OOF predictions per target and tune model weights against MCRMSE.

### Google Quest Challenge

Evidence basis:

The V4 evidence pack shows a progression from spelling/EDA features and general feature engineering to end-to-end BERT training and final postprocessing. The recurring signal is multi-target semantic scoring.

Core improvement:

1. basic text/question features
2. pretrained BERT semantic representation
3. end-to-end multi-target training
4. postprocess/calibrate outputs per target

AutoDS lesson:

For multi-target Q&A/text scoring, use pretrained semantic representation early, but reserve effort for per-target calibration and postprocessing.

### NLP Getting Started

Evidence read:

- 70pct: weak notebook.
- 40pct: full educational comparison of TF-IDF, BiLSTM with GloVe/attention, BERT; threshold optimization; validation F1/accuracy.
- 20pct: DistilBERT transfer learning, raw text, tokenizer, small learning rate.
- 10pct: RoBERTa 5-fold ensemble; explicitly states raw text > cleaned text for BERT, RoBERTa > DistilBERT, 5-fold ensemble improves public score.
- 1st: selected notebook is model-load inference only and not a strong learning source.

Core improvement:

The useful path is:

1. TF-IDF baseline
2. compare with sequence/embedding models
3. use raw text for transformers
4. fine-tune stronger pretrained model
5. stratified 5-fold probability ensemble
6. threshold tune for F1

AutoDS lesson:

For tweet-style classification, preserve raw signal such as hashtags for transformers; do not over-clean before testing.

### Quora Question Pairs

Evidence read:

- 70pct/1st: selected high-rank artifact only calls an external experiment script; not useful standalone evidence.
- 40pct: XGBoost starter with clean text, train/validation split, logloss, word-match/TF-IDF-share/cross-IDF/sum-IDF/diff-IDF features, feature importance.
- 20pct: sentiment features for question pairs, correlation check, cleaning/tokenization.
- 10pct: EDA/starter notebook but weak evidence.

Core improvement:

The actual lesson is feature construction for pairwise similarity:

1. clean/tokenize question pairs
2. compute word overlap and TF-IDF weighted overlap
3. add difference/sum/cross-IDF features
4. optionally add sentiment or other semantic side features
5. train XGBoost with logloss validation

AutoDS lesson:

Pairwise NLP tasks need pair-interaction features, not independent text classification alone.

## Batch 1 Cross-Domain Upgrade Factors

Across Tabular, Time-Series, and NLP, the recurring core upgrade factors are:

1. Metric-first design: RMSLE, SMAPE, MCRMSE, F1, logloss, and QWK each change modeling/postprocessing.
2. Validation-first reliability: KFold/Stratified/Group/time split appears before trustworthy improvements.
3. Representation upgrade: lag panel, holiday kernel, transformer embedding, pairwise text overlap, categorical/group features.
4. OOF/ensemble only after representation works.
5. Bad rank buckets must be rejected when content is sample-only, random, downloader-only, or external-script-only.

