# V4 Deep Trace Batch 2: CV, Medical, Audio

Date: 2026-06-07

This batch is based on reading the generated evidence packs for 15 competitions:

- CV: 5 competitions
- Medical: 5 competitions
- Audio: 5 competitions

The purpose is to extract the upgrade mechanics visible across rank buckets, and to mark low-quality or non-model notebooks when they should not be distilled as gold behavior.

## CV

### Cassava Leaf Disease Classification

Evidence read:

- 70pct: EfficientNetB5 / ImageDataGenerator baseline, validation split, early stopping, version notes about resolution and backbone changes.
- 40pct: DenseNet169 training with Keras ImageDataGenerator and validation.
- 20pct: inference ensemble using `timm` backbones: `tf_efficientnet_b4_ns`, `resnext50_32x4d`, `vit_base_patch16_384`; `StratifiedKFold`/`GroupKFold`, albumentations, TTA, weighted model ensemble.
- 10pct: confusion matrix and performance analysis, 5-fold inference CSV.
- 1st: ResNeXt50/timm training, `n_fold=5`, fixed validation split, seed, albumentations, fold loop, OOF saved.

Core improvement:

The useful path is pretrained backbone scaling plus validation discipline:

1. start with transfer learning on ImageNet-style backbones
2. lock a fold split and save OOF predictions
3. increase image resolution/backbone only after validation is stable
4. use albumentations and TTA
5. ensemble diverse CNN/ViT/timm backbones by OOF behavior
6. use confusion matrix to target class-specific failures

AutoDS lesson:

For plant/pathology image classification, AutoDS should not jump straight to an ensemble. It should first establish fold-stable OOF, then add backbone diversity, TTA, and class-error diagnostics.

### Digit Recognizer

Evidence read:

- 70pct: basic Keras CNN, data augmentation, train/validation curves, confusion matrix with specific error pairs such as 3/8 and 2/7.
- 40pct: PyTorch SinCNN, heavier augmentation, cosine annealing; validation is weak because full train is partly treated as valid.
- 20pct: scratch CNN vs LeNet, validation split, callbacks, ReduceLROnPlateau, train/validation curves.
- 10pct: CNN architecture and augmentation-oriented training.
- 1st: public MNIST-variant style evidence rather than a materially novel modeling pattern.

Core improvement:

Small image classification improves through controlled regularization:

1. compact CNN/LeNet baseline
2. holdout validation and learning-curve inspection
3. augmentation for shifts/rotation/scale
4. LR scheduling and callbacks
5. confusion-matrix-driven error review
6. optional public/external variants only if rules permit

AutoDS lesson:

For small clean vision tasks, the best gain is not architecture novelty; it is augmentation, schedule, validation, and error analysis.

### Humpback Whale Identification

Evidence read:

- 40pct: triplet-loss Siamese / pretrained ResNet retrieval setup, whale ranking utilities, MAP@5 validation, weighted score combining submissions.
- 20pct and 10pct: mostly ensemble CSV artifacts rather than full modeling notebooks.
- 1st: DenseNet/Siamese visualization and activation-map material.
- Other buckets: limited direct modeling content.

Core improvement:

This task is retrieval/identity matching rather than ordinary multiclass classification:

1. create image embeddings with a pretrained CNN
2. train metric learning / Siamese / triplet objective
3. rank candidates per test image
4. validate with MAP@5, not only accuracy
5. combine submissions only after ranking quality is measured

AutoDS lesson:

For open-set identity competitions, AutoDS must switch from classifier thinking to retrieval thinking: embedding quality, candidate ranking, and MAP@k validation are the core.

### iMaterialist Fashion 2018

Evidence read:

- All rank buckets are mostly data acquisition or downloader scripts: Google Drive helpers, multithreaded downloaders, and R/Python download logic.
- There is not enough modeling evidence to infer a gold solution path.

Core improvement:

No reliable modeling distillation should be taken from this competition in the current corpus. It can only contribute operational lessons:

1. large image datasets require resumable downloads
2. metadata/image integrity checks matter
3. local cache layout should be explicit

AutoDS lesson:

Mark this competition as low-value for skill distillation until real modeling notebooks are added. Do not let downloader-only notebooks bias the CV recipe.

### TGS Salt Identification Challenge

Evidence read:

- 70pct: Keras U-Net-style segmentation, train/valid split, flip augmentation, early stopping, checkpointing, ReduceLROnPlateau.
- 40pct: U-Net segmentation pipeline with callbacks.
- 20pct: ResNet segmentation, IoU metric, threshold tuning, augmentation, callbacks on `val_my_iou_metric`.
- 10pct: generator with shift/mirror augmentation.
- 1st: leaderboard/opponent analysis artifact rather than a model solution.

Core improvement:

Segmentation gains come from metric-aware U-Net training:

1. start with U-Net or encoder-decoder baseline
2. use masks/IoU validation rather than image-level proxies
3. augment with flips/shifts/mirrors
4. checkpoint on the competition metric
5. tune probability thresholds for mask submission
6. ignore leaderboard-only notebooks as model evidence

AutoDS lesson:

For segmentation, AutoDS should implement the full mask pipeline first: image/mask alignment, IoU metric, augmentations, threshold search, and visual mask checks.

## Medical

### OSIC Pulmonary Fibrosis Progression

Evidence read:

- 70pct/40pct/20pct/10pct: EfficientNet/image plus clinical feature patterns, folds, TTA, lungmask and Laplace metric references.
- 1st: DNN + LightGBM + NGBoost + ElasticNet, `GroupKFold`, custom Laplace log likelihood / LightGBM loss, NFOLD ensemble.

Core improvement:

The winning pattern is patient-level clinical uncertainty modeling:

1. use patient/group-aware splits
2. merge clinical tabular features with image-derived features when useful
3. optimize the Laplace-style competition metric, including confidence/uncertainty
4. ensemble heterogeneous model families: DNN, GBM, NGBoost, linear
5. keep patient trajectory consistency at inference

AutoDS lesson:

For medical progression tasks, leakage-safe patient grouping and metric-specific uncertainty are first-class requirements. AutoDS should not treat confidence as a formatting afterthought.

### Prostate Cancer Grade Assessment

Evidence read:

- 70pct: OpenSlide EDA, whole-slide image levels, rotations, color/stain differences.
- 40pct: MI-Zero demo with CTransPath + BioClinicalBERT/CLIP-style text prompts, pathology-specific embeddings.
- 20pct: stain normalization, downsample and tile inference.
- 10pct: tile factory.
- 1st: 2DL submission artifact.

Core improvement:

The useful medical-CV skill is whole-slide handling:

1. inspect WSI pyramid levels and metadata
2. tile slides at appropriate magnification
3. normalize stain/color variation
4. run tile-level inference
5. aggregate tile predictions to slide-level labels
6. prefer pathology-specific representations when available

AutoDS lesson:

For WSI competitions, the core is data geometry and aggregation. AutoDS should build a tiling/stain/slide-aggregation pipeline before model tuning.

### RSNA Breast Cancer Detection

Evidence read:

- Lower buckets include baseline inference and EDA.
- Mid/high buckets show PyTorch Lightning KFold-style image training, DICOM/PNG handling, tabular metadata and logistic regression side models.
- Stronger evidence includes rejection/ensemble behavior and OOF-based combination.

Core improvement:

Breast screening tasks require reducing false positives while preserving sensitivity:

1. parse image metadata and patient/exam grouping
2. train fold-stable image models
3. use tabular/metadata side signals
4. calibrate or re-rank suspicious cases
5. ensemble image and metadata predictions using OOF validation

AutoDS lesson:

For screening competitions, AutoDS should include calibration and false-positive analysis in the loop, not only raw AUC/logloss optimization.

### RSNA Intracranial Hemorrhage Detection

Evidence read:

- 70pct: EDA and albumentations.
- 40pct: preprocessing with adjacent images and cropping.
- 20pct/10pct/1st: final CNN-style kernels; DICOM HU windowing/masking, weighted log loss matching the competition, KFold, ResNet/VGG/Inception families, checkpointed predictions, weighted epoch averaging.

Core improvement:

The upgrade path is medical image preprocessing plus metric alignment:

1. read DICOM correctly
2. apply HU/windowing/masking appropriate to CT
3. include adjacent slice context
4. train CNN backbones with KFold
5. optimize weighted log loss
6. average checkpoints/epochs/folds

AutoDS lesson:

For DICOM competitions, AutoDS must create a modality-specific preprocessing checklist. Generic PNG loading is a weak baseline.

### SIIM-ISIC Melanoma Classification

Evidence read:

- 70pct: fastai2 ResNet34, `StratifiedKFold`, TTA, ROC AUC.
- 40pct: ResNet50 plus XGBoost/meta features, LabelEncoder, validation split.
- 20pct/1st: min/max ensemble artifacts.
- 10pct: many OOF/submission files, ensemble weight optimization using ROC AUC.

Core improvement:

The real signal is image plus metadata plus OOF-weighted ensemble:

1. train stratified image folds
2. include patient/lesion metadata where allowed
3. generate OOF predictions for every component
4. optimize ensemble weights on ROC AUC
5. use TTA cautiously and validate calibration

AutoDS lesson:

For medical image classification, metadata can be as important as the image model. AutoDS should always test an image-only, metadata-only, and combined model.

## Audio

### BirdCLEF 2021

Evidence read:

- 70pct/1st: ResNest50 inference; 1st uses secondary-label checkpoint.
- 40pct/20pct: multi-year BirdCLEF dataset cleaning and combination across 2020-2025 style resources.
- 10pct: strong pretraining notebook: `tf.data` audio pipeline, mel spectrogram GPU/TPU path, TimeFreqMask/Normalization, CutMix/MixUp, EfficientNet FSR, pretraining on 2020/2021/2022/Xeno-Canto, StratifiedKFold.

Core improvement:

Bird audio competitions are pretraining/data competitions:

1. build efficient log-mel/spectrogram pipeline
2. include secondary labels when rules permit
3. pretrain on related bird datasets and Xeno-Canto
4. use TimeFreqMask, MixUp/CutMix and normalization
5. train fold-stable EfficientNet/ResNest-style spectrogram classifiers

AutoDS lesson:

For BirdCLEF, AutoDS should prioritize data coverage and label semantics before architecture tuning.

### BirdCLEF 2022

Evidence read:

- 40pct: random guess; negative evidence.
- 20pct: PyTorch/TPU pipeline with torchlibrosa, timm, `StratifiedKFold`/`GroupKFold`, EfficientNet, CutMix/MixUp, metric `epoch_f1_at_03`.
- 10pct: Hawaii-focused training, 32-fold stratified setup, noise/pitch/time audio augmentations, torchlibrosa, EfficientNet/timm.
- 1st: BirdNET inference.

Core improvement:

The upgrade is from placeholder submissions to thresholded audio classification:

1. convert clips to log-mel features
2. train timm/EfficientNet spectrogram models
3. use stratified/group folds
4. optimize thresholded F1, not only BCE
5. add time/pitch/noise augmentations
6. compare against strong external BirdNET-style inference if allowed

AutoDS lesson:

Audio competitions need metric-threshold search. AutoDS should report threshold, F1 curve, and per-class recall.

### BirdCLEF 2023

Evidence read:

- Lower/mid buckets show log-mel TFRecord and baseline classifier patterns.
- Stronger evidence includes Google bird embeddings and inference-only pipelines.
- Several notebooks are more about inference packaging than original training.

Core improvement:

The useful pattern is embedding/pretraining plus robust inference:

1. prepare log-mel or embedding features
2. reuse strong bird-audio pretrained representations when allowed
3. segment long soundscapes into scored windows
4. aggregate window predictions per species/time row
5. validate thresholding with the competition metric

AutoDS lesson:

For recent BirdCLEF tasks, AutoDS should treat pretrained embeddings as a first baseline, then decide whether custom training adds enough value.

### Freesound Audio Tagging 2019

Evidence read:

- Stronger buckets show 2D CNN/log-mel/fastai-style training and lwlrap metric awareness.
- 10pct is a random sample submission and should be rejected as solution evidence.
- Other notebooks contain simple CNN baselines.

Core improvement:

Multi-label audio tagging needs metric-aware spectrogram classification:

1. create fixed log-mel spectrogram inputs
2. train 2D CNN or pretrained image backbone over spectrograms
3. handle noisy/curated label split
4. optimize lwlrap locally
5. tune prediction thresholds and blend only validated models

AutoDS lesson:

For multi-label audio, AutoDS must implement lwlrap and noisy-label handling before trusting leaderboard scores.

### RFCX Species Audio Detection

Evidence read:

- 40pct: ResNet50 TPU, `MultilabelStratifiedKFold`, SpecAugment, MixUp, multilabel folds and ResNet head.
- 20pct: write-up: crop mel spectrograms around labels, make validation mirror inference by slicing/stacking and max predictions, attention head, lwlrap.
- 10pct: inverse STFT / learnable time-frequency representation.
- 1st: `rfcx-minimal`, ResNest50, PyTorch Lightning, weighted sampler, BCEWithLogitsLoss, lwlrap rank metric, click-noise augmentation, validation should match test, split clips into pieces and max/aggregate.

Core improvement:

The decisive factor is validation-inference parity:

1. crop/segment audio according to label timing
2. train spectrogram CNN/ResNest with multilabel folds
3. use BCE plus lwlrap validation
4. use SpecAugment/MixUp/noise augmentation
5. at inference, split long clips exactly like validation and max/aggregate windows

AutoDS lesson:

For detection-like audio tasks, AutoDS should first align train labels, validation slicing, and test aggregation. Architecture comes second.

