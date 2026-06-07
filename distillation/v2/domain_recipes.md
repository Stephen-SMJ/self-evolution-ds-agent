# V2 Distilled Domain Recipes

This reference is generated from `distillation/v2/domain_recipes.json` and `competition_evolution_traces.json`.

Use it as the default retrieval entry for offline self-evolution. The JSON files remain the source of truth.

## Audio

Quality counts: `{'strong': 17, 'usable': 6, 'weak': 1, 'reject': 1}`

Top extracted patterns:
- `features:audio_features`: 21
- `postprocess:submission_format`: 15
- `validation:metric_eval`: 15
- `features:image_features`: 13
- `features:aggregations`: 13
- `models:cnn`: 12
- `training:scheduler`: 10
- `training:augmentation`: 10
- `validation:kfold`: 8
- `postprocess:ranking`: 7

Distilled lessons:
- Start by matching validation to the competition split risk.
- Use pretrained CNN/vision backbones before custom architectures.
- Add controlled ensembles after individual models are validated.
- Prioritize group/entity/time aggregations as high-ROI features.
- Standardize waveform-to-spectrogram preprocessing and clip aggregation.

## CV

Quality counts: `{'strong': 15, 'usable': 3, 'weak': 2, 'reject': 5}`

Top extracted patterns:
- `models:cnn`: 15
- `postprocess:submission_format`: 14
- `features:image_features`: 13
- `postprocess:thresholding`: 13
- `training:augmentation`: 12
- `postprocess:ensemble`: 10
- `validation:train_valid_split`: 10
- `validation:metric_eval`: 9
- `training:scheduler`: 9
- `features:time_features`: 8

Distilled lessons:
- Start by matching validation to the competition split risk.
- Use pretrained CNN/vision backbones before custom architectures.
- Add controlled ensembles after individual models are validated.

## GenAI

Quality counts: `{'strong': 16, 'usable': 7, 'reject': 1, 'weak': 1}`

Top extracted patterns:
- `models:transformer`: 21
- `validation:metric_eval`: 16
- `features:text_features`: 15
- `postprocess:submission_format`: 13
- `features:time_features`: 10
- `postprocess:ensemble`: 10
- `training:augmentation`: 9
- `postprocess:thresholding`: 8
- `features:missing`: 7
- `training:scheduler`: 6

Distilled lessons:
- Start by matching validation to the competition split risk.
- Use pretrained transformer representations and keep a simple baseline for sanity.
- Add controlled ensembles after individual models are validated.
- Prioritize group/entity/time aggregations as high-ROI features.

## Medical

Quality counts: `{'strong': 17, 'usable': 8}`

Top extracted patterns:
- `features:image_features`: 18
- `postprocess:submission_format`: 16
- `models:cnn`: 15
- `postprocess:thresholding`: 15
- `validation:metric_eval`: 13
- `postprocess:ensemble`: 12
- `training:augmentation`: 11
- `validation:kfold`: 10
- `features:aggregations`: 10
- `training:scheduler`: 8

Distilled lessons:
- Start by matching validation to the competition split risk.
- Use pretrained CNN/vision backbones before custom architectures.
- Add controlled ensembles after individual models are validated.
- Prioritize group/entity/time aggregations as high-ROI features.

## NLP

Quality counts: `{'strong': 15, 'usable': 4, 'reject': 4, 'weak': 2}`

Top extracted patterns:
- `features:text_features`: 16
- `validation:metric_eval`: 15
- `postprocess:submission_format`: 15
- `postprocess:ensemble`: 13
- `models:transformer`: 11
- `postprocess:thresholding`: 11
- `validation:train_valid_split`: 8
- `validation:kfold`: 6
- `features:time_features`: 6
- `features:aggregations`: 6

Distilled lessons:
- Start by matching validation to the competition split risk.
- Use pretrained transformer representations and keep a simple baseline for sanity.
- Add controlled ensembles after individual models are validated.
- Prioritize group/entity/time aggregations as high-ROI features.

## RL

Quality counts: `{'strong': 10, 'usable': 13, 'weak': 2}`

Top extracted patterns:
- `features:time_features`: 19
- `models:rl_search`: 16
- `validation:metric_eval`: 12
- `training:augmentation`: 10
- `postprocess:thresholding`: 8
- `postprocess:ensemble`: 8
- `models:cnn`: 4
- `postprocess:submission_format`: 4
- `features:categorical`: 3
- `features:recsys_features`: 3

Distilled lessons:
- Start by matching validation to the competition split risk.
- Use pretrained CNN/vision backbones before custom architectures.
- Add controlled ensembles after individual models are validated.
- Prioritize group/entity/time aggregations as high-ROI features.
- Separate candidate generation from ranking and measure candidate recall.
- Build a local simulator/evaluation harness before optimizing policy logic.

## RecSys

Quality counts: `{'strong': 10, 'usable': 11, 'reject': 1, 'weak': 3}`

Top extracted patterns:
- `features:time_features`: 15
- `features:aggregations`: 11
- `validation:metric_eval`: 10
- `postprocess:submission_format`: 9
- `postprocess:ensemble`: 9
- `features:missing`: 8
- `features:categorical`: 7
- `postprocess:ranking`: 7
- `postprocess:thresholding`: 6
- `features:recsys_features`: 6

Distilled lessons:
- Start by matching validation to the competition split risk.
- Use GBM models as the first serious model family when structured features exist.
- Add controlled ensembles after individual models are validated.
- Prioritize group/entity/time aggregations as high-ROI features.
- Separate candidate generation from ranking and measure candidate recall.

## Tabular

Quality counts: `{'usable': 4, 'strong': 17, 'weak': 4}`

Top extracted patterns:
- `features:categorical`: 20
- `features:missing`: 19
- `validation:metric_eval`: 18
- `models:gbm`: 17
- `postprocess:submission_format`: 14
- `postprocess:ensemble`: 13
- `features:aggregations`: 13
- `postprocess:thresholding`: 10
- `models:linear`: 10
- `validation:train_valid_split`: 9

Distilled lessons:
- Start by matching validation to the competition split risk.
- Use GBM models as the first serious model family when structured features exist.
- Add controlled ensembles after individual models are validated.
- Prioritize group/entity/time aggregations as high-ROI features.

## Time-Series

Quality counts: `{'strong': 16, 'usable': 7, 'weak': 1, 'reject': 1}`

Top extracted patterns:
- `features:missing`: 16
- `features:time_features`: 15
- `features:aggregations`: 15
- `postprocess:submission_format`: 14
- `postprocess:ensemble`: 13
- `validation:metric_eval`: 11
- `models:gbm`: 8
- `postprocess:thresholding`: 6
- `models:linear`: 4
- `features:audio_features`: 4

Distilled lessons:
- Start by matching validation to the competition split risk.
- Use GBM models as the first serious model family when structured features exist.
- Add controlled ensembles after individual models are validated.
- Prioritize group/entity/time aggregations as high-ROI features.
- Standardize waveform-to-spectrogram preprocessing and clip aggregation.
