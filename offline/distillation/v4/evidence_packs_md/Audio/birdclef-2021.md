# Audio / birdclef-2021

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 398 quality strong score 67

File: `Audio/birdclef-2021/rank398_70pct/clean-fast-simple-bird-identifier-20210502.ipynb`
Cells: 33 total, 24 code, 9 markdown

### Cell 2 markdown

```
* The inference is based on these [resnest50 weights](https://www.kaggle.com/kneroma/kkiller-birdclef-models-public). Please, don't forget upvoting the dataset to make it more visible for others
* The inference pipeline is optimized as much as I can in order to reduce execution time
```

### Cell 7 code

```
import torch
from torch import nn
from  torch.utils.data import Dataset, DataLoader
from resnest.torch import resnest50
```

### Cell 9 code

```
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_SUB_PATH = "../input/birdclef-2021/sample_submission.csv"
    # SAMPLE_SUB_PATH = "../input/birdclef-2021/sample_submission.csv"
```

### Cell 12 code

```
        V = np.clip(X, _min, _max)
```

### Cell 13 code

```
        image = np.stack([image, image, image])
        images = np.stack(images)
```

### Cell 15 code

```
df_train = pd.read_csv("../input/birdclef-2021/train_metadata.csv")
```

### Cell 18 code

```
    dummy_device = torch.device("cpu")
    d = torch.load(checkpoint_path, map_location=dummy_device)
```

### Cell 19 code

```
    Path("../input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth"),
```

### Cell 20 code

```
@torch.no_grad()
    o = (-out).argsort(1)
```

### Cell 22 code

```
    with torch.no_grad():
            xb = torch.from_numpy(test_data[idx]).to(DEVICE)
                o = torch.sigmoid(o)
```

### Cell 25 code

```
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH, usecols=["row_id"])
        sub = sample_sub.merge(sub, on="row_id", how="left")
```

### Cell 27 code

```
sub.to_csv("submission.csv", index=False)
```

## 40pct rank 295 quality usable score 43

File: `Audio/birdclef-2021/rank295_40pct/birdclef-21-22-23-data-cleaning.ipynb`
Cells: 44 total, 29 code, 15 markdown

### Cell 0 markdown

```
- A csv file with the 2023 dataset reduced to 'primary_label', 'secondary_labels', 'type', 'filename','filepath'.  File path is relative to the kaggle input folder.
All the file names in both csv files are now unique, and in the same format. I have removed 6,686 cases where there was overlap between competitions, plus a remaining 7 cases where the same file name existed in more than one class folder.
You could link the datasets from all three years, and use these two CSV files to create your training and validation splits, for pretraining on all the data and finetuning on the 2023 dataset, all in the same format.
```

### Cell 2 code

```
data_folder = Path('/kaggle/input')  #modify to suit
out_folder = Path('/kaggle/working')
train_21 = data_folder / 'birdclef-2021' / 'train_short_audio'
train_22 =  data_folder / 'birdclef-2022' / 'train_audio'
train_23 = data_folder / 'birdclef-2023' / 'train_audio'
train_21_csv = data_folder / 'birdclef-2021' / 'train_metadata.csv'
train_22_csv = data_folder / 'birdclef-2022' / 'train_metadata.csv'
train_23_csv = data_folder / 'birdclef-2023' / 'train_metadata.csv'
out_csv_all_pth = out_folder / 'train_21_22_23.csv'
out_csv_23_pth = out_folder / 'train_23.csv'
```

### Cell 4 code

```
unique_names_w_folder = set([str(k.parent.name) + '/' + str(k.name) for (k,v) in all_sfiles.items()])
print(f'Unique file names including folder name: {len(unique_names_w_folder)}')
```

### Cell 32 code

```
print(f'{all_sf_len - len(all_sfiles)} items were removed because they were the same file name in different class folders')
```

### Cell 34 code

```
df_23 = pd.read_csv(train_23_csv,  usecols=in_csv_fields, dtype=csv_datatypes, index_col=None)
df_22 = pd.read_csv(train_22_csv,  usecols=in_csv_fields, dtype=csv_datatypes, index_col=None)
df_21 = pd.read_csv(train_21_csv,  usecols=in_csv_fields, dtype=csv_datatypes, index_col=None)
```

### Cell 43 code

```
df_23.to_csv(out_csv_23_pth, index=False)
all_dfs.to_csv(out_csv_all_pth, index=False)
```

## 20pct rank 130 quality usable score 43

File: `Audio/birdclef-2021/rank130_20pct/birdclef-2020-2025-all-training-extra-npy-dataset.ipynb`
Cells: 21 total, 19 code, 2 markdown

### Cell 0 markdown

```
# Credits goes to [@cpmpml](https://www.kaggle.com/cpmpml), its a copy of his notebook for 2025 [BirdCLEF 2024 3rd solution](https://github.com/jfpuget/birdclef-2024/)
```

### Cell 5 code

```
train = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
```

### Cell 7 code

```
train_20 = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')
```

### Cell 9 code

```
train_21 = pd.read_csv('/kaggle/input/birdclef-2021/train_metadata.csv')
```

### Cell 10 code

```
train_22 = pd.read_csv('/kaggle/input/birdclef-2022/train_metadata.csv')
```

### Cell 11 code

```
train_23 = pd.read_csv('/kaggle/input/birdclef-2023/train_metadata.csv')
```

### Cell 12 code

```
train_24 = pd.read_csv('/kaggle/input/birdclef-2024/train_metadata.csv')
```

### Cell 15 code

```
all_train['rank'] = all_train.groupby('primary_label').source.rank(method='first', ascending=False)
```

### Cell 18 code

```
all_train.to_csv('/datasets/all_train.csv', index=False)
```

## 10pct rank 34 quality strong score 73

File: `Audio/birdclef-2021/rank34_10pct/birdclef23-pretraining-is-all-you-need-train.ipynb`
Cells: 109 total, 50 code, 59 markdown

### Cell 1 markdown

```
In this notebook, we will explore how to identify bird calls using TensorFlow. Specifically, this notebook will cover:
* How to use tf.data for audio processing tasks and reading .ogg files in TensorFlow
* How to extract spectrogram features from raw audio on TPU/GPU, which reduces CPU bottleneck significantly, speeding up the process by ~$4 \times$ on **P100 GPU** compared to the [previous notebook](https://www.kaggle.com/code/awsaf49/birdclef23-effnet-fsr-cutmixup-train).
* Unlike the previous tutorial, this notebook will perform spectrogram augmentation such as `TimeFreqMask` and `Normalization` on **GPU/TPU** and perform `CutMix` and `MixUp` with audio data on **CPU**.
* This notebook demonstrates how **pre-training on the BirdCLEF - 2020, 2021, 2022 & Xeno-Canto Extend** dataset can improve transfer learning performance. CNN backbones, like `EfficientNet`, struggle with spectrogram data even with ImageNet pre-trained weights as they are not fimilar with audio data. Pre-training on an audio dataset, like BirdCLEF, can mitigate this issue and can yield a ~$5\%$ improvement in local validation and ~$2\%$ improvement in leaderboard.
```

### Cell 6 code

```
# # Tensorflow for tpu-vm
# !pip install -q /lib/wheels/tensorflow-2.9.1-cp38-cp38-linux_x86_64.whl
# # Tensorflow utilities
!pip install -q tensorflow-addons==0.19.0
!pip install -q tensorflow-probability==0.19.0
!pip install -q tensorflow-io==0.32.0
```

### Cell 7 markdown

```
In this notebook to make the code as compact as possible `tensorflow_extra` library. This library contains all the necessary layers that will do ops like `MelSpectrogram`, `TimeFreqMask`, `ZScoreMinMax` on GPU/TPU. You are welcome to check the [source code](https://github.com/awsaf49/tensorflow_extra).
> **Note**: `tensorflow_extra` library is not any offical library  from TensorFlow rather a custom library built by me to ease the workflow.
Additionally, this notebook will use [this library](https://github.com/awsaf49/efficientnet-spec) for `EfficientNet` models with **Filter Stride Reduction (FSR)**.
```

### Cell 8 code

```
!pip install -qU tensorflow_extra --no-deps
# efficientnet with filter stride reduction (FSR)
!pip install -qU git+https://github.com/awsaf49/efficientnet-spec
```

### Cell 10 code

```
# Import tensorflow
import tensorflow as tf
# Import required tensorflow modules
import tensorflow_io as tfio
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
```

### Cell 14 code

```
    comment = 'EfficientNetB1|No-FSR|t=10s|128x384|cutmix'
    # Inference batch size, test time augmentation, and drop remainder
    tta = 1
    # Number of epochs, model name, and number of folds
    model_name = 'EfficientNetB1'
    num_fold = 5
    # Selected folds for training and evaluation
    selected_folds = [0]
    # Learning rate, optimizer, and scheduler
    scheduler = 'cos'
    # Data augmentation parameters
    augment=True
    # Audio Augmentation Settings
    audio_augment_prob = 0.5
    mixup_prob = 0.65
    mixup_alpha = 0.5
    cutmix_prob = 0.65
    cutmix_alpha = 2.5
    timeshift_prob = 0.0
```

### Cell 16 code

```
tf.keras.utils.set_random_seed(CFG.seed)
```

### Cell 17 markdown

```
# WandB 🪄
<img src="https://camo.githubusercontent.com/dd842f7b0be57140e68b2ab9cb007992acd131c48284eaf6b1aca758bfea358b/68747470733a2f2f692e696d6775722e636f6d2f52557469567a482e706e67" width="400" alt="Weights & Biases" />

To track model's training I'll be using **Weights & Biases** tool. Weights & Biases (W&B) is MLOps platform for tracking our experiemnts. We can use it to Build better models faster with experiment tracking, dataset versioning, and model management. Specifically for this notebook, we can do error analysis to check in which audio files models are struggling as we can also log **audio** file in WandB.
```

### Cell 26 code

```
df_23 = pd.read_csv(f'{BASE_PATH3}/train_metadata.csv')
```

### Cell 28 code

```
df_20 = pd.read_csv(f'{BASE_PATH0}/train.csv')
df_xam = pd.read_csv(f'{BASE_PATH4}/train_extended.csv')
df_xnz = pd.read_csv(f'{BASE_PATH5}/train_extended.csv')
df_21 = pd.read_csv(f'{BASE_PATH1}/train_metadata.csv')
df_22 = pd.read_csv(f'{BASE_PATH2}/train_metadata.csv')
# Merge 2021 and 2022 for pretraining
```

### Cell 43 code

```
from sklearn.model_selection import StratifiedKFold
# Initialize the StratifiedKFold object with 5 splits and shuffle the data
skf1 = StratifiedKFold(n_splits=25, shuffle=True, random_state=CFG.seed)
skf2 = StratifiedKFold(n_splits=CFG.num_fold, shuffle=True, random_state=CFG.seed)
# Create a new column in the dataframe to store the fold number for each row
df_pre["fold"] = -1
df_23["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf1.split(df_pre, df_pre['primary_label'])):
    df_pre.loc[val_idx, 'fold'] = fold
for fold, (train_idx, val_idx) in enumerate(skf2.split(df_23, df_23['primary_label'])):
    df_23.loc[val_idx, 'fold'] = fold
```

### Cell 45 code

```
    # Add a new column to select samples for cross validation
    # identify the classes that have less than the threshold number of samples
    # identify the classes that have less than the threshold number of samples
```

## 1st rank 16 quality strong score 72

File: `Audio/birdclef-2021/rank16_1st/kneroma-secondary-labels.ipynb`
Cells: 31 total, 22 code, 9 markdown

### Cell 2 markdown

```
* The inference is based on these [resnest50 weights](https://www.kaggle.com/kneroma/kkiller-birdclef-models-public). Please, don't forget upvoting the dataset to make it more visible for others
* The inference pipeline is optimized as much as I can in order to reduce execution time
```

### Cell 7 code

```
import torch
from torch import nn
from  torch.utils.data import Dataset, DataLoader
from resnest.torch import resnest50
```

### Cell 9 code

```
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_SUB_PATH = "../input/birdclef-2021/sample_submission.csv"
    # SAMPLE_SUB_PATH = "../input/birdclef-2021/sample_submission.csv"
```

### Cell 12 code

```
        V = np.clip(X, _min, _max)
```

### Cell 13 code

```
        image = np.stack([image, image, image])
        images = np.stack(images)
```

### Cell 15 code

```
df_train = pd.read_csv("../input/birdclef-2021/train_metadata.csv")
```

### Cell 18 code

```
    dummy_device = torch.device("cpu")
    d = torch.load(checkpoint_path, map_location=dummy_device)
```

### Cell 19 code

```
    Path("../input/kneroma-secondl/secondl-label0.50-birdclef_resnest50_fold0_epoch_11_f1_val_07361_20210521014834.pth"),
```

### Cell 20 code

```
@torch.no_grad()
    o = (-out).argsort(1)
```

### Cell 22 code

```
    with torch.no_grad():
            xb = torch.from_numpy(test_data[idx]).to(DEVICE)
                o = torch.sigmoid(o)
```

### Cell 25 code

```
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH, usecols=["row_id"])
        sub = sample_sub.merge(sub, on="row_id", how="left")
```

### Cell 27 code

```
sub.to_csv("submission.csv", index=False)
```
