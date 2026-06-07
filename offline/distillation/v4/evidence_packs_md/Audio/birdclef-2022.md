# Audio / birdclef-2022

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 555 quality usable score 37

File: `Audio/birdclef-2022/rank555_70pct/birdclef2022-simple-starter-code-error-analysis.ipynb`
Cells: 25 total, 17 code, 8 markdown

### Cell 0 markdown

```
- This notebook is based on inference results on validation set  from ["PyTorch Simple Starter Using only 21 classes"](https://www.kaggle.com/code/myso1987/pytorch-simple-starter-using-only-21-classes)
```

### Cell 1 code

```
import torch
import torchaudio
```

### Cell 2 code

```
df = pd.read_csv(csv_file)
```

### Cell 3 markdown

```
`csv_file` is the output of using the validation set and model of the above code.
```

### Cell 5 markdown

```
The vocab of the scored bird to be classified in 21 species
```

### Cell 14 code

```
multiclass.to_csv("multi_hot.csv")
```

### Cell 18 code

```
mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=1024,hop_length=256,n_mels=128)
spec_converter = torchaudio.transforms.Spectrogram(n_fft=1024)
db_converter = torchaudio.transforms.AmplitudeToDB()
```

### Cell 19 code

```
    return vocab[torch.where(torch.tensor(multiclass[multiclass.filename==x].iloc[:,1:].values).squeeze())]
```

### Cell 22 code

```
    y,sr = torchaudio.load(file_path)
    prediction = vocab[torch.where(torch.tensor(multiclass[multiclass.filename==filename].iloc[:,1:].values).squeeze())]
```

## 40pct rank 331 quality weak score 21

File: `Audio/birdclef-2022/rank331_40pct/birdclef-random-guess-baseline.ipynb`
Cells: 7 total, 6 code, 1 markdown

### Cell 1 code

```
# but for now, let's stick with parsing the test_soundscape folder.
```

### Cell 2 code

```
# Load scored birds
with open('../input/birdclef-2022/scored_birds.json') as sbfile:
    scored_birds = json.load(sbfile)
```

### Cell 3 code

```
    # Each scored bird gets a random value in our case
        for bird in scored_birds:
            # This is our random prediction score for this bird
            score = np.random.uniform()
            # Assemble the row_id which we need to do for each scored bird
            # apply a "confidence" threshold of 0.5
            pred['target'].append(True if score > 0.5 else False)
```

### Cell 6 code

```
results.to_csv("submission.csv", index=False)
```

## 20pct rank 175 quality strong score 67

File: `Audio/birdclef-2022/rank175_20pct/birdclef2022pytorch-tpu-doesnt-work-what-is-wrong.ipynb`
Cells: 31 total, 25 code, 6 markdown

### Cell 3 code

```
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py  > /dev/null
!python pytorch-xla-env-setup.py --version 1.8.1 > /dev/null
```

### Cell 4 code

```
!pip install ../input/torchlibrosa/torchlibrosa-0.0.5-py3-none-any.whl > /dev/null
```

### Cell 5 code

```
!pip install timm  > /dev/null
```

### Cell 6 code

```
# fold 1 test run
```

### Cell 8 code

```
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn import metrics
from sklearn.metrics import mean_squared_error, roc_auc_score
from albumentations.core.transforms_interface import ImageOnlyTransform
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation
import albumentations as A
import albumentations.pytorch.transforms as T
```

### Cell 9 code

```
import transformers
from torch.cuda.amp import autocast, GradScaler
```

### Cell 11 code

```
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.serialization as xser
```

### Cell 12 code

```
train = pd.read_csv('../input/birdclef-2022/train_metadata.csv')
```

### Cell 14 code

```
train = pd.merge(train, path_df, on='filename')
```

### Cell 15 code

```
Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for n, (trn_index, val_index) in enumerate(Fold.split(train, train['primary_label'])):
    train.loc[val_index, 'kfold'] = int(n)
train['kfold'] = train['kfold'].astype(int)
```

### Cell 16 code

```
train.to_csv('train_folds.csv', index=False)
```

### Cell 17 code

```
    cutmix_and_mixup_epochs = 18
    folds = [0] # [0, 1, 2, 3, 4]
    N_FOLDS = 5
    valid_bs = 32 # 64
    base_model_name = "tf_efficientnet_b0_ns"
    EARLY_STOPPING = True
    main_metric = "epoch_f1_at_03"
```

## 10pct rank 80 quality strong score 61

File: `Audio/birdclef-2022/rank80_10pct/hawaii-train-nhah.ipynb`
Cells: 20 total, 20 code, 0 markdown

### Cell 0 code

```
!pip install ../input/torchlibrosa/torchlibrosa-0.0.5-py3-none-any.whl > /dev/null
```

### Cell 2 code

```
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn import metrics
from sklearn.metrics import mean_squared_error, roc_auc_score
from albumentations.core.transforms_interface import ImageOnlyTransform
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation
import albumentations as A
import albumentations.pytorch.transforms as T
import transformers
from torch.cuda.amp import autocast, GradScaler
```

### Cell 4 code

```
train = pd.read_csv('../input/birdies/train_NHaH.csv')
```

### Cell 7 code

```
train = pd.merge(train, path_df, on='filename')
```

### Cell 8 code

```
Fold = StratifiedKFold(n_splits=32, shuffle=True, random_state=56)
for n, (trn_index, val_index) in enumerate(Fold.split(train, train['primary_label'])):
    train.loc[val_index, 'kfold'] = int(n)
train['kfold'] = train['kfold'].astype(int)
```

### Cell 9 code

```
train.to_csv('train_folds.csv', index=False)
```

### Cell 10 code

```
    cutmix_and_mixup_epochs = 20
    folds = [1]
    N_FOLDS = 32
    valid_bs = 32 # 64
    base_model_name = "tf_efficientnet_b0_ns"
    EARLY_STOPPING = True
    main_metric = "epoch_f1_at_03"
```

### Cell 11 code

```
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented
class PitchShift(AudioTransform):
        augmented = librosa.effects.pitch_shift(y, sr, n_steps)
        return augmented
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented
```

### Cell 12 code

```
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Cell 13 code

```
    return metrics.roc_auc_score(np.array(y_true), np.array(y_pred))
class MetricMeter(object):
        self.f1_03 = metrics.f1_score(np.array(self.y_true), np.array(self.y_pred) > 0.3, average="micro")
        self.f1_05 = metrics.f1_score(np.array(self.y_true), np.array(self.y_pred) > 0.5, average="micro")
        probas = torch.sigmoid(preds)
    X = np.stack([X, X, X], axis=-1)
        V = np.clip(X, _min, _max)
                   'valid' : A.Compose([A.Normalize(mean, std),]), }
class WaveformDataset(torch.utils.data.Dataset):
```

### Cell 14 code

```
def interpolate(x: torch.Tensor, ratio: int):
def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        x = torch.sum(norm_att * cla, dim=2)
            return torch.sigmoid(x)
class TimmSED(nn.Module):
        self.spec_augmenter = SpecAugmentation(time_drop_width=64//2, time_stripes_num=2,
        base_model = timm.create_model(base_model_name, pretrained=pretrained, in_chans=in_channels)
                x = self.spec_augmenter(x)
        x = torch.mean(x, dim=3)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
```

### Cell 15 code

```
def interpolate(x: torch.Tensor, ratio: int):
def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        x = torch.sum(norm_att * cla, dim=2)
            return torch.sigmoid(x)
class TimmSED(nn.Module):
        self.spec_augmenter = SpecAugmentation(time_drop_width=64//2, time_stripes_num=2,
        base_model = timm.create_model(base_model_name, pretrained=pretrained, in_chans=in_channels)
                x = self.spec_augmenter(x)
        x = torch.mean(x, dim=3)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
```

## 1st rank 1 quality usable score 38

File: `Audio/birdclef-2022/rank1_1st/birdnet-inference.ipynb`
Cells: 13 total, 10 code, 3 markdown

### Cell 1 markdown

```
## Note 2: The ensemble of the ensemble of our own model could also reach gold zone, and we published all trained model (refer: https://www.kaggle.com/code/leonshangguan/private-7-8-final-of-submission Version 6)
```

### Cell 6 code

```
    data = pd.read_csv(file, sep='\t')
```

### Cell 8 code

```
sample_submission = pd.read_csv('../input/birdclef-2022/sample_submission.csv')
sample_submission
```

### Cell 9 code

```
for i in range(len(sample_submission)):
    sample = sample_submission.row_id[i]
        sample_submission.iat[i, 1] = (target_bird in pred_dicts[key])
```

### Cell 11 code

```
sample_submission.to_csv("submission.csv", index=False)
```

### Cell 12 code

```
sample_submission
```
