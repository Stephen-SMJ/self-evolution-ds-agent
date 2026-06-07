# Audio / rfcx-species-audio-detection

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 575 quality strong score 67

File: `Audio/rfcx-species-audio-detection/rank575_70pct/recensementoiseauxamazonie.ipynb`
Cells: 31 total, 16 code, 15 markdown

### Cell 2 code

```
    !pip -q install timm
    !pip -q install torchlibrosa
```

### Cell 4 code

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from transformers import get_linear_schedule_with_warmup
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
```

### Cell 5 markdown

```
### About Sound Event Detection(SED)

Sound event detection (SED) is the task of detecting the type as well as
the onset and offset times of sound events in audio streams.

In this notebook i will show how to train Sound Event Detection (SED) model with only weak annotation.

In SED task, we need to detect sound events from continuous (long) audio clip, and provide prediction of what sound event exists from when to when.

for more details

-> [Polyphonic Sound Event Detection
with Weak Labeling Paper](http://www.cs.cmu.edu/~yunwang/papers/cmu-thesis.pdf)

-> [Introduction to Sound Event Detection Notebook](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection)
```

### Cell 7 code

```
def interpolate(x: torch.Tensor, ratio: int):
    resolution reduction in downsampling of a CNN.
def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    output = torch.cat((framewise_output, pad), dim=1)
```

### Cell 9 code

```
def create_folds():
    Split kaggle dataset into [5] folds. Re-write the csv with fold information.
    output: ./train_folds.csv
    train = pd.read_csv(args.train_tp_csv).sort_values("recording_id")
    train_gby = train.groupby("recording_id")[["species_id"]].first().reset_index()
    train_gby.loc[:, 'kfold'] = -1
    kfold = StratifiedKFold(n_splits=args.FOLDS)
    for fold, (t_idx, v_idx) in enumerate(kfold.split(X, y)):
        train_gby.loc[v_idx, "kfold"] = fold # mark validation set : 0..4
    train = train.merge(train_gby[['recording_id', 'kfold']], on="recording_id", how="left")
    print('train.kfold.value_counts:\n{}'.format(train.kfold.value_counts()))
    train.to_csv(args.train_csv, index=False)
def create_foldsfp():
    Split kaggle dataset into [5] folds. Re-write the csv with fold information.
    output: ./train_folds.csv
    trainfp = pd.read_csv(args.train_fp_csv).sort_values("recording_id")
    trainfp_gby = trainfp.groupby("recording_id")[["species_id"]].first().reset_index()
    trainfp_gby.loc[:, 'kfold'] = -1
    kfold = StratifiedKFold(n_splits=args.FOLDS)
    for fold, (t_idx, v_idx) in enumerate(kfold.split(X, y)):
        trainfp_gby.loc[v_idx, "kfold"] = fold # mark validation set : 0..4
    trainfp = trainfp.merge(trainfp_gby[['recording_id', 'kfold']], on="recording_id", how="left")
    print('trainfp.kfold.value_counts:\n{}'.format(trainfp.kfold.value_counts()))
    trainfp.to_csv(args.trainfp_csv, index=False)
```

### Cell 10 markdown

```
1. Model takes raw waveform and converted into log-melspectogram using `torchlibrosa`'s module
2. spectogram converted into 3-channels input for ImageNet pretrain model to extract features from CNN's
```

### Cell 11 code

```
    #     norm_att = torch.softmax(torch.clamp(x1, -10, 10), dim=-1) #[16, 24, 10]
    #     cla = torch.sigmoid(x2) #[16, 24, 10]
    #     x = torch.sum(norm_att * cla, dim=2) #[16, 24]
        x = torch.max(x, dim=2)[0] #[16, 24]
        # x = torch.sum(x, dim=2) #[16, 24] insteqd of max => extremely high loss !
        x = torch.sigmoid(x)
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
        self.encoder = torchvision.models.resnet18(pretrained=True)
            x = self.spec_augmenter(x)
```

### Cell 14 code

```
        self.df = df.groupby("recording_id").agg(lambda x: list(x)).reset_index()
        _, idx = torch.max(target, dim=1)
            # y = np.stack([y[i:i+effective_length].astype(np.float32) for i in range(0, 60*sr+stride-effective_length, stride)])
            y = np.stack(y_)
```

### Cell 16 code

```
    #AA.PitchShift(min_semitones=-0.5, max_semitones=0.5, p=0.1),
    #AA.Shift(p=0.1),
    #AA.ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=1, p=0.05),
```

### Cell 18 code

```
def _lwlrap_sklearn(truth, scores):
    overall_lwlrap = metrics.label_ranking_average_precision_score(
        scores[nonzero_weight_sample_indices, :],                    # (144, 24)
class MetricMeter(object):
        # print(f'torch.max(y_true)={torch.max(y_true)} shape{y_true.shape} tot-len {len(self.y_true)}')
        #score_class, weight = lwlrap(np.array(self.y_true), np.array(self.y_pred))
        self.score = _lwlrap_sklearn(np.array(self.y_true), np.array(self.y_pred)) #(score_class * weight).sum()
            "lwlrap" : self.score
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        checkpoint = torch.load(args.pretrain_weights, map_location=args.device)
```

### Cell 20 code

```
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
        input = torch.clamp(input, 0.0, 0.99999)
```

### Cell 22 code

```
# def train_epoch(args, model, loader, criterion, optimizer, scheduler, epoch, fpDs):
#     scores = MetricMeter()
#         input = torch.tensor(input).to(args.device)
#         target = torch.tensor(target).to(args.device)
#         if scheduler and args.step_scheduler:
#             scheduler.step()
#         scores.update(target, output)
#     return scores.avg, losses.avg
def train_epoch(args, model, loader, criterion, optimizer, scheduler, epoch, fpDs):
    scores = MetricMeter()
        loss = torch.mean(loss)
        fp_input = torch.tensor([fp_sample[i]['image'] for i in range(len(fp_sample))])
        fp_target = torch.tensor([fp_sample[i]['target'] for i in range(len(fp_sample))])
        fp_loss = torch.mean(fp_loss)*0.1
        if scheduler and args.step_scheduler:
            scheduler.step()
        scores.update(target, output)
    return scores.avg, losses.avg
def valid_epoch(args, model, loader, criterion, epoch):
    scores = MetricMeter()
    with torch.no_grad():
            scores.update(target, output)
    # print(f"Valid E:{epoch} - Loss:{losses.avg:0.4f}")
    return scores.avg, losses.avg
    with torch.no_grad():
            output = torch.sum(output, dim=1) / seq
            #output, _ = torch.max(output, dim=1)
```

## 40pct rank 452 quality strong score 73

File: `Audio/rfcx-species-audio-detection/rank452_40pct/rfcx-train-resnet50-with-tpu.ipynb`
Cells: 57 total, 42 code, 15 markdown

### Cell 2 code

```
import tensorflow as tf
import tensorflow_addons as tfa
# from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
```

### Cell 3 code

```
        'arch': tf.keras.applications.ResNet50,
        'arch_preprocess': tf.keras.applications.resnet50.preprocess_input,
        'mixup': False
```

### Cell 11 code

```
    # center crop in validation
```

### Cell 14 code

```
#     #mod = tf.stack([first_slice,second_slice],1)
```

### Cell 20 code

```
# in validation, annotations will come to the center
```

### Cell 25 code

```
    def _specaugment(image):
        gau = tf.keras.layers.GaussianNoise(0.3)
        # specaugment
        image = tf.cond(tf.random.uniform([]) < 0.5, lambda: _specaugment(image), lambda: image)
```

### Cell 27 markdown

```
# Model
```

### Cell 28 code

```
        head = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(CLASS_N, bias_initializer=tf.keras.initializers.Constant(pi))
        model = tf.keras.Sequential([backbone, head])
```

### Cell 29 code

```
def _mixup(inp, targ):
```

### Cell 32 code

```
df = pd.read_csv("/kaggle/input/rfcx-species-audio-detection/train_tp.csv")
table = df.groupby('recording_id')['species_id'].apply(
skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)
idx_splits = list(skf.split(table.recording_id, np.stack(table.species_id.to_numpy())))
```

### Cell 33 code

```
         bins=CLASS_N,stacked=True)
```

### Cell 36 code

```
    if cfg['model_params']['mixup']:
        dataset = (dataset.map(_mixup, num_parallel_calls=AUTOTUNE)
```

## 20pct rank 181 quality usable score 43

File: `Audio/rfcx-species-audio-detection/rank181_20pct/rainforest-write-up.ipynb`
Cells: 17 total, 9 code, 8 markdown

### Cell 0 markdown

```
# My Training Script
This is the training script I used to approach the competition. If you have any questions, feel free to reach out and ask.
```

### Cell 1 code

```
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.metrics import label_ranking_average_precision_score
```

### Cell 2 code

```
annotations = pd.read_csv("../input/rfcx-species-audio-detection/train_tp.csv")
annotations_nl = pd.read_csv("../input/rfcx-species-audio-detection/train_fp.csv")
models = { 'inception': keras.applications.InceptionV3,
           'efficientnet': keras.applications.EfficientNetB3,
           'resnet': keras.applications.ResNet50 }
```

### Cell 3 markdown

```
# Data Preprocessing
Changing the input data from 1D sound signals to 2D Mel Spectrograms, cropped to where the labels are located along the time (x) axis, and label gathering.

The 1D sound signals are being converted to Mel Spectrograms because neural network architectures in image classification are well established while the data conversion doesn't lead to any loss of information, so it makes sense to use better tools for what is essentially the same data.

We're cropping the input images to the locations of the labels because feeding instead a 60-second clip would provide too much information for the model to learn properly what is what. e.g. if the label is tf.one_hot([0, 3, 15, 23], 24) given the entire clip, then distinguishing which part is which class is much more difficult than feeding the model each instance of the label directly.
```

### Cell 4 code

```
def get_labels_validation_numpy(recording_id, times, frequencies, annotations):
def mixup(inp, targ):
```

### Cell 5 markdown

```
Actual making of the TensorFlow Datasets using generators to fetch the data. The created datasets are then divided into five different folds. If you're unfamiliar with k-fold validation, see https://en.wikipedia.org/wiki/Cross-validation_(statistics).
Something I tried to do for some cross-validation and leaderboard correlation was to similarize the validation and inference methods i.e. validation follows inference in slicing and stacking each image then taking the max prediction for each slice to get the prediction for the entire image.
```

### Cell 6 code

```
def generate_dataset(batch_size, fold, train_files, validation_files):
            images = augment(images)
    def validation_generator():
        for i in range(len(validation_files)):
            recording_id = validation_files[i].split('/')[-1][:-5]
            image = tf.expand_dims(load_spec(validation_files[i], 0, 60), axis=-1)
                im = augment(im, training=False)
            images = tf.stack(images)
            labels = tf.stack(labels)
                                                                                  tf.float32)).repeat().batch(batch_size).map(mixup)
    validation_dataset = tf.data.Dataset.from_generator(validation_generator, output_types=(tf.float32,
    return train_dataset, validation_dataset
# Creating each fold
# the lists containing each class before selecting recording_ids for the folds
folds = 5
for fold in range(folds):
    validation_files = []
        class_ratio = class_records.shape[0] // folds
        train_files += class_records[:class_ratio*fold] + class_records[class_ratio*(fold+1):]
        validation_files += class_records[class_ratio*fold:class_ratio*(fold+1)]
    shuffle(validation_files)
    validation_files = [train_path+s+".flac" for s in validation_files]
    datasets.append(generate_dataset(BATCH_SIZE, fold, train_files, validation_files))
```

### Cell 7 markdown

```
# Data Augmentations
Various augmentations to improve the generalization of the model. I wasn't able to find any success in using them in this notebook.
```

### Cell 8 code

```
def apply_augmentation(image):
def augment(images, training=True):
    #    images = apply_augmentation(images)
def get_validation_data(index):
    f, start, end = load_spec(validation_files[index])
    s = validation_files[index].split('/')[-1][:-5]
```

### Cell 9 markdown

```
# Model Definition
After being sent through the model, the output is averaged across the frequency-axis (y-axis) then sent through an attention mechanism before the final predictions are made.
```

### Cell 10 code

```
    inp = keras.Input(input_shape)
    norm_att = tf.keras.layers.Conv1D(class_count, 1)(x)
    conv = tf.keras.layers.Conv1D(class_count, 1)(x)
    return tf.keras.Model(inp, x)
```

### Cell 11 markdown

```
# Competition metric
See https://stackoverflow.com/questions/55881642/how-to-interpret-label-ranking-average-precision-score.
Code here taken from https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2#Competition-Metric
```

## 10pct rank 103 quality strong score 55

File: `Audio/rfcx-species-audio-detection/rank103_10pct/inverse-stft.ipynb`
Cells: 29 total, 16 code, 13 markdown

### Cell 0 markdown

```
# Learnable Time-Frequency Representations
This assignment was about training models to learn different time-frequency representations of audio. I chose to work with the Rainforest dataset because I'm familiar with the dataset from a previous Kaggle competition and because I thought it was fun to use a bioacoustics dataset for the assignment.
```

### Cell 1 code

```
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
torchaudio.set_audio_backend('sox_io')
```

### Cell 3 code

```
df = pd.read_csv(rainforest_data / 'train_tp.csv')
```

### Cell 6 markdown

```
For each row in the dataframe we are going to extract a short segment of audio of the file corresponding to `recording_id`. We are going to center the extracted segment in the middle of `t_min` and `t_max`, and all extracted segments will be of the same lenght, for simplicity. We are going to do the segmentation in advance and save the extracted and normalized segments as tensors, for faster loading when we train the models.
We specify the wanted length and sample rate of the extracted segments, as well as the encoder we want to use:
```

### Cell 7 code

```
ENCODER = torchaudio.transforms.Spectrogram(n_fft=N_FFT)
```

### Cell 12 code

```
    audio, sr = torchaudio.load(fpath)
    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
    seg -= torch.mean(seg)
    seg /= torch.max(torch.abs(seg))
    assert segment.shape == torch.Size([CLIP_LEN_SAMPLES])
    torch.save(segment, waveform_tensors / fname)
```

### Cell 14 code

```
        return torch.load(self.fpaths[idx])
            waveform = torch.load(self.fpaths[idx])
```

### Cell 17 markdown

```
We now define the model we are going to use. The model takes normalized audio waveforms as input, the first part of the model encodes the input with some waveform -> time-frequency transformation. The second part of the model is the decoder, it consists of transposed 1d convolution blocks that upsamples the time dimension of the activations through the layers and simultanously downsamples the frequency dimension, producing a pure time domain signal
```

### Cell 20 code

```
    log_pred = torch.log(encoded_pred + 1e-8)
    log_target = torch.log(encoded_target+ 1e-8)
```

### Cell 21 markdown

```
The training loop is pretty straightforward, I decrease the learning rate by a factor of 10 when the validation loss stops to decrease.
```

### Cell 22 code

```
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.1)
        with torch.no_grad():
            torch.save(model, weights_path)
        scheduler.step(val_loss)
```

### Cell 23 code

```
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

## 1st rank 3 quality strong score 78

File: `Audio/rfcx-species-audio-detection/rank3_1st/rfcx-minimal.ipynb`
Cells: 23 total, 17 code, 6 markdown

### Cell 0 markdown

```
it starts "forgetting". The confidence for species 3 goes on decreasing which negatively impacts the lb score.
2. Augmenting other datasets\
    - Validation scheme should be similar to test scheme.
    be done during validation also.
    - I found Click Noise Augmentation to be very useful (https://librosa.org/doc/0.8.0/generated/librosa.clicks.html)
```

### Cell 2 code

```
import albumentations as A
from resnest.torch.resnet import ResNet, Bottleneck
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from torchvision.models import resnet18, resnet34, resnet50
from resnest.torch import resnest50
import torchaudio
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
```

### Cell 3 code

```
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
```

### Cell 4 code

```
    # Ranks of the predictions
    ranked_classes = torch.argsort(preds, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes).to(preds.device)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(
        ground_truth_ranks, dim=-1, descending=False)
    pos_matrix = torch.tensor(
    score_matrix = pos_matrix / sorted_ground_truth_ranks
    score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
    scores = score_matrix * score_mask_matrix
    score = scores.sum() / labels.sum()
    return score.item()
```

### Cell 5 code

```
    loss_fn = torch.nn.BCEWithLogitsLoss()
```

### Cell 7 code

```
# Mostly taken from https://www.kaggle.com/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented
class TimeShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_shift_second=2, sr=32000, padding_mode="zero"):
        self.max_shift_second = max_shift_second
        shift = np.random.randint(-self.sr * self.max_shift_second,
                                  self.sr * self.max_shift_second)
        augmented = np.roll(y, shift)
        #     if shift > 0:
        #         augmented[:shift] = 0
        #         augmented[shift:] = 0
        return augmented
        augmented = y * db_translated
        return augmented
```

### Cell 8 code

```
    X = np.stack([X, X, X], axis=-1)
        V = np.clip(X, _min, _max)
```

### Cell 9 code

```
        self.fp = pd.read_csv("../input/rfcxextras/cornell-train.csv")
        self.resampler = torchaudio.transforms.Resample(
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_mels=self.nmels,
            TimeShift(sr=self.sr),
        valid_df = self.tp[self.tp.recording_id == recording_id]
        valid_df = valid_df[(valid_df.t_min < t1) & (valid_df.t_max > t0)]
        if len(valid_df):
            np.put(labels, valid_df.species_id.unique(), 1)
            end_idx = int((valid_df.t_max.max() - t0)*self.sr)
        y = self.resampler(torch.from_numpy(y).float()).numpy()
        # do augmentation
```

### Cell 10 code

```
        self.resampler = torchaudio.transforms.Resample(
            melspec_stacked = np.load(fn)
            y_stacked = np.stack(np.split(y, self.num_splits), 0)
            melspec_stacked = []
            for y in y_stacked:
                y = self.resampler(torch.from_numpy(y).float()).numpy()
                melspec_stacked.append(melspec)
            melspec_stacked = np.stack(melspec_stacked)
            np.save(fn, melspec_stacked)
            return melspec_stacked, labels
            melspec_stacked = np.load(fn)
            return melspec_stacked
```

### Cell 11 code

```
    # model = torchvision.models.resnext50_32x4d(pretrained=False)
    # model = torchvision.models.resnext101_32x8d(pretrained=False)
    model = ResNet(**MODEL_CONFIGS["resnest50_fast_1s1x64d"])
    # model.load_state_dict(torch.load('resnext50_32x4d_extra_2.pt'))
    # model.load_state_dict(torch.load('resnext101_32x8d_wsl_extra_4.pt'))
    model.load_state_dict(torch.load(fn, map_location='cpu'))
```

### Cell 12 code

```
        pos_weight = torch.ones((24,))
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        train_sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset),
        optim = torch.optim.AdamW(self.parameters(), lr=self.config.lr,
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
        self.scheduler = scheduler
        return [optim], [scheduler]
```

### Cell 13 code

```
        with torch.no_grad():
        metrics = {"train_loss": loss.item(), "train_lwlrap": lwlrap}
        self.log_dict(metrics,
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        for i, x_partial in enumerate(torch.split(x, 1, dim=1)):
                preds = torch.max(preds, self(x_partial))
        metrics = {"val_loss": val_loss, "val_lwlrap": val_lwlrap}
        self.log_dict(metrics, prog_bar=True,
```
