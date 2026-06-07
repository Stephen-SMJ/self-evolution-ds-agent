# Audio / freesound-audio-tagging-2019

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 275 quality strong score 61

File: `Audio/freesound-audio-tagging-2019/rank275_70pct/simple-2d-cnn-classifier-with-pytorch.ipynb`
Cells: 31 total, 25 code, 6 markdown

### Cell 1 code

```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
```

### Cell 2 code

```
torch.cuda.is_available()
```

### Cell 4 code

```
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
```

### Cell 6 code

```
def _one_sample_positive_class_precisions(scores, truth):
      scores: np.array of (num_classes,) giving the individual classifier scores.
    num_classes = scores.shape[0]
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
            _one_sample_positive_class_precisions(scores[sample_num, :],
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
```

### Cell 9 code

```
    'sample_submission': dataset_dir / 'sample_submission.csv',
```

### Cell 10 code

```
train_curated = pd.read_csv(csvs['train_curated'])
train_noisy = pd.read_csv(csvs['train_noisy'])
```

### Cell 11 code

```
test_df = pd.read_csv(csvs['sample_submission'])
```

### Cell 16 code

```
        label = torch.from_numpy(label).float()
```

### Cell 17 code

```
    def __init__(self, fnames, mels, transforms, tta=5):
        self.tta = tta
        return len(self.fnames) * self.tta
```

### Cell 19 markdown

```
### model
```

### Cell 21 code

```
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
```

### Cell 24 code

```
    valid_dataset = FATTrainDataset(x_val, y_val, train_transforms)
    valid_loader = DataLoader(valid_dataset, batch_size=test_batch_size, shuffle=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        valid_preds = np.zeros((len(x_val), num_classes))
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            preds = torch.sigmoid(preds)
            valid_preds[i * test_batch_size: (i+1) * test_batch_size] = preds.cpu().numpy()
            avg_val_loss += loss.item() / len(valid_loader)
        score, weight = calculate_per_class_lwlrap(y_val, valid_preds)
        lwlrap = (score * weight).sum()
        scheduler.step()
            torch.save(model.state_dict(), 'weight_best.pt')
```

## 40pct rank 173 quality strong score 73

File: `Audio/freesound-audio-tagging-2019/rank173_40pct/fat-2019-80-place-public-lb-solution-lwlrap-70-2.ipynb`
Cells: 23 total, 19 code, 4 markdown

### Cell 0 markdown

```
* Pure CNN with global average pooling to handle different input size at training and inference stage
* Using SpatialDropout in CNN helped a lot. Using PRelu helped a bit.
* 5 fold cv was used using skmultilearn.model_selection.IterativeStratification
```

### Cell 1 markdown

```
* https://www.kaggle.com/CVxTz/keras-cnn-starter
* https://www.kaggle.com/jmourad100/keras-eda-and-cnn-starter
```

### Cell 2 code

```
from imgaug import augmenters as iaa
from keras.models import Model, Sequential
from keras.layers import (Convolution1D, Input, Dense, Flatten, Dropout, GlobalAveragePooling1D, concatenate,
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback
from keras.optimizers import Adam, SGD, RMSprop
from keras.losses import sparse_categorical_crossentropy
from keras.utils.np_utils import to_categorical
from keras.constraints import max_norm, MinMaxNorm
```

### Cell 4 code

```
# Keras reproduce score (then init all model seed)
import tensorflow as tf
```

### Cell 6 code

```
def pad2d(inp, use_mfcc=True, use_conv2d=True, augment=False, maxshape=maxshape, channels=1):
    if augment:
        if augment:
```

### Cell 7 code

```
import sklearn.metrics
def _one_sample_positive_class_precisions(scores, truth):
    scores: np.array of (num_classes,) giving the individual classifier scores.
  num_classes = scores.shape[0]
  retrieved_classes = np.argsort(scores)[::-1]
  # class_rankings[top_scoring_class_index] == 0 etc.
  class_rankings = np.zeros(num_classes, dtype=np.int)
  class_rankings[retrieved_classes] = range(num_classes)
  retrieved_class_true[class_rankings[pos_class_indices]] = True
      retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
      (1 + class_rankings[pos_class_indices].astype(np.float)))
  def accumulate_samples(self, batch_truth, batch_scores):
    """Cumulate a new batch of samples into the metric.
      scores: np.array of (num_samples, num_classes) giving the
        classifier-under-test's real-valued score for each class for each
    assert batch_scores.shape == batch_truth.shape
    for truth, scores in zip(batch_truth, batch_scores):
        _one_sample_positive_class_precisions(scores, truth))
```

### Cell 8 code

```
        print("validation lwrap: ", lwrap.overall_lwlrap())
```

### Cell 9 code

```
    model.add(Conv2D(filters=nclass, kernel_size=(1,1), padding="valid", trainable = trainable,  kernel_initializer='he_uniform', activation="sigmoid"))
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['categorical_accuracy'])
```

### Cell 10 code

```
softened_weights = {0:"../input/softened-noisy-kfold-fold-0/validation_lwrap_curated_noisy_0_best_weight.h5",
                   1:"../input/softened-noisy-kfold-fold-1/validation_lwrap_curated_noisy_1_best_weight.h5",
                   2:"../input/softened-noisy-kfold-fold-2/validation_lwrap_curated_noisy_2_best_weight.h5",
                   3:"../input/softened-noisy-kfold-fold-3/validation_lwrap_curated_noisy_3_best_weight.h5",
                   4:"../input/softened-noisy-kfold-fold-4/validation_lwrap_curated_noisy_4_best_weight.h5"}
```

### Cell 12 code

```
df = pd.read_csv('../input/freesound-audio-tagging-2019/sample_submission.csv')
```

### Cell 22 code

```
df.to_csv("submission.csv", index=False)
```

## 20pct rank 100 quality strong score 61

File: `Audio/freesound-audio-tagging-2019/rank100_20pct/cnn-2d-basic-solution-powered-by-fast-ai.ipynb`
Cells: 32 total, 23 code, 9 markdown

### Cell 0 markdown

```
# CNN 2D Basic Solution Powered by fast.ai
It's CNN, even ImageNet pretrained model works fine with audio 2D image like data.
- Converting audio to 2D image like array, so that we can simply exploit strong CNN classifier.
- Now fast.ai library ready to use lwlrap as metric: https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8
- And TTA! https://github.com/fastai/fastai/blob/master/fastai/vision/tta.py --> Oops, it might not be effective for this problem. Now planning to update one more...
```

### Cell 1 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
```

### Cell 3 code

```
CSV_SUBMISSION = DATA/'sample_submission.csv'
for folder in [WORK, IMG_TRN_CURATED, IMG_TRN_NOISY, IMG_TEST]:
    Path(folder).mkdir(exist_ok=True, parents=True)
df = pd.read_csv(CSV_TRN_CURATED)
test_df = pd.read_csv(CSV_SUBMISSION)
```

### Cell 7 code

```
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)
```

### Cell 11 code

```
def _one_sample_positive_class_precisions(scores, truth):
      scores: np.array of (num_classes,) giving the individual classifier scores.
    num_classes = scores.shape[0]
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
            _one_sample_positive_class_precisions(scores[sample_num, :],
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
def lwlrap(scores, truth, **kwargs):
    score, weight = calculate_per_class_lwlrap(to_np(truth), to_np(scores))
    return torch.Tensor([(score * weight).sum()])
```

### Cell 12 code

```
src = (ImageList.from_csv(WORK/'image', Path('../../')/CSV_TRN_CURATED, folder='trn_curated')
```

### Cell 14 code

```
learn = cnn_learner(data, models.resnet18, pretrained=False, metrics=[lwlrap])
```

### Cell 24 code

```
# https://discuss.pytorch.org/t/how-to-visualize-the-actual-convolution-filters-in-cnn/13850
    if isinstance(conv1, torch.nn.modules.container.Sequential):
```

### Cell 25 code

```
learn.save('fat2019_fastai_cnn2d_stage-2')
```

### Cell 27 code

```
test = ImageList.from_csv(WORK/'image', Path('../..')/CSV_SUBMISSION, folder='test')
preds, _ = learn.TTA(ds_type=DatasetType.Test) # <== Simply replacing from learn.get_preds()
```

### Cell 28 code

```
test_df.to_csv('submission.csv', index=False)
```

### Cell 29 code

```
learn = cnn_learner(data, models.resnet18, pretrained=False, metrics=[lwlrap])
learn.load('fat2019_fastai_cnn2d_stage-2');
```

## 10pct rank 43 quality reject score -67

File: `Audio/freesound-audio-tagging-2019/rank43_10pct/sample-submission-all-random.ipynb`
Cells: 2 total, 2 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv("../input/sample_submission.csv")
data.to_csv("submission.csv", index=False)
```

## 1st rank 100 quality strong score 66

File: `Audio/freesound-audio-tagging-2019/rank100_1st/cnn-2d-basic-solution-powered-by-fast-ai.ipynb`
Cells: 32 total, 23 code, 9 markdown

### Cell 0 markdown

```
# CNN 2D Basic Solution Powered by fast.ai
It's CNN, even ImageNet pretrained model works fine with audio 2D image like data.
- Converting audio to 2D image like array, so that we can simply exploit strong CNN classifier.
- Now fast.ai library ready to use lwlrap as metric: https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8
- And TTA! https://github.com/fastai/fastai/blob/master/fastai/vision/tta.py --> Oops, it might not be effective for this problem. Now planning to update one more...
```

### Cell 1 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
```

### Cell 3 code

```
CSV_SUBMISSION = DATA/'sample_submission.csv'
for folder in [WORK, IMG_TRN_CURATED, IMG_TRN_NOISY, IMG_TEST]:
    Path(folder).mkdir(exist_ok=True, parents=True)
df = pd.read_csv(CSV_TRN_CURATED)
test_df = pd.read_csv(CSV_SUBMISSION)
```

### Cell 7 code

```
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)
```

### Cell 11 code

```
def _one_sample_positive_class_precisions(scores, truth):
      scores: np.array of (num_classes,) giving the individual classifier scores.
    num_classes = scores.shape[0]
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
            _one_sample_positive_class_precisions(scores[sample_num, :],
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
def lwlrap(scores, truth, **kwargs):
    score, weight = calculate_per_class_lwlrap(to_np(truth), to_np(scores))
    return torch.Tensor([(score * weight).sum()])
```

### Cell 12 code

```
src = (ImageList.from_csv(WORK/'image', Path('../../')/CSV_TRN_CURATED, folder='trn_curated')
```

### Cell 14 code

```
learn = cnn_learner(data, models.resnet18, pretrained=False, metrics=[lwlrap])
```

### Cell 24 code

```
# https://discuss.pytorch.org/t/how-to-visualize-the-actual-convolution-filters-in-cnn/13850
    if isinstance(conv1, torch.nn.modules.container.Sequential):
```

### Cell 25 code

```
learn.save('fat2019_fastai_cnn2d_stage-2')
```

### Cell 27 code

```
test = ImageList.from_csv(WORK/'image', Path('../..')/CSV_SUBMISSION, folder='test')
preds, _ = learn.TTA(ds_type=DatasetType.Test) # <== Simply replacing from learn.get_preds()
```

### Cell 28 code

```
test_df.to_csv('submission.csv', index=False)
```

### Cell 29 code

```
learn = cnn_learner(data, models.resnet18, pretrained=False, metrics=[lwlrap])
learn.load('fat2019_fastai_cnn2d_stage-2');
```
