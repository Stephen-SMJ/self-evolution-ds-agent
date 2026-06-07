# CV / cassava-leaf-disease-classification

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 2727 quality strong score 67

File: `CV/cassava-leaf-disease-classification/rank2727_70pct/cassava-leaf-disease.ipynb`
Cells: 41 total, 26 code, 15 markdown

### Cell 1 code

```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB5
from keras.callbacks import EarlyStopping
print("Tensorflow version " + tf.__version__)
```

### Cell 4 code

```
train = pd.read_csv(root_path + 'train.csv')
```

### Cell 14 code

```
datagen = ImageDataGenerator(validation_split=0.2,
```

### Cell 16 code

```
valid_datagen_flow = datagen.flow_from_dataframe(
    subset='validation',
```

### Cell 17 markdown

```
Build the Model
```

### Cell 18 markdown

```
# Running the model
```

### Cell 19 markdown

```
    Early stopping, epocs 100
* Version 9: EfficientNetB0 epocs 10 --> accuracy: 0.7385
* Version 11-13: EfficientNetB0 epocs 50 --> accuracy: 0.84  validation=0.72
* Version 14: EfficientNetB0 epocs 20 --> accuracy:?
* Version 16: Batchsize 32 + flatten +drop+ dense512 +ephocs =10  accuracy: 0.74  validation=0.72
* Version 17: image size 128x128 accuracy: 0.81  validation=0.79
* Version 18: image size 256x256 , epoch=6 (<8H) occuracy 0.84 validation=0.8441
* version 19: image size 128X170 , epoch=6  occuracy=0.787 Valid=0.787
* version 20: image size 164X164 , epoch=6 , change from B0 to EfficientNetB4
* version 21: image size 164X164 , epoch=8 , change from B0 to EfficientNetB4 -accuaracy 0.81 validation 0.79
* Version 22: image size 164X164 , epoch=6 , change from B4 to EfficientNetB5  accuracy 078 valid=0.79
* version 25: save model file  +epoch=10  accuracy: 0.8186 - val_loss: 0.5699 - val_accuracy: 0.8004  (score 0.772)
* version 26: 256X256 +epoch=10 loss: 0.4181 - accuracy: 0.8583 - val_loss: 0.4416 - val_accuracy: 0.8537  (score 0.862)
Base on: https://www.kaggle.com/bununtadiresmenmor/starter-keras-efficientnet?select=sample_submission.csv
```

### Cell 20 code

```
early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=4)
```

### Cell 21 markdown

```
1. # Model EfficientNetB5
```

### Cell 22 code

```
model.add(EfficientNetB5(include_top = False, weights = "imagenet",
# model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation = "relu"))
model.add(tf.keras.layers.Dense(64, activation = "relu"))
model.add(tf.keras.layers.Dense(5, activation = "softmax"))
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            metrics = ["accuracy"])
# from tensorflow.keras.applications import EfficientNetB0
#     x = img_augmentation(inputs)
#     outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(x)
#     model = tf.keras.Model(inputs, outputs)
#         optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
# hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)
```

### Cell 24 code

```
from tensorflow.keras import utils
```

### Cell 25 code

```
from keras.callbacks import LearningRateScheduler
lrs = LearningRateScheduler(my_learning_rate)
```

## 40pct rank 1556 quality strong score 67

File: `CV/cassava-leaf-disease-classification/rank1556_40pct/very-simple-notebook-keras-densenet-gpu.ipynb`
Cells: 23 total, 17 code, 6 markdown

### Cell 1 code

```
from keras import layers
from keras.applications import DenseNet121, DenseNet169
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import tensorflow as tf
```

### Cell 5 code

```
train_df = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
test_df = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')
```

### Cell 16 code

```
    weights='../input/densenet-keras/DenseNet-BC-169-32-no-top.h5',
```

### Cell 17 code

```
        metrics=['accuracy']
```

### Cell 19 code

```
    validation_data=(x_val, y_val)
```

### Cell 22 code

```
test_df.to_csv('submission.csv',index=False)
```

## 20pct rank 780 quality strong score 61

File: `CV/cassava-leaf-disease-classification/rank780_20pct/efn-resnext-vit-ensemble-inference.ipynb`
Cells: 21 total, 10 code, 11 markdown

### Cell 2 code

```
from sklearn.model_selection import GroupKFold, StratifiedKFold
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from  torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
from sklearn.metrics import log_loss
!pip install ../input/timmpackagelatestwhl/timm-0.3.4-py3-none-any.whl
import timm
```

### Cell 4 code

```
    'model_arch': ['tf_efficientnet_b4_ns','resnext50_32x4d','vit_base_patch16_384'],
    'weight_path': sorted(os.listdir('../input/cassava-ensemble-model')),
    'valid_bs': 64,
    'tta': 2,
```

### Cell 6 code

```
train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
```

### Cell 8 code

```
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
```

### Cell 12 code

```
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize)
from albumentations.pytorch import ToTensorV2
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
```

### Cell 13 markdown

```
# Create Model
```

### Cell 14 code

```
class EnsembleModel(nn.Module):
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        if model_arch == 'tf_efficientnet_b4_ns':
```

### Cell 16 code

```
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
```

### Cell 18 code

```
        tst_loader = DataLoader(testset, batch_size=CFG['valid_bs'],num_workers=CFG['num_workers'],shuffle=False,pin_memory=False,)
        device = torch.device(CFG['device'])
        model = EnsembleModel(model_arch, train.label.nunique()).to(device)
            model.load_state_dict(torch.load(os.path.join('../input/cassava-ensemble-model',weight))['model'])
            with torch.no_grad():
                for _ in range(CFG['tta']):
                    tst_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, tst_loader, device)]
        torch.cuda.empty_cache()
```

### Cell 20 code

```
test.to_csv('submission.csv', index=False)
```

## 10pct rank 385 quality usable score 27

File: `CV/cassava-leaf-disease-classification/rank385_10pct/analyze-your-model-performance-by-confusion-matrix.ipynb`
Cells: 12 total, 10 code, 2 markdown

### Cell 0 markdown

```
# Analyze your model performance by confusion matrix
```

### Cell 2 code

```
df_train = pd.read_csv("../input/inference5folds/train_inference_5folds.csv")
```

### Cell 4 code

```
# accuracy for 5-fold
```

### Cell 6 code

```
from sklearn.metrics import confusion_matrix
```

### Cell 11 code

```
from sklearn.metrics import classification_report
```

## 1st rank 1 quality strong score 66

File: `CV/cassava-leaf-disease-classification/rank1_1st/cassava-leaf-disease-resnext50.ipynb`
Cells: 12 total, 11 code, 1 markdown

### Cell 0 markdown

```
This is the training notebook of the ResNeXt50 (32x4d) model we used in our final submission which scored ~91.3% on the public and private leaderboard of the Cassava Leaf Disease Classification 2020 competition ([Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification/)). You can find a description of our overall approach in this discussion post: ["1st Place Solution"](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/221957)
```

### Cell 2 code

```
    n_fold = 5
    trn_fold = [1,2,3,4,5]
```

### Cell 3 code

```
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from albumentations import (Compose, Normalize, Resize, RandomResizedCrop, HorizontalFlip, VerticalFlip, ShiftScaleRotate, Transpose)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import timm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Cell 4 code

```
def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
def seed_torch(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_torch(seed=CFG.seed)
```

### Cell 5 code

```
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
```

### Cell 6 code

```
            ShiftScaleRotate(p=0.5),
    elif data == 'valid':
```

### Cell 7 code

```
        self.model = timm.create_model(model_name, pretrained=pretrained)
```

### Cell 8 code

```
def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    scores = AverageMeter()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
def valid_fn(valid_loader, model, criterion, device):
    scores = AverageMeter()
    for step, (images, labels) in enumerate(valid_loader):
        with torch.no_grad():
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
                   step, len(valid_loader), batch_time=batch_time,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
```

### Cell 9 code

```
def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    train_dataset = TrainDataset(train_folds, transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data='valid'))
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size,
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
    best_score = 0.
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_folds[CFG.target_col].values
        scheduler.step(avg_val_loss)
        score = get_score(valid_labels, preds.argmax(1))
        LOGGER.info(f'Epoch {epoch+1} - Accuracy: {score}')
        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 'preds': preds}, OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best.pth')
    check_point = torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best.pth')
    valid_folds[[str(c) for c in range(5)]] = check_point['preds']
    valid_folds['preds'] = check_point['preds'].argmax(1)
    return valid_folds
```

### Cell 10 code

```
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.5f}')
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            _oof_df = train_loop(folds, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== fold: {fold} result ==========")
            get_result(_oof_df)
    get_result(oof_df)
    oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)
```

### Cell 11 code

```
train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
# Split into folds for cross validation - we used the same split for all the models we trained!
folds = train.merge(
    pd.read_csv("../input/cassava-leaf-disease-resnext/validation_data.csv")[["image_id", "fold"]], on="image_id")
```
