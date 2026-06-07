# CV / tgs-salt-identification-challenge

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1908 quality strong score 73

File: `CV/tgs-salt-identification-challenge/rank1908_70pct/tgs-wj.ipynb`
Cells: 47 total, 34 code, 13 markdown

### Cell 0 markdown

```
enlarge validation set with 3*3 filter -> 79.56%<br/>
```

### Cell 1 code

```
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization,Activation,Add
from keras import backend as K
from keras import optimizers
```

### Cell 5 code

```
train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
```

### Cell 16 markdown

```
Split dataset into training data and validation data
```

### Cell 17 code

```
ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
```

### Cell 22 markdown

```
Define the Structure of model
```

### Cell 24 code

```
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
```

### Cell 26 code

```
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
```

### Cell 28 code

```
x_valid = np.append(x_valid, [np.flipud(x) for x in x_valid], axis=0)
y_valid = np.append(y_valid, [np.flipud(x) for x in y_valid], axis=0)
x_valid = np.append(x_valid, [np.flipud(x) for x in x_valid], axis=0)
y_valid = np.append(y_valid, [np.flipud(x) for x in y_valid], axis=0)
```

### Cell 30 markdown

```
Train the model
```

### Cell 31 code

```
early_stopping = EarlyStopping(patience=10, verbose=1)
model_checkpoint = ModelCheckpoint("./keras.model", save_best_only=True, verbose=1)
                    validation_data=[x_valid, y_valid],
                    callbacks=[early_stopping, model_checkpoint, reduce_lr])
```

### Cell 32 code

```
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax_acc.plot(history.epoch, history.history["val_acc"], label="Validation accuracy")
model = load_model("./keras.model")
```

## 40pct rank 1247 quality strong score 49

File: `CV/tgs-salt-identification-challenge/rank1247_40pct/u-net-image-segmentation.ipynb`
Cells: 25 total, 21 code, 4 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```

### Cell 2 code

```
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout,concatenate
from tensorflow.keras.layers import Conv2D,MaxPool2D, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler,ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
```

### Cell 3 code

```
train_df = pd.read_csv("/kaggle/input/tgs-salt-identification-challenge/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("/kaggle/input/tgs-salt-identification-challenge/depths.csv", index_col="id")
```

### Cell 12 code

```
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.1, random_state=42)
```

### Cell 13 code

```
X_train.shape, X_valid.shape
```

### Cell 16 markdown

```
U-Net model
```

### Cell 18 code

```
def UNet_Model(input_img):
```

### Cell 19 code

```
model = UNet_Model(input_img)
```

### Cell 21 code

```
    EarlyStopping(patience=10, verbose=1),
```

### Cell 22 code

```
results = model.fit(X_train,y_train,batch_size= 32, epochs= 50, callbacks=callbacks,validation_data=(X_valid, y_valid))
```

### Cell 24 code

```
model.evaluate(X_valid, y_valid, verbose=1)
```

## 20pct rank 687 quality strong score 73

File: `CV/tgs-salt-identification-challenge/rank687_20pct/tgs-resnet-final1.ipynb`
Cells: 73 total, 49 code, 24 markdown

### Cell 1 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
```

### Cell 3 code

```
train = pd.read_csv("../input/train.csv",
```

### Cell 10 code

```
thresholds = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
n_thresholds = len(thresholds)
    for t in thresholds:
    return hitCnt/n_thresholds
    scores = []
        score = meanHit(mask[i], mask_pred[i])
        scores.append(score)
    return np.mean(scores)
# Define IoU metric
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
    metric = []
#             metric.append(0)
#             metric.append(0)
#             metric.append(1)
        thresholds = np.arange(0.5, 1, 0.05)
        for thresh in thresholds:
        metric.append(np.mean(s))
    return np.mean(metric)
def my_iou_metric(label, pred):
```

### Cell 12 code

```
# Testing accuracy metric
```

### Cell 26 markdown

```
#### Train Validation Split
```

### Cell 35 code

```
print("ValidationX Shape = ", val_X.shape, "ValidationY Shape = ", val_Y.shape)
```

### Cell 37 code

```
def augment(X, Y):
print("Training Augmentation")
train_X, train_Y = augment(train_X, train_Y)
val_X, val_Y = augment(val_X, val_Y)
```

### Cell 41 code

```
import keras
# keras.backend.set_session(sess)
# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
```

### Cell 42 code

```
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
```

### Cell 44 code

```
optimizer = keras.optimizers.Adam()
              metrics = [my_iou_metric])
```

### Cell 45 code

```
import keras
earlyStopping = EarlyStopping(patience = 20, verbose = 1)
checkpointer = ModelCheckpoint('model-tgs-salt-1.h5',monitor = 'val_my_iou_metric',
reducelr=ReduceLROnPlateau(monitor='val_my_iou_metric',patience=5,
                    validation_data = (val_X, val_Y),
                    callbacks=[earlyStopping, checkpointer, reducelr])
```

### Cell 46 code

```
                   custom_objects = {'my_iou_metric':my_iou_metric})
```

## 10pct rank 284 quality weak score 19

File: `CV/tgs-salt-identification-challenge/rank284_10pct/data-generator-with-shift-and-mirror-augmentation.ipynb`
Cells: 2 total, 1 code, 1 markdown

### Cell 1 code

```
def transform_image(X, shift):
    Return a shifted image with mirrored padding
    return np.concatenate([X, X[:, np.arange(X.shape[1]-2, 0, -1), :], X], axis=1)[:, shift:shift+X.shape[1], :]
def batch_augmentation(X, y, shift_ratio=0.3, mirror_ratio=0.35):
    Return a batch with randomized augmentation
        if r < shift_ratio:
            shift = np.random.randint(2*X.shape[2]-1)
            X[i] = transform_image(X[i], shift)
            y[i] = transform_image(y[i], shift)
        elif r < shift_ratio+mirror_ratio:
def batch_generator(X, y, batch_size=32, shift_ratio=0.3, mirror_ratio=0.35):
            yield batch_augmentation(X[idx], y[idx], shift_ratio, mirror_ratio)
```

## 1st rank 4 quality usable score 26

File: `CV/tgs-salt-identification-challenge/rank4_1st/know-your-opponents-well.ipynb`
Cells: 10 total, 10 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
```

### Cell 1 code

```
df = pd.read_csv("../input/tgslb/tgs.csv")
```

### Cell 2 code

```
df=df.sort_values(ascending=False,by=['Score'])
```
