# Medical / rsna-intracranial-hemorrhage-detection

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 299 quality usable score 43

File: `Medical/rsna-intracranial-hemorrhage-detection/rank299_70pct/basic-eda-albumentations-augs.ipynb`
Cells: 60 total, 42 code, 18 markdown

### Cell 0 markdown

```
# Basic EDA + albumentations augs
```

### Cell 1 code

```
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
```

### Cell 4 code

```
train_df = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
```

### Cell 5 code

```
stage_1_sample_submission = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')
```

### Cell 13 code

```
train_df.groupby('diagnosis').sum().plot(kind='bar',figsize = (10, 5));
```

### Cell 15 code

```
image_lable = train_df.groupby('image').sum()
```

### Cell 18 code

```
image_lable = train_df.query('diagnosis!="any"').groupby('image').sum()
```

### Cell 30 code

```
stage_1_sample_submission['image'] = stage_1_sample_submission['ID'].str.slice(stop=12)
stage_1_sample_submission['diagnosis'] = stage_1_sample_submission['ID'].str.slice(start=13)
```

### Cell 31 code

```
stage_1_sample_submission.shape
```

### Cell 36 code

```
stage_1_sample_submission_d = stage_1_sample_submission.drop_duplicates(subset='image')
```

### Cell 37 code

```
stage_1_sample_submission_d.shape
```

### Cell 39 code

```
test_widths, test_heights, max_test, min_test = get_image_sizes(stage_1_sample_submission_d.sample(10000), train = False)
```

## 40pct rank 171 quality strong score 49

File: `Medical/rsna-intracranial-hemorrhage-detection/rank171_40pct/5th-preprocessing-adjacent-images-and-cropping.ipynb`
Cells: 29 total, 18 code, 11 markdown

### Cell 0 markdown

```
# Preprocessing: Spatially adjacent RGB images and cropping
This is a follow on from my previous notebook https://www.kaggle.com/anjum48/reconstructing-3d-volumes-from-metadata

This is also a part of our 5th place solution https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117232#latest-672657
```

### Cell 6 code

```
    studies = metadata.groupby("StudyInstanceUID")
```

### Cell 7 code

```
train_triplets.to_csv("train_triplets.csv")
test_triplets.to_csv("stage_1_test_triplets.csv")
```

### Cell 12 code

```
img = np.stack(rgb, -1)
img = np.clip(img, 0, 255).astype(np.uint8)
```

### Cell 13 markdown

```
On the final line above, we actually clip the image between 0 & 255 Hounfield units. This is ok, since most of the important features in the image are between this range.

Since the image is still technically in Hounsfield units, you can apply a window to it later (e.g. brain, subdural etc.), however since we have already clipped this image, this may impact the bone window.

Let's check out our image:
```

### Cell 24 code

```
        Originally made as a image transform for use with PyTorch, but too slow to run on the fly :(
```

## 20pct rank 62 quality strong score 67

File: `Medical/rsna-intracranial-hemorrhage-detection/rank62_20pct/final-cnn-kernel.ipynb`
Cells: 19 total, 19 code, 0 markdown

### Cell 0 code

```
import tensorflow as tf
import keras
# from keras_applications.resnet import ResNet50
from keras_applications.inception_v3 import InceptionV3
from sklearn.model_selection import KFold
import tensorflow as tf
import keras
from keras_applications.resnet import ResNet50
from keras.applications import ResNet50, VGG16
from keras.applications.resnet50 import preprocess_input as preprocess_resnet_50
from keras.applications.vgg16 import preprocess_input as preprocess_vgg_16
from keras.layers import GlobalAveragePooling2D, Dense, Activation
from keras.models import Model
from keras.utils import Sequence
from keras import metrics
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import BatchNormalization
#from tensorflow.keras.applications.resnet50 import preprocess_input
#from tensorflow.keras.applications import ResNet50
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import *
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras_applications.inception_v3 import InceptionV3
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.utils import np_utils
from keras import optimizers
from skimage.filters.rank import median
```

### Cell 1 code

```
def read_and_prep_images(img_paths, img_size=(224,224), pixels_out=True, folder=train_images_dir, display_mode=0):
        img_array.append(resize(apply_mask(get_pixels_hu(img_paths[i], folder=folder),-150,2000,display_mode=display_mode),(img_size[0],img_size[1])))
def get_ds(img_name, folder=train_images_dir):
    pydicom_filedataset = pydicom.read_file(os.path.join(folder, img_name))
def get_pixel_array(img_name, folder=train_images_dir):
    pydicom_filedataset = pydicom.read_file(os.path.join(folder, img_name))
def get_pixels_hu(img_name, folder=train_images_dir):
        image = get_pixel_array(img_name, folder=folder)
        dataset = get_ds(img_name, folder=folder)
    #thresh_img = np.where((img - 1000) / 150 < 1,1.0,0.0)  # threshold the image
    thresh_img = np.where(filter_hu_bounds,1.0,0.0)  # threshold the image
        ax[0, 1].set_title("Threshold")
```

### Cell 2 code

```
class DataGenerator(keras.utils.Sequence):
                X[i,] = read_and_prep_images([ID+".dcm"], self.img_size,pixels_out=True,folder=self.img_dir)
```

### Cell 3 code

```
from keras import backend as K
def weighted_log_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, eps, 1.0-eps)
def _normalized_weighted_average(arr, weights=None):
    A simple Keras implementation that mimics that of
def weighted_loss(y_true, y_pred):
    Will be used as the metric in model.compile()
    Similar to the custom loss function 'weighted_log_loss()' above
    to the official competition metric:
        sklearn.metrics.log_loss with sample weights
    y_pred = K.clip(y_pred, eps, 1.0-eps)
    loss_samples = _normalized_weighted_average(loss, class_weights)
def weighted_log_loss_metric(trues, preds):
    of the validation set in PredictionCheckpoint()
    preds = np.clip(preds, epsilon, 1-epsilon)
```

### Cell 4 code

```
class PredictionCheckpoint(keras.callbacks.Callback):
    def __init__(self, test_df, valid_df,
                 valid_images_dir=train_images_dir,
        self.valid_df = valid_df
        self.valid_images_dir = valid_images_dir
        self.valid_predictions = []
#         self.valid_predictions.append(
#                 DataGenerator(self.valid_df.index, None, self.batch_size, self.input_size, self.valid_images_dir), verbose=2)[:len(self.valid_df)])
#         print("validation loss: %.4f" %
#               weighted_log_loss_metric(self.valid_df.values,
#                                    np.average(self.valid_predictions, axis=0,
#                                               weights=[2**i for i in range(len(self.valid_predictions))])))
```

### Cell 5 code

```
                             backend = keras.backend, layers = keras.layers,
                             models = keras.models, utils = keras.utils)
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(engine.output)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(keras.backend.int_shape(x)[1], activation="relu", name="dense_hidden_1")(x)
        x = keras.layers.Dropout(0.1)(x)
        out = keras.layers.Dense(6, activation="sigmoid", name='dense_output')(x)
        self.model = keras.models.Model(inputs=engine.input, outputs=out)
        self.model.compile(loss=weighted_log_loss, optimizer=keras.optimizers.Adam(), metrics=[weighted_loss])
        train_idx, valid_idx = next(ss)
        model.fit_and_predict(sample_df.iloc[train_idx], sample_df.iloc[valid_idx], test_df_sub)
    def fit_and_predict(self, train_df, valid_df, test_df):
        pred_history = PredictionCheckpoint(test_df, valid_df, input_size=self.input_dims)
        checkpointer = keras.callbacks.ModelCheckpoint(filepath='%s-{epoch:02d}.hdf5' % self.engine.__name__, verbose=1, save_weights_only=True, save_best_only=False)
        scheduler = keras.callbacks.LearningRateScheduler(lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))
        print('scheduler done')
            #callbacks=[scheduler]
            callbacks=[pred_history, scheduler]
```

### Cell 6 code

```
def read_testset(filename="../input/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv"):
    df = pd.read_csv(filename)
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    df = pd.read_csv(filename)
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
```

### Cell 12 code

```
# obtain test + validation predictions (history.test_predictions, history.valid_predictions)
    #history = model.fit_and_predict(train_df.iloc[train_idx], train_df.iloc[valid_idx], test_df)
```

### Cell 17 code

```
test_df.iloc[:, :] = np.average(history.test_predictions, axis=0, weights=[0, 2, 4, 6]) # let's do a weighted average for epochs (>1)
test_df = test_df.stack().reset_index()
```

### Cell 18 code

```
test_df.to_csv('submission_with_history.csv', index=False)
```

## 10pct rank 62 quality strong score 67

File: `Medical/rsna-intracranial-hemorrhage-detection/rank62_10pct/final-cnn-kernel.ipynb`
Cells: 19 total, 19 code, 0 markdown

### Cell 0 code

```
import tensorflow as tf
import keras
# from keras_applications.resnet import ResNet50
from keras_applications.inception_v3 import InceptionV3
from sklearn.model_selection import KFold
import tensorflow as tf
import keras
from keras_applications.resnet import ResNet50
from keras.applications import ResNet50, VGG16
from keras.applications.resnet50 import preprocess_input as preprocess_resnet_50
from keras.applications.vgg16 import preprocess_input as preprocess_vgg_16
from keras.layers import GlobalAveragePooling2D, Dense, Activation
from keras.models import Model
from keras.utils import Sequence
from keras import metrics
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import BatchNormalization
#from tensorflow.keras.applications.resnet50 import preprocess_input
#from tensorflow.keras.applications import ResNet50
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import *
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras_applications.inception_v3 import InceptionV3
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.utils import np_utils
from keras import optimizers
from skimage.filters.rank import median
```

### Cell 1 code

```
def read_and_prep_images(img_paths, img_size=(224,224), pixels_out=True, folder=train_images_dir, display_mode=0):
        img_array.append(resize(apply_mask(get_pixels_hu(img_paths[i], folder=folder),-150,2000,display_mode=display_mode),(img_size[0],img_size[1])))
def get_ds(img_name, folder=train_images_dir):
    pydicom_filedataset = pydicom.read_file(os.path.join(folder, img_name))
def get_pixel_array(img_name, folder=train_images_dir):
    pydicom_filedataset = pydicom.read_file(os.path.join(folder, img_name))
def get_pixels_hu(img_name, folder=train_images_dir):
        image = get_pixel_array(img_name, folder=folder)
        dataset = get_ds(img_name, folder=folder)
    #thresh_img = np.where((img - 1000) / 150 < 1,1.0,0.0)  # threshold the image
    thresh_img = np.where(filter_hu_bounds,1.0,0.0)  # threshold the image
        ax[0, 1].set_title("Threshold")
```

### Cell 2 code

```
class DataGenerator(keras.utils.Sequence):
                X[i,] = read_and_prep_images([ID+".dcm"], self.img_size,pixels_out=True,folder=self.img_dir)
```

### Cell 3 code

```
from keras import backend as K
def weighted_log_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, eps, 1.0-eps)
def _normalized_weighted_average(arr, weights=None):
    A simple Keras implementation that mimics that of
def weighted_loss(y_true, y_pred):
    Will be used as the metric in model.compile()
    Similar to the custom loss function 'weighted_log_loss()' above
    to the official competition metric:
        sklearn.metrics.log_loss with sample weights
    y_pred = K.clip(y_pred, eps, 1.0-eps)
    loss_samples = _normalized_weighted_average(loss, class_weights)
def weighted_log_loss_metric(trues, preds):
    of the validation set in PredictionCheckpoint()
    preds = np.clip(preds, epsilon, 1-epsilon)
```

### Cell 4 code

```
class PredictionCheckpoint(keras.callbacks.Callback):
    def __init__(self, test_df, valid_df,
                 valid_images_dir=train_images_dir,
        self.valid_df = valid_df
        self.valid_images_dir = valid_images_dir
        self.valid_predictions = []
#         self.valid_predictions.append(
#                 DataGenerator(self.valid_df.index, None, self.batch_size, self.input_size, self.valid_images_dir), verbose=2)[:len(self.valid_df)])
#         print("validation loss: %.4f" %
#               weighted_log_loss_metric(self.valid_df.values,
#                                    np.average(self.valid_predictions, axis=0,
#                                               weights=[2**i for i in range(len(self.valid_predictions))])))
```

### Cell 5 code

```
                             backend = keras.backend, layers = keras.layers,
                             models = keras.models, utils = keras.utils)
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(engine.output)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(keras.backend.int_shape(x)[1], activation="relu", name="dense_hidden_1")(x)
        x = keras.layers.Dropout(0.1)(x)
        out = keras.layers.Dense(6, activation="sigmoid", name='dense_output')(x)
        self.model = keras.models.Model(inputs=engine.input, outputs=out)
        self.model.compile(loss=weighted_log_loss, optimizer=keras.optimizers.Adam(), metrics=[weighted_loss])
        train_idx, valid_idx = next(ss)
        model.fit_and_predict(sample_df.iloc[train_idx], sample_df.iloc[valid_idx], test_df_sub)
    def fit_and_predict(self, train_df, valid_df, test_df):
        pred_history = PredictionCheckpoint(test_df, valid_df, input_size=self.input_dims)
        checkpointer = keras.callbacks.ModelCheckpoint(filepath='%s-{epoch:02d}.hdf5' % self.engine.__name__, verbose=1, save_weights_only=True, save_best_only=False)
        scheduler = keras.callbacks.LearningRateScheduler(lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))
        print('scheduler done')
            #callbacks=[scheduler]
            callbacks=[pred_history, scheduler]
```

### Cell 6 code

```
def read_testset(filename="../input/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv"):
    df = pd.read_csv(filename)
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    df = pd.read_csv(filename)
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
```

### Cell 12 code

```
# obtain test + validation predictions (history.test_predictions, history.valid_predictions)
    #history = model.fit_and_predict(train_df.iloc[train_idx], train_df.iloc[valid_idx], test_df)
```

### Cell 17 code

```
test_df.iloc[:, :] = np.average(history.test_predictions, axis=0, weights=[0, 2, 4, 6]) # let's do a weighted average for epochs (>1)
test_df = test_df.stack().reset_index()
```

### Cell 18 code

```
test_df.to_csv('submission_with_history.csv', index=False)
```

## 1st rank 62 quality strong score 72

File: `Medical/rsna-intracranial-hemorrhage-detection/rank62_1st/final-cnn-kernel.ipynb`
Cells: 19 total, 19 code, 0 markdown

### Cell 0 code

```
import tensorflow as tf
import keras
# from keras_applications.resnet import ResNet50
from keras_applications.inception_v3 import InceptionV3
from sklearn.model_selection import KFold
import tensorflow as tf
import keras
from keras_applications.resnet import ResNet50
from keras.applications import ResNet50, VGG16
from keras.applications.resnet50 import preprocess_input as preprocess_resnet_50
from keras.applications.vgg16 import preprocess_input as preprocess_vgg_16
from keras.layers import GlobalAveragePooling2D, Dense, Activation
from keras.models import Model
from keras.utils import Sequence
from keras import metrics
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import BatchNormalization
#from tensorflow.keras.applications.resnet50 import preprocess_input
#from tensorflow.keras.applications import ResNet50
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import *
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras_applications.inception_v3 import InceptionV3
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.utils import np_utils
from keras import optimizers
from skimage.filters.rank import median
```

### Cell 1 code

```
def read_and_prep_images(img_paths, img_size=(224,224), pixels_out=True, folder=train_images_dir, display_mode=0):
        img_array.append(resize(apply_mask(get_pixels_hu(img_paths[i], folder=folder),-150,2000,display_mode=display_mode),(img_size[0],img_size[1])))
def get_ds(img_name, folder=train_images_dir):
    pydicom_filedataset = pydicom.read_file(os.path.join(folder, img_name))
def get_pixel_array(img_name, folder=train_images_dir):
    pydicom_filedataset = pydicom.read_file(os.path.join(folder, img_name))
def get_pixels_hu(img_name, folder=train_images_dir):
        image = get_pixel_array(img_name, folder=folder)
        dataset = get_ds(img_name, folder=folder)
    #thresh_img = np.where((img - 1000) / 150 < 1,1.0,0.0)  # threshold the image
    thresh_img = np.where(filter_hu_bounds,1.0,0.0)  # threshold the image
        ax[0, 1].set_title("Threshold")
```

### Cell 2 code

```
class DataGenerator(keras.utils.Sequence):
                X[i,] = read_and_prep_images([ID+".dcm"], self.img_size,pixels_out=True,folder=self.img_dir)
```

### Cell 3 code

```
from keras import backend as K
def weighted_log_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, eps, 1.0-eps)
def _normalized_weighted_average(arr, weights=None):
    A simple Keras implementation that mimics that of
def weighted_loss(y_true, y_pred):
    Will be used as the metric in model.compile()
    Similar to the custom loss function 'weighted_log_loss()' above
    to the official competition metric:
        sklearn.metrics.log_loss with sample weights
    y_pred = K.clip(y_pred, eps, 1.0-eps)
    loss_samples = _normalized_weighted_average(loss, class_weights)
def weighted_log_loss_metric(trues, preds):
    of the validation set in PredictionCheckpoint()
    preds = np.clip(preds, epsilon, 1-epsilon)
```

### Cell 4 code

```
class PredictionCheckpoint(keras.callbacks.Callback):
    def __init__(self, test_df, valid_df,
                 valid_images_dir=train_images_dir,
        self.valid_df = valid_df
        self.valid_images_dir = valid_images_dir
        self.valid_predictions = []
#         self.valid_predictions.append(
#                 DataGenerator(self.valid_df.index, None, self.batch_size, self.input_size, self.valid_images_dir), verbose=2)[:len(self.valid_df)])
#         print("validation loss: %.4f" %
#               weighted_log_loss_metric(self.valid_df.values,
#                                    np.average(self.valid_predictions, axis=0,
#                                               weights=[2**i for i in range(len(self.valid_predictions))])))
```

### Cell 5 code

```
                             backend = keras.backend, layers = keras.layers,
                             models = keras.models, utils = keras.utils)
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(engine.output)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(keras.backend.int_shape(x)[1], activation="relu", name="dense_hidden_1")(x)
        x = keras.layers.Dropout(0.1)(x)
        out = keras.layers.Dense(6, activation="sigmoid", name='dense_output')(x)
        self.model = keras.models.Model(inputs=engine.input, outputs=out)
        self.model.compile(loss=weighted_log_loss, optimizer=keras.optimizers.Adam(), metrics=[weighted_loss])
        train_idx, valid_idx = next(ss)
        model.fit_and_predict(sample_df.iloc[train_idx], sample_df.iloc[valid_idx], test_df_sub)
    def fit_and_predict(self, train_df, valid_df, test_df):
        pred_history = PredictionCheckpoint(test_df, valid_df, input_size=self.input_dims)
        checkpointer = keras.callbacks.ModelCheckpoint(filepath='%s-{epoch:02d}.hdf5' % self.engine.__name__, verbose=1, save_weights_only=True, save_best_only=False)
        scheduler = keras.callbacks.LearningRateScheduler(lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))
        print('scheduler done')
            #callbacks=[scheduler]
            callbacks=[pred_history, scheduler]
```

### Cell 6 code

```
def read_testset(filename="../input/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv"):
    df = pd.read_csv(filename)
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    df = pd.read_csv(filename)
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
```

### Cell 12 code

```
# obtain test + validation predictions (history.test_predictions, history.valid_predictions)
    #history = model.fit_and_predict(train_df.iloc[train_idx], train_df.iloc[valid_idx], test_df)
```

### Cell 17 code

```
test_df.iloc[:, :] = np.average(history.test_predictions, axis=0, weights=[0, 2, 4, 6]) # let's do a weighted average for epochs (>1)
test_df = test_df.stack().reset_index()
```

### Cell 18 code

```
test_df.to_csv('submission_with_history.csv', index=False)
```
