# Medical / osic-pulmonary-fibrosis-progression

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1461 quality strong score 67

File: `Medical/osic-pulmonary-fibrosis-progression/rank1461_70pct/pulmonaryfibrosis-environment-51f0d2.ipynb`
Cells: 15 total, 12 code, 3 markdown

### Cell 1 code

```
import tensorflow as tf
import tensorflow.keras.backend as K
from wandb.keras import WandbCallback
import keras
from keras.models import Sequential
from pfutils import (get_test_data, get_train_data, get_pseudo_test_data, get_exponential_decay_lr_callback, TTA_on_test,
                     build_model, get_cosine_annealing_lr_callback, get_fold_indices, DataGenerator, make_lungmask)
                    Laplace_metric, Laplace_log_likelihood, experimental_loss_function)
PSEUDO_TEST_PATIENTS = 0
```

### Cell 2 code

```
    PSEUDO_TEST_PATIENTS = 0
```

### Cell 5 code

```
#Image Flags
DROP_OUT_LAYERS = [] # [0,1,2] voor dropout in de eerste 3 lagen
```

### Cell 6 code

```
# Number of folds. A number between 1 and 176-PSEUDO_TEST_PATIENTS. 176 = 2^4 * 11
FOLDS = 5
```

### Cell 7 code

```
#TTA steps and TTA gaussian multiplier
TTA_STEPS = 1
TTA_MULTIPLIER = 0
LEARNING_RATE_SCHEDULER = 'exp' #'exp', 'cos' or None
MODEL_NAME = "FinalWithoutTTA"
              COSINE_CYCLES = COSINE_CYCLES, MODEL_NAME=MODEL_NAME, LEARNING_RATE_SCHEDULER = LEARNING_RATE_SCHEDULER, PREDICT_SLOPE = PREDICT_SLOPE,
              DROP_OUT_LAYERS = DROP_OUT_LAYERS, BATCH_SIZE = BATCH_SIZE, GAUSSIAN_NOISE_FVC_CORRELATED = GAUSSIAN_NOISE_FVC_CORRELATED, TTA_STEPS = TTA_STEPS,
              APPLY_LUNGMASK = APPLY_LUNGMASK, USE_IMAGES = USE_IMAGES, DIM = DIM, IMG_FEATURES = IMG_FEATURES, EFFNET = EFFNET, TTA_MULTIPLIER = TTA_MULTIPLIER)
```

### Cell 8 code

```
train_data, train_images, train_labels = get_train_data('../input/osic-pulmonary-fibrosis-progression/train.csv', PSEUDO_TEST_PATIENTS, TRAIN_ON_BACKWARD_WEEKS, USE_IMAGES, APPLY_LUNGMASK, DIM)
if PSEUDO_TEST_PATIENTS > 0:
    test_data, test_check = get_pseudo_test_data('../input/osic-pulmonary-fibrosis-progression/train.csv', PSEUDO_TEST_PATIENTS, INPUT_NORMALIZATION)
```

### Cell 9 code

```
tf.keras.utils.plot_model(model)
```

### Cell 11 code

```
fold_pos = get_fold_indices(FOLDS, train_data)
print(fold_pos)
```

### Cell 12 code

```
for fold in range(FOLDS):
    train_ID = list(range(fold_pos[0],fold_pos[fold])) + list(range(fold_pos[fold+1],fold_pos[-1]))
    val_ID = list(range(fold_pos[fold], fold_pos[fold+1]))
    validation_generator = DataGenerator(val_ID, config, validation = True)
    sv = tf.keras.callbacks.ModelCheckpoint(
    'fold-%i.h5'%fold, monitor='val_loss', verbose=0, save_best_only=True,
    if LEARNING_RATE_SCHEDULER == 'exp':
    if LEARNING_RATE_SCHEDULER == 'cos':
    print(fold+1, "of", FOLDS)
        name = MODEL_NAME + '-F{}'.format(fold+1)
        config.update({'fold': fold+1})
    history = model.fit(training_generator, validation_data = validation_generator, epochs = EPOCHS,
    if SUBMIT or PSEUDO_TEST_PATIENTS > 0:
        model.load_weights('fold-%i.h5'%fold)
        TTA_test_data = TTA_on_test(test_data.to_numpy(), config)
        for i in range(TTA_STEPS):
            predictions.append(model.predict(TTA_test_data[:,:,i], batch_size = 256))
```

### Cell 13 code

```
    submission.to_csv("submission.csv", index = False)
```

### Cell 14 code

```
if PSEUDO_TEST_PATIENTS > 0:
        postprocess = np.abs(predictions)
            postprocess[:,:,1] = gmean(postprocess[:,:,1], axis = 0)
            postprocess = np.mean(postprocess, axis = 0)
            postprocess[:,:,1] = np.power(postprocess[:,:,1],i)
            postprocess = np.mean(postprocess, axis = 0)
            postprocess[:,1] = np.power(postprocess[:,1],1/i)
        FVC_pred = postprocess[:,0]
        sigma = postprocess[:,1]
```

## 40pct rank 743 quality strong score 73

File: `Medical/osic-pulmonary-fibrosis-progression/rank743_40pct/pulmonary-fibrosis.ipynb`
Cells: 53 total, 47 code, 6 markdown

### Cell 0 code

```
!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index
!pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index
```

### Cell 2 code

```
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from tensorflow.keras.optimizers import Nadam
```

### Cell 4 code

```
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
```

### Cell 7 code

```
    c = np.vstack([weeks, np.ones(len(weeks))]).T
```

### Cell 10 code

```
from tensorflow.keras.utils import Sequence
```

### Cell 11 code

```
from tensorflow.keras.layers import (
import efficientnet.tfkeras as efn
def get_efficientnet(model, shape):
        'b0': efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False),
        'b1': efn.EfficientNetB1(input_shape=shape,weights=None,include_top=False),
        'b2': efn.EfficientNetB2(input_shape=shape,weights=None,include_top=False),
        'b3': efn.EfficientNetB3(input_shape=shape,weights=None,include_top=False),
        'b4': efn.EfficientNetB4(input_shape=shape,weights=None,include_top=False),
        'b5': efn.EfficientNetB5(input_shape=shape,weights=None,include_top=False),
        'b6': efn.EfficientNetB6(input_shape=shape,weights=None,include_top=False),
        'b7': efn.EfficientNetB7(input_shape=shape,weights=None,include_top=False)
    base = get_efficientnet(model_class, shape)
    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)
```

### Cell 14 code

```
def score(fvc_true, fvc_pred, sigma):
    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)
    return np.mean(metric)
```

### Cell 15 code

```
    metric = []
            m.append(score(fvc_true, fvc, percent))
        metric.append(np.mean(m))
    q = (np.argmin(metric) + 1)/ 10
    sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
    test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
```

### Cell 19 code

```
sub[["Patient_Week","FVC","Confidence"]].to_csv("submission_img.csv", index=False)
```

### Cell 22 code

```
tr = pd.read_csv(f"{ROOT}/train.csv")
chunk = pd.read_csv(f"{ROOT}/test.csv")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")
```

### Cell 25 code

```
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')
```

### Cell 26 code

```
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
```

## 20pct rank 423 quality strong score 73

File: `Medical/osic-pulmonary-fibrosis-progression/rank423_20pct/fibrosis-osic-submission-21.ipynb`
Cells: 53 total, 47 code, 6 markdown

### Cell 1 code

```
!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index
!pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index
```

### Cell 2 code

```
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from tensorflow.keras.optimizers import Nadam
```

### Cell 4 markdown

```
## 3. Download data, auxiliary functions and model tuning <a class="anchor" id="3"></a>

[Back to Table of Contents](#0.1)
```

### Cell 5 code

```
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
```

### Cell 7 code

```
    c = np.vstack([weeks, np.ones(len(weeks))]).T
```

### Cell 9 code

```
from tensorflow.keras.utils import Sequence
```

### Cell 10 code

```
from tensorflow.keras.layers import (
import efficientnet.tfkeras as efn
def get_efficientnet(model, shape):
        'b0': efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False),
        'b1': efn.EfficientNetB1(input_shape=shape,weights=None,include_top=False),
        'b2': efn.EfficientNetB2(input_shape=shape,weights=None,include_top=False),
        'b3': efn.EfficientNetB3(input_shape=shape,weights=None,include_top=False),
        'b4': efn.EfficientNetB4(input_shape=shape,weights=None,include_top=False),
        'b5': efn.EfficientNetB5(input_shape=shape,weights=None,include_top=False),
        'b6': efn.EfficientNetB6(input_shape=shape,weights=None,include_top=False),
        'b7': efn.EfficientNetB7(input_shape=shape,weights=None,include_top=False)
    base = get_efficientnet(model_class, shape)
    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)
```

### Cell 12 code

```
def score(fvc_true, fvc_pred, sigma):
    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)
    return np.mean(metric)
```

### Cell 13 code

```
    metric = []
            m.append(score(fvc_true, fvc, percent))
        metric.append(np.mean(m))
    q = (np.argmin(metric) + 1)/ 10
    sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
    test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
```

### Cell 18 code

```
sub[["Patient_Week","FVC","Confidence"]].to_csv("submission_img.csv", index=False)
```

### Cell 21 code

```
tr = pd.read_csv(f"{ROOT}/train.csv")
chunk = pd.read_csv(f"{ROOT}/test.csv")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")
```

### Cell 24 code

```
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')
```

## 10pct rank 209 quality strong score 73

File: `Medical/osic-pulmonary-fibrosis-progression/rank209_10pct/change-in-dropout-and-mloss-values.ipynb`
Cells: 54 total, 47 code, 7 markdown

### Cell 0 markdown

```
    - Model that uses images can be found at: https://www.kaggle.com/miklgr500/linear-decay-based-on-resnet-cnn
- Michael Kazachok's Linear Decay (based on ResNet CNN)
- Replaced Michael's model with EfficientNets B0, B2, B4
```

### Cell 2 code

```
!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index
!pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index
```

### Cell 3 code

```
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from tensorflow.keras.optimizers import Nadam
```

### Cell 5 code

```
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
```

### Cell 8 code

```
    c = np.vstack([weeks, np.ones(len(weeks))]).T
```

### Cell 11 code

```
from tensorflow.keras.utils import Sequence
```

### Cell 12 code

```
from tensorflow.keras.layers import (
import efficientnet.tfkeras as efn
def get_efficientnet(model, shape):
        'b0': efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False),
        'b1': efn.EfficientNetB1(input_shape=shape,weights=None,include_top=False),
        'b2': efn.EfficientNetB2(input_shape=shape,weights=None,include_top=False),
        'b3': efn.EfficientNetB3(input_shape=shape,weights=None,include_top=False),
        'b4': efn.EfficientNetB4(input_shape=shape,weights=None,include_top=False),
        'b5': efn.EfficientNetB5(input_shape=shape,weights=None,include_top=False),
        'b6': efn.EfficientNetB6(input_shape=shape,weights=None,include_top=False),
        'b7': efn.EfficientNetB7(input_shape=shape,weights=None,include_top=False)
    base = get_efficientnet(model_class, shape)
    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)
```

### Cell 15 code

```
def score(fvc_true, fvc_pred, sigma):
    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)
    return np.mean(metric)
```

### Cell 16 code

```
    metric = []
            m.append(score(fvc_true, fvc, percent))
        metric.append(np.mean(m))
    q = (np.argmin(metric) + 1)/ 10
    sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
    test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
```

### Cell 20 code

```
sub[["Patient_Week","FVC","Confidence"]].to_csv("submission_img.csv", index=False)
```

### Cell 23 code

```
tr = pd.read_csv(f"{ROOT}/train.csv")
chunk = pd.read_csv(f"{ROOT}/test.csv")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")
```

### Cell 26 code

```
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')
```

## 1st rank 567 quality strong score 78

File: `Medical/osic-pulmonary-fibrosis-progression/rank567_1st/dnn-lgbm-ngboost-elasticnet.ipynb`
Cells: 37 total, 35 code, 2 markdown

### Cell 1 code

```
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import make_scorer
import lightgbm as lgb
from tqdm.keras import TqdmCallback
from ngboost.scores import MLE
```

### Cell 2 code

```
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
```

### Cell 5 code

```
tr = pd.read_csv(f"{ROOT}/train.csv")
chunk = pd.read_csv(f"{ROOT}/test.csv")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")
```

### Cell 7 code

```
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')
```

### Cell 8 code

```
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
```

### Cell 9 code

```
data = data.merge(base, on='Patient', how='left')
```

### Cell 19 code

```
    def score(y_true, y_pred):
        metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
        return K.mean(metric)
            return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    model = M.Model(z, preds, name="CNN")
    #model.compile(loss=qloss, optimizer="adam", metrics=[score])
    model.compile(loss=mloss(delta), optimizer=tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False), metrics=[score])
```

### Cell 21 code

```
    Calculates the modified Laplace Log Likelihood score for this competition.
    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
        return metric
        return np.mean(metric)
scorer_210 = make_scorer(lambda y_true, y_pred: laplace_log_likelihood(y_true, y_pred, 210))
```

### Cell 23 code

```
class OSICLossForLGBM:
    Custom Loss for LightGBM.
    * Evaluation: return competition metric
        """Return Loss for lightgbm"""
        """Return Grad and Hess for lightgbm"""
```

### Cell 24 code

```
NFOLD = 5
gkf = GroupKFold(n_splits = NFOLD)
NFOLD_MODELS = NFOLD * MODELS
val_scores = []
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
fold_n = 0
    fold_n += 1
    ngb = NGBoost(Base = default_tree_learner, Dist = Normal, Score=MLE, natural_gradient = True, verbose = False).fit(tr[FE3].iloc[tr_idx].values, y[tr_idx])
            validation_data=(z2[val_idx], y[val_idx]), verbose=0, callbacks = [TqdmCallback(verbose = 0), lr_scheduler])
    'metric': 'None',
        "early_stopping_rounds": 500,
    loss = OSICLossForLGBM()
                       valid_sets = test_data,
    print(f"Loss Keras #{fold_n}: {nn_val_loss}")
    print(f"Loss LGBM #{fold_n}: {lgb_val_loss}, {lgb_val_loss_conf}")
    pe1 += subm_predict / NFOLD_MODELS
```

### Cell 34 code

```
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
```

### Cell 35 code

```
subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)
```
