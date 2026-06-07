# Time-Series / ubiquant-market-prediction

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1707 quality strong score 57

File: `Time-Series/ubiquant-market-prediction/rank1707_70pct/beginner-keras-end-to-end-solution-usingnbeats.ipynb`
Cells: 12 total, 8 code, 4 markdown

### Cell 0 markdown

```
Originally I was trying to reimplement the N-Beats paper with keras, though I eventually found out theres an easy to use github repo: https://github.com/philipperemy/n-beats
This notebook is meant to be a very simple guide - quick and easy to run. With no kfold, ensembling and learning rate tuning implemented, it acheived 0.1358 score with only 1 epochs.
To improve the model, it is recommended to try kfold, ensembling, including the investment ID and hyperparameter tuning. It is not an exact replication of the paper - which uses an ensemble of different loss functions and horizon windows. Changing batch size, blocks per stack and number of hidden layers may improve model as it seems to overfit quite quickly.
```

### Cell 1 code

```
import tensorflow as tf
copy_tree("../input/nbeats-keras/nbeats_keras", "./nbeats_keras") # copy into nbeats working directory
from nbeats_keras.model import NBeatsNet as NBeatsKeras
```

### Cell 7 markdown

```
### NBeats Model
```

### Cell 8 code

```
backend = NBeatsKeras(
            stack_types=(NBeatsKeras.GENERIC_BLOCK, NBeatsKeras.GENERIC_BLOCK),
            nb_blocks_per_stack=4, thetas_dim=(4, 4), share_weights_in_stack=True,
c = num_samples // 10 # 10% for validation
```

### Cell 9 code

```
backend.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=1024)
```

## 40pct rank 958 quality strong score 73

File: `Time-Series/ubiquant-market-prediction/rank958_40pct/xgboost-with-clustering-different-model-cluster.ipynb`
Cells: 25 total, 20 code, 5 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
from sklearn.cluster import MeanShift, KMeans
```

### Cell 12 code

```
train.groupby(['cluster']).count()
```

### Cell 14 markdown

```
# 3. Train Models
```

### Cell 15 code

```
    score = np.sqrt(mean_square_distance)
    return score
rmse_score = make_scorer(rmse, greater_is_better = False)
```

### Cell 16 code

```
#     grid = GridSearchCV(SVR(),{'C': [10],'gamma': [1e-8],'epsilon':[0.001],'kernel': ['rbf']}, cv=10, return_train_score=False, verbose = 0,n_jobs=3,scoring=rmse_score)
    model=XGBRegressor()
```

## 20pct rank 443 quality strong score 67

File: `Time-Series/ubiquant-market-prediction/rank443_20pct/regime-classification-with-aggregated-features.ipynb`
Cells: 26 total, 13 code, 13 markdown

### Cell 0 markdown

```
I have previously posted the idea that clustering using the k-means method could be used to identify market regimes. (Link [here](https://www.kaggle.com/code/tmrtj9999/regime-classification-by-k-means/notebook))

To summarize the content of this article, each row is clustered by the k-means method as it is, using the given 300 features, and the classification result is considered as a regime.

When I published this code, someone commented to me that it might be possible to classify regimes by aggregating features for each time_id and clustering each time_id using the aggregate features.

Therefore, I would like to take the average of each feature for each time_id and use the aggregate features for clustering.
```

### Cell 2 code

```
import lightgbm as lgbm
from lightgbm import *
```

### Cell 5 code

```
num_investment = df.groupby('time_id').count()
```

### Cell 10 code

```
df_agg = df.groupby('time_id').mean()
```

### Cell 11 markdown

```
Determine the optimal number of clusters using the elbow method.
```

### Cell 13 markdown

```
The elbow method considers the optimal cluster to be the one where the slope is smooth. In this case, the optimal number of clusters is 4.

Since I know that the optimal number of clusters is 4, I model with n_clusters=4.
```

## 10pct rank 233 quality strong score 73

File: `Time-Series/ubiquant-market-prediction/rank233_10pct/ubiquant-market-prediction-model-ensemble.ipynb`
Cells: 15 total, 11 code, 4 markdown

### Cell 0 markdown

```
Credit : https://www.kaggle.com/code/chengskin/dnn-model-ensemble-of-ubiquant
```

### Cell 1 markdown

```
The majority of the content of this notebook comes from the original author. The contribution that has been made is in the choice of weights in the submission function (the last cell). You can improve it and get a higher score than 0.1554.
```

### Cell 3 code

```
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dot, Reshape, Add, Subtract
from keras import backend as K
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, KFold, GroupKFold
from tensorflow.python.ops import math_ops
```

### Cell 9 code

```
    investment_id_inputs = tf.keras.Input((1, ), dtype=tf.uint16)
    features_inputs = tf.keras.Input((300, ), dtype=tf.float16)
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[investment_id_inputs, features_inputs], outputs=[output])
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss='mse', metrics=['mse', "mae", "mape", rmse])
    investment_id_inputs = tf.keras.Input((1, ), dtype=tf.uint16)
    features_inputs = tf.keras.Input((300, ), dtype=tf.float16)
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[investment_id_inputs, features_inputs], outputs=[output])
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss='mse', metrics=['mse', "mae", "mape", rmse])
    investment_id_inputs = tf.keras.Input((1, ), dtype=tf.uint16)
    features_inputs = tf.keras.Input((300, ), dtype=tf.float32)
    output = tf.keras.layers.BatchNormalization(axis=1)(output)
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[investment_id_inputs, features_inputs], outputs=[output])
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss='mse', metrics=['mse', "mae", "mape", rmse])
    features_inputs = tf.keras.Input((300, ), dtype=tf.float16)
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[features_inputs], outputs=[output])
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss='mse', metrics=['mse', "mae", "mape", rmse])
```

### Cell 10 code

```
    model.load_weights(f'../input/train-dnn-v2-10fold/model_{i}')
```

### Cell 11 code

```
    features_inputs = tf.keras.Input((300, ), dtype=tf.float32)
    output = tf.keras.layers.BatchNormalization(axis=1)(output)
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[features_inputs], outputs=[output])
    model.compile(optimizer=tf.optimizers.Adam(0.001),  loss = correlationLoss, metrics=[correlationMetric])
def correlationMetric(x, y, axis=-2):
  """Metric returning the Pearson correlation coefficient of two tensors over some axis, default -2."""
def correlationMetric_01mse(x, y, axis=-2):
  """Metric returning the Pearson correlation coefficient of two tensors over some axis, default -2."""
# list(GroupKFold(5).split(train , groups = train.index))[0]
def evaluate_metric(valid_df):
    return np.mean(valid_df[['time_id_', 'target', 'preds']].groupby('time_id').apply(pearson_coef))
```

## 1st rank 47 quality strong score 78

File: `Time-Series/ubiquant-market-prediction/rank47_1st/ml-from-the-beginning-to-the-end-for-newbies.ipynb`
Cells: 49 total, 26 code, 23 markdown

### Cell 1 markdown

```
To attempt to predict returns, there are many computer-based algorithms and models for financial market trading. <br>
**Yet,** with new techniques and approaches, **data science could improve quantitative researchers' ability to forecast an investment's return.**

> Ubiquant is committed to creating long-term stable returns for investors.

In this competition, you’ll build **a model that forecasts an investment's return rate**. <br>
Train and test your algorithm on historical prices. Top entries will solve this real-world data science problem with as much accuracy as possible.
```

### Cell 3 markdown

```
**Your challenge is to predict the value of an obfuscated metric relevant for making trading decisions.**
```

### Cell 4 markdown

```
- row_id - A unique identifier for the row.
- time_id - The ID code for the time the data was gathered. The time IDs are in order, but the real time between the time IDs is not constant and will likely be shorter for the final private test set than in the training set.
- investment_id - The ID code for an investment. Not all investment have data in all time IDs.
- **target - The target.**
- [f_0:f_299] - Anonymized features generated from market data.
```

### Cell 7 code

```
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
```

### Cell 18 code

```
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
```

### Cell 25 code

```
time_count = ubiquant['time_id'].groupby(ubiquant['investment_id']).count()
time_mean = ubiquant['time_id'].groupby(ubiquant['investment_id']).mean()
time_std = ubiquant['time_id'].groupby(ubiquant['investment_id']).std()
```

### Cell 29 code

```
investment_count = ubiquant.groupby(['investment_id'])['target'].count()
investment_mean = ubiquant.groupby(['investment_id'])['target'].mean()
```

### Cell 30 markdown

```
# **Sixth. Feature Engineering -🛠**
```

### Cell 35 code

```
investment_count = remove_df.groupby(['investment_id'])['target'].count()
investment_mean = remove_df.groupby(['investment_id'])['target'].mean()
```

### Cell 39 markdown

```
# **Seventh. Modeling & Training -🗡**
```

### Cell 40 code

```
import lightgbm
import xgboost
train_ds = lightgbm.Dataset(train_x, label = train_target)
val_ds = lightgbm.Dataset(test_x, label = test_target)
          'metric': 'mse',
          'is_training_metric': True,
model = lightgbm.train(params, train_ds, 85, val_ds)
```

### Cell 41 code

```
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
```
