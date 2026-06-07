# Time-Series / g-research-crypto-forecasting

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1328 quality usable score 37

File: `Time-Series/g-research-crypto-forecasting/rank1328_70pct/tree-based-methods.ipynb`
Cells: 12 total, 12 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```

### Cell 1 code

```
combined = pd.read_csv("../input/g-research-crypto-forecasting/train.csv")
```

### Cell 2 code

```
asset_details = pd.read_csv("../input/g-research-crypto-forecasting/asset_details.csv")
```

### Cell 3 code

```
            price.shift(periods=-16) /
            price.shift(periods=-1)
    num = targets.multiply(m.values, axis=0).rolling(3734).mean().values
    denom = m.multiply(m.values, axis=0).rolling(3734).mean().values
        targets["BetaRolling"+asset_names[i]] = beta[i,:]
```

### Cell 4 code

```
        targets=pd.merge(targets,data, how="left",on="datetime")
```

### Cell 6 code

```
features = ["Bitcoin","BetaRollingBitcoin"]
```

### Cell 7 code

```
#lgbm model to predict a regression var
import lightgbm as lgb
param['metric'] = ['l1']
lightModel = lgb.train(param, train_data, num_round, valid_sets=[train_data,test_data], early_stopping_rounds=30)
```

### Cell 9 code

```
       'Maker', 'Dogecoin',"BetaRollingBitcoin"]
```

### Cell 10 code

```
#lgbm model to predict a regression var
import lightgbm as lgb
param['metric'] = ['l1']
lightModel = lgb.train(param, train_data, num_round, valid_sets=[train_data,test_data], early_stopping_rounds=30)
```

## 40pct rank 819 quality weak score 13

File: `Time-Series/g-research-crypto-forecasting/rank819_40pct/random-predictions.ipynb`
Cells: 2 total, 1 code, 1 markdown

### Cell 0 markdown

```
# Random predictions

This dummy submission presents random predictions based on a fitted Gaussian on the target distribution across all assets.
```

### Cell 1 code

```
from scipy.stats import norm
import numpy as np
import gresearch_crypto

target_mean, target_std = (2.40946942906261e-06, 0.004333437997098369)

def random_prediction(size: int) -> np.array:
    return norm.rvs(
        loc = target_mean,
        scale = target_std,
        size = size
    )

env = gresearch_crypto.make_env()
iter_test = env.iter_test()

for i, (df_test, df_pred) in enumerate(iter_test):
    df_pred['Target'] = random_prediction(size = df_pred.shape[0])
    env.predict(df_pred)
```

## 20pct rank 363 quality usable score 37

File: `Time-Series/g-research-crypto-forecasting/rank363_20pct/crypto-candle-analysis.ipynb`
Cells: 40 total, 26 code, 14 markdown

### Cell 3 code

```
asset_details = pd.read_csv(data_path + "asset_details.csv")
```

### Cell 13 code

```
            train_single_new = shift(train_single_new.copy(), ["upper_shadow", "lower_shadow", "real_body", "candle", "body_mean", "open", "close", "high", "low",
# Creating lagged features
def shift(df, columns, lags=[]):
    for lag in lags:
        df = df.merge(df[columns].shift(lag), left_index=True, right_index=True, suffixes=('', f'_lag{lag}'), how="inner")
```

### Cell 16 code

```
    moments["mean"] = df[["open", "high", "low", "close"]].groupby(df.index // N).agg(func=np.mean, axis=0).mean(axis=1)
    moments["std"] = df[["open", "high", "low", "close"]].groupby(df.index // N).agg(func=np.std, axis=0).std(axis=1)
```

### Cell 17 code

```
def calc_metrics(df, pattern, plot_label=[], col_label=[], thr=None):
    # z scores
```

### Cell 18 code

```
def calc_metrics_2(df, pattern, plot_label=[], col_label=[], thr=None):
    # z scores
```

### Cell 22 code

```
# calc_metrics_2(coin_df.reset_index(), "asset_ID", ["upper", "lower"], [[6], [6]])
calc_metrics(coin_df.reset_index(), "asset_ID", [[0, 0], [1, 0], [0, 2]], [[6], [6], [4]])
```

### Cell 25 code

```
# calc_metrics_2(coin_df2, "bull_marubozu", ["upper","lower", "lower left", "upper right", "lower right"], [[6], [6], [6], [6], [6]])
calc_metrics(coin_df2, "bull_marubozu", [[0, 0], [1, 0], [1, 1], [0, 2], [1, 2]], [[6], [6], [6], [6], [6]])
```

### Cell 27 code

```
# calc_metrics_2(coin_df2, "bear_marubozu", ["upper", "lower", "upper left", "lower left"], [[4], [4], [4], [4]])
calc_metrics(coin_df2, "bear_marubozu", [[0, 0], [1, 0], [0, 1], [1, 1]], [[4], [4], [4], [4]])
```

### Cell 29 code

```
# calc_metrics_2(coin_df2, "hammer", ["upper", "lower", "upper right", "lower right"], [[6], [6], [6], [6]])
calc_metrics(coin_df2, "hammer", [[0, 0], [1, 0], [0, 2], [1, 2]], [[6], [6], [6], [6]])
```

### Cell 31 code

```
# calc_metrics_2(coin_df2, "hang_man", ["upper", "lower"], [[0], [6]])
calc_metrics(coin_df2, "hang_man", [[0, 0], [1, 0], [1, 1]], [[0], [6], [6]])
```

### Cell 33 code

```
# calc_metrics_2(coin_df2, "inv_hammer", ["upper", "lower", "upper left", "lower left"], [[4], [4], [4], [5]])
calc_metrics(coin_df2, "inv_hammer", [[0, 0], [1, 0], [0, 1], [1, 1]], [[4], [4], [4], [5]])
```

### Cell 35 code

```
# calc_metrics_2(coin_df2, "shot_star", ["upper","lower", "upper left", "lower left", "upper right", "lower right"], [[5], [5], [6], [5], [4], [5]])
calc_metrics(coin_df2, "shot_star", [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]], [[5], [5], [6], [5], [4], [5]])
```

## 10pct rank 112 quality strong score 49

File: `Time-Series/g-research-crypto-forecasting/rank112_10pct/classic-pipeline.ipynb`
Cells: 8 total, 8 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
```

### Cell 1 code

```
supplemental_train = pd.read_csv("../input/g-research-crypto-forecasting/supplemental_train.csv")
asset_details = pd.read_csv("../input/g-research-crypto-forecasting/asset_details.csv")
ss = pd.read_csv("../input/g-research-crypto-forecasting/example_sample_submission.csv")
example_test=  pd.read_csv("../input/g-research-crypto-forecasting/example_test.csv")
train = pd.read_csv("../input/g-research-crypto-forecasting/train.csv")
```

### Cell 3 code

```
    df = df.groupby("Asset_ID").apply(nans).reset_index(drop = True)
```

### Cell 4 code

```
                df[gap_name] = df[col].shift(periods=gap, fill_value=0)
    df = df.groupby("Asset_ID").apply(get_prev, cols, 1, 2)
```

### Cell 5 code

```
        train = pd.read_csv("../input/g-research-crypto-forecasting/train.csv")
    train = pd.read_csv("../input/g-research-crypto-forecasting/train.csv")
```

## 1st rank 90 quality strong score 64

File: `Time-Series/g-research-crypto-forecasting/rank90_1st/g-research-starter-lgbm-pipeline-lb-0-174.ipynb`
Cells: 15 total, 8 code, 7 markdown

### Cell 0 markdown

```
# 🪙 G-Research Crypto - Starter LGBM Pipeline
### Just a simple pipeline going from zero to a valid submission
We train one `LGBMRegressor` for each asset over a very very naive set of features (the input dataframe `['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']`), we get the predictions correctly using the iterator and we submit. No validation for now, no cross validation... nothing at all lol: just the bare pipeline!
```

### Cell 1 markdown

```
This is a fork of the original notebook. My intention here to show is what can be done with this method to improve your score.
```

### Cell 3 code

```
from lightgbm import LGBMRegressor
```

### Cell 4 code

```
train_df = pd.read_csv(TRAIN_CSV)
asset_df = pd.read_csv(ASSET_DETAILS_CSV)
```

### Cell 7 markdown

```
## Utility functions to train a model for one asset
```

### Cell 8 code

```
    model = LGBMRegressor(n_estimators=1000)
```
