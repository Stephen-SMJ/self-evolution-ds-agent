# Time-Series / web-traffic-time-series-forecasting

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 268 quality reject score -5

File: `Time-Series/web-traffic-time-series-forecasting/rank268_70pct/calculate-smape-with-submission-and-solution.py`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
    ground_truth_df = pd.read_csv(ground_truth_file)
    predction_df = pd.read_csv(prediction_file)
```

## 40pct rank 152 quality strong score 53

File: `Time-Series/web-traffic-time-series-forecasting/rank152_40pct/beginning-web-traffic-time-series-forecasting.ipynb`
Cells: 13 total, 9 code, 4 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
```

### Cell 1 code

```
train = pd.read_csv("../input/train_1.csv")
test = pd.read_csv("../input/key_1.csv")
```

### Cell 2 markdown

```
First submission with the worst score expected
```

### Cell 3 code

```
#dfTest[['Id','Visits']].to_csv('first_submit.csv', index=False)
```

### Cell 5 code

```
#dfTest2 = dfTest2.merge(dfTrain2[['Page','Visits']], how='left')
#dfTest2[['Id','Visits']].to_csv('mean_submit.csv', index=False)
```

### Cell 7 code

```
#dfTest3 = dfTest3.merge(dfTrain3[['Page','Visits']], how='left')
#dfTest3[['Id','Visits']].to_csv('median_submit.csv', index=False)
```

### Cell 11 code

```
    #fill outliers that are out of 1.5*std with rolling median of 56 days
    score = smape_fast(y_truth, y_forecasted)
    print(score)
```

## 20pct rank 47 quality usable score 27

File: `Time-Series/web-traffic-time-series-forecasting/rank47_20pct/top-50-silver-benchmark.ipynb`
Cells: 8 total, 4 code, 4 markdown

### Cell 1 code

```
YEAR_SHIFT = 364 #number of days in a year, use multiple of 7 to be able to capture week behavior
```

### Cell 3 code

```
train = pd.read_csv("../input/train_2.csv")
train = pd.melt(train[list(train.columns[-(YEAR_SHIFT + 2*PERIOD):])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
train = train.groupby(['Page'])["Visits"].apply(lambda x: list(x))
```

### Cell 5 code

```
                s = np.hstack([array[:, (i-1)%7], array[:, i], array[:, (i+1)%7]]).reshape(-1)
```

### Cell 7 code

```
test = pd.read_csv("../input/key_2.csv")
#test[['Id','Visits']].to_csv('submission.csv', index=False)
```

## 10pct rank 47 quality usable score 27

File: `Time-Series/web-traffic-time-series-forecasting/rank47_10pct/top-50-silver-benchmark.ipynb`
Cells: 8 total, 4 code, 4 markdown

### Cell 1 code

```
YEAR_SHIFT = 364 #number of days in a year, use multiple of 7 to be able to capture week behavior
```

### Cell 3 code

```
train = pd.read_csv("../input/train_2.csv")
train = pd.melt(train[list(train.columns[-(YEAR_SHIFT + 2*PERIOD):])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
train = train.groupby(['Page'])["Visits"].apply(lambda x: list(x))
```

### Cell 5 code

```
                s = np.hstack([array[:, (i-1)%7], array[:, i], array[:, (i+1)%7]]).reshape(-1)
```

### Cell 7 code

```
test = pd.read_csv("../input/key_2.csv")
#test[['Id','Visits']].to_csv('submission.csv', index=False)
```

## 1st rank 2 quality strong score 66

File: `Time-Series/web-traffic-time-series-forecasting/rank2_1st/better-smape-for-out-of-box-l2-regression.ipynb`
Cells: 36 total, 17 code, 19 markdown

### Cell 0 markdown

```
The purpose of this notebook is to show some simple transformations that allow out of box non-parametric L2 regressors to achieve better SMAPE scores. By default, these algorithms are not optimizing for a better SMAPE score. However, by changing the target variables by an invertible transformation, we can get these algorithms to get closer to optimizing SMAPE. However a transformation alone won't make the algorithm actually optimize on SMAPE. Furthermore, fitting on transformed target variables will change a parametric model (e.g. linear regression), but a non-parametric model (e.g. decision tree or neural network) will be okay as it has no strict structure.
```

### Cell 1 code

```
from sklearn.model_selection import train_test_split # We will do a simple train, validate, test split.
```

### Cell 4 markdown

```
From the graph we see that when the true values are non-negative (as is the case in our data), we have the following:
* SMAPE punishes underpredictions more than over predictions.
* As the prediction gets closer closer to 0, the SMAPE error approaches 1.
* For over predicitions, as the size of the prediction gets really large, the SMAPE error again approaches 1.

We use these properties to guide us towards finding some transformation functions z = f(y) that will let an L2 regression for target z values more closely emulate minimizing SMAPE.

# Get the Data
```

### Cell 5 code

```
all_df = pd.read_csv('../input/train_1.csv')
```

### Cell 8 markdown

```
# Dividing the data

The competition has us predicting values 64 days into the future. To measure the accuracy of our models, we will simply let the last 64 days of our training data be the target values Y to predict using the data from the previous days X. Also, this notebook is meant to be as simple as possible, so we will not look at extracting features from the Page data. So we will only be using the time series data.
```

### Cell 10 markdown

```
For simplicity, we will split our data into a training set, a validation set, and a test set.
```

### Cell 11 code

```
# First split into test and a combination of training and validation.
X_trainvalid, X_test, Y_trainvalid, Y_test = train_test_split(X_all, Y_all, test_size = 0.33, random_state = 32)
# Now split up the training and validation sets.
X_train, X_valid, Y_train, Y_valid = train_test_split(X_trainvalid, Y_trainvalid, test_size = 0.33, random_state = 35)
print('X_train.shape = ', X_train.shape, '\tX_valid.shape = ', X_valid.shape, '\tX_test.shape = ', X_test.shape)
print('Y_train.shape = ', Y_train.shape, '\tY_valid.shape = ', Y_valid.shape, '\tY_test.shape = ', Y_test.shape)
```

### Cell 15 code

```
model.fit(X_trainvalid, Y_trainvalid)
```

### Cell 16 markdown

```
Now we use simple functions f(Y) to make new target variables Z = f(Y). Then we do regression using the transformed targets Z. To compute the SMAPE score, we invert to get the predicted Y values and then compute the SMAPE score.
For each function f(Y), we will actually search over a hyper-parameter to minimize the SMAPE score.
```

### Cell 18 markdown

```
Now let us try searching for the optimal value of C to minimize the SMAPE score when doing L2 regression on Z. We will do a simple grid search.
Now, sometimes our fit will give a predicted value of 0 while the true value is 0. Our SMAPE function is simple and so it will return a NaN value. Let's bypass this by adding a small value of epsilon to the predicted Y values that will not affect the SMAPE score that much.
```

### Cell 19 code

```
    Z_predict = model.predict(X_valid)
    newsmape = smape(epsilon + Y_predict, Y_valid)
```

### Cell 20 markdown

```
Now let's do a test for param = 240. We will train on all of the training and validation data.
```
