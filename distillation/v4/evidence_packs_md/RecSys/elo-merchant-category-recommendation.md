# RecSys / elo-merchant-category-recommendation

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 2836 quality usable score 33

File: `RecSys/elo-merchant-category-recommendation/rank2836_70pct/elo-merchant-category-recommendation.ipynb`
Cells: 27 total, 13 code, 14 markdown

### Cell 1 markdown

```
## Contents
- [Introduction](#introduction)
    * [Problem Statement](#problem-statement)
- [Exploratory Data Analysis](#eda)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
    * [Linear Regression](#linear-regression)
    * [Gradient Bososted Tree](#gradient-boosted-trees)
- [Model Analysis](#model-analysis)
- [Conclusion](#conclusion)


```

### Cell 3 markdown

```

Elo, one of Brazil's largest payment brands, is partnered with many merchants to offer promotions and discounts to their cardholders. Elo aimed to reduce marketting that is irrelevant to members and offer them custom-tailored promotions, thereby providing an enjoyable experience and beneficial service. To that end, Elo launched a Kaggle competition, enlisting the Kaggle community's help, to produce a machine learning model that can find signal between trasaction data and loyalty. Such a model will help Elo gauge customer loyalty and how promotional strategies affect it.

**The data provided is simulated and fictitious. It does not contain real customer data.*
```

### Cell 4 markdown

```
Build a machine learning model that can effectively predict customer loyalty scores using trasaction data.
```

### Cell 6 markdown

```
**train data** table contained card_ids, loyalty scores and 3 arbitrary features provided by ELO. The aribtrary features were not very useful as they did not provide much signal in predicting loyalty scores.
**test data** table contained the same arbitrary features as **train data** and card_id but did not contain loyalty scores.
```

### Cell 7 code

```
traindf = pd.read_csv('../input/elo-merchant-category-recommendation/train.csv')
# given test data with no loyaltly score
giventestdf = pd.read_csv('../input/elo-merchant-category-recommendation/test.csv')
histtransdf = pd.read_csv('../input/elo-merchant-category-recommendation/historical_transactions.csv')
newtransdf = pd.read_csv('../input/elo-merchant-category-recommendation/new_merchant_transactions.csv')
merchdf = pd.read_csv('../input/elo-merchant-category-recommendation/merchants.csv')
```

### Cell 15 markdown

```
There are 123,623 card ids in the test data. This data cannot be used to train the model as it does not have loyalty scores (the response variable)
```

### Cell 16 code

```
# plotting loyalty score distribution
plt.xlabel('Loyalty Score')
plt.title('Loyalty Score Distribution');
```

### Cell 17 markdown

```
Loyalty scores are normally distriubted ranging from -10 to 10. There are some outliers at -33.
```

### Cell 18 markdown

```
## Feature Engineering<a class='anchor' id='feature-engineering'></a>
```

### Cell 22 code

```
trx.authorized_flag = trx.authorized_flag.apply(lambda x: True if x == 'Y' else False)
```

### Cell 23 markdown

```
## Model Building<a class='anchor' id='model-building'></a>
```

## 40pct rank 1647 quality strong score 73

File: `RecSys/elo-merchant-category-recommendation/rank1647_40pct/3-first-pca-components-explain-99-99-of-variance.ipynb`
Cells: 21 total, 15 code, 6 markdown

### Cell 0 markdown

```
## Goal of the Notebook
After looking at the various kernels by fellow Kagglers I came to realize that incresign complexity was only marginally improving RMSE (at least from available kernels). So I thought of running a PCA to perform a dimensionality check. I soon came to realize that most of the available data is just noise: the three first components of PCA account for 99.99% of the variance as we will see below. I also used these three components as added features and re-run my model but to no avail. I hope you find this analysis useful and you come to new ideas of why this might be happening.
```

### Cell 1 code

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
```

### Cell 2 code

```
df_train = pd.read_csv('../input/train.csv')
df_history = pd.read_csv("../input/historical_transactions.csv")
```

### Cell 3 code

```
df_history['authorized_flag'] = df_history['authorized_flag'].map({'Y':1, 'N':0})
```

### Cell 4 code

```
        'authorized_flag': ['mean'],
        'month_lag': ['min', 'max'],
agg_history = df_history.groupby(['card_id']).agg(agg_func)
```

### Cell 5 code

```
#Merge with train and test
df_train_all = pd.merge(df_train, agg_history, on='card_id', how='left')
```

### Cell 7 markdown

```
First lets train a model to check its performance using the original data.
```

### Cell 8 code

```
#Train LightGBM model on original data
         "metric": 'rmse',
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof_normal = np.zeros(len(df_train_all))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x.values, train_y)):
    print("fold n°{}".format(fold_))
                    valid_sets=[trn_data, val_data],
                    early_stopping_rounds=200)
    oof_normal[val_idx] = clf.predict(train_x.iloc[val_idx][features], num_iteration=clf.best_iteration)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    predictions_normal += clf.predict(test_x[features], num_iteration=clf.best_iteration) / folds.n_splits
```

### Cell 14 markdown

```
Now let's train a LightGBM model using only these 3 components.
```

### Cell 15 code

```
#Train LightGBM model on these 3 PCA components only
         "metric": 'rmse',
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof_pca = np.zeros(len(pca_train_x))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(pca_train_x.values, train_y)):
    print("fold n°{}".format(fold_))
                    valid_sets=[trn_data, val_data],
                    early_stopping_rounds=200)
    oof_pca[val_idx] = clf.predict(pca_train_x.iloc[val_idx][features], num_iteration=clf.best_iteration)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    predictions_pca += clf.predict(pca_test_x[features], num_iteration=clf.best_iteration) / folds.n_splits
```

### Cell 17 markdown

```
We see that the loss is minimal. Maybe not minimal for a Kaggle competition but a) in real life situations we would indeed opt for the three new features vs original 29 and b) the 3-feature model is not fine-tuned thus the difference may well be even smaller than that.
```

### Cell 18 markdown

```
Let's now re-run our initial model after adding the three PCA components as features and see what we get.
```

## 20pct rank 910 quality strong score 61

File: `RecSys/elo-merchant-category-recommendation/rank910_20pct/ts-7-survival-analysis.ipynb`
Cells: 55 total, 32 code, 23 markdown

### Cell 0 markdown

```
* [Part 10: Validation methods for time series](https://www.kaggle.com/code/konradb/ts-10-validation-methods-for-time-series/)
```

### Cell 6 markdown

```
We start with a theoretical intro and proceed with the well-trodden path for modeling: time-invariant $\rightarrow$ linear model $\rightarrow$ trees (and DL is possible ofc, but *outside the scope* of this notebook ;-)

<a id="section-one"></a>
# Crash intro to survival analysis

So far we have mostly seen time series where we observed the past, we knew what happened / when - predictions are based on that info. That's not always the case:

**Q1**: Do we need to act before we observe all of the data?

**Q2**: Does time to event matter?
```

### Cell 8 markdown

```
- summary 1: quantify / visualise distribution of durations $\rightarrow$ expectations / baselines / thresholds
![0_5WSuIWs_UswmAn2J.png](attachment:68778e99-9ed4-48ff-9075-dfba22958d6f.png)
```

### Cell 9 markdown

```
![surv3.png](attachment:7bd4a636-382a-4fa3-adcd-c5defb83cb0b.png)
![surv4.png](attachment:95bce2d6-fa13-4dd6-b971-5cba99a74cc9.png)
![surv2.png](attachment:17b4fa0c-744a-4807-b4df-caf4d037aa9d.png)
![surv1.png](attachment:9b20842c-4b7b-42e3-8af9-08257320e12a.png)
```

### Cell 12 code

```
df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
```

### Cell 13 code

```
# we only need time-to-event and the churn flag for the basic model
```

### Cell 14 markdown

```
Kaplan-Meier estimate can be used like this to get a general idea over the population:

- estimated $\hat{S}$ is a stepwise function of overall population

- x axis: tenure in months, y axis: probability the customer has **not** churned up that point in time

- confidence bands: [Greenwood’s exponential formula](https://www.stat.berkeley.edu/~freedman/greenwd.pdf)

What if we'd like to get more insight into behaviour per subset of the population (= level of a categorical feature)?
```

### Cell 15 code

```
    flag = df['PaymentMethod'] == payment_method
    kmf.fit(T[flag], event_observed = E[flag], label = payment_method)
```

### Cell 17 code

```
from lifelines.statistics import logrank_test, pairwise_logrank_test
credit_card_flag = df['PaymentMethod'] == 'Credit card (automatic)'
bank_transfer_flag = df['PaymentMethod'] == 'Bank transfer (automatic)'
results = logrank_test(T[credit_card_flag],
                       T[bank_transfer_flag],
                       E[credit_card_flag],
                       E[bank_transfer_flag])
```

### Cell 19 code

```
results = pairwise_logrank_test(df['tenure'], df['PaymentMethod'], df['churn'])
```

### Cell 23 code

```
    flag = df['PaymentMethod'] == payment_method
    naf.fit(T[flag], event_observed=E[flag], label=payment_method)
```

### Cell 26 markdown

```
Performance of a Cox model $\rightarrow$ Concordance index = for a survival model C-index $\approx$ weighted average of the area under time-specific ROC.
- only evaluates the rank: does model predict same order of churn as really happened?
```

## 10pct rank 146 quality usable score 27

File: `RecSys/elo-merchant-category-recommendation/rank146_10pct/elo-deep-automl.ipynb`
Cells: 14 total, 14 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```

### Cell 1 code

```
train = pd.read_csv("../input/mitsuru-features-only-train-and-test/train_df_noindex.csv")
test = pd.read_csv("../input/mitsuru-features-only-train-and-test/test_df_noindex.csv")
```

### Cell 7 code

```
keras_model_type =  "auto" ## always try "fast" first, then "fast2", "auto", etc.
### always set early_stopping to True first and then change it to False
model_options = {'nlp_char_limit':50, 'cat_feat_cross_flag':False,
keras_options = {"patience":10, 'class_weight': True, 'early_stopping': True,
                 'lr_scheduler': '', "optimizer": 'RMS'}
```

### Cell 9 code

```
model, cat_vocab_dict = deepauto.fit(train, target, keras_model_type=keras_model_type,
                                     project_name=project_name, keras_options=keras_options,
                                     model_options=model_options, save_model_flag=False, use_my_model='',
```

### Cell 10 code

```
                                 keras_model_type=keras_model_type,
```

### Cell 12 code

```
sample_submission = pd.read_csv('/kaggle/input/elo-merchant-category-recommendation/sample_submission.csv')
```

### Cell 13 code

```
sample_submission['target'] = y_preds
sample_submission.to_csv('submission_1.csv', index=False)
```

## 1st rank 7 quality strong score 48

File: `RecSys/elo-merchant-category-recommendation/rank7_1st/elo-federated-learning-data-etl.ipynb`
Cells: 58 total, 53 code, 5 markdown

### Cell 5 code

```
df_train = pd.read_csv('../input/elo-merchant-category-recommendation/train.csv')
#df_test = pd.read_csv('../input/test.csv')
df_hist_trans = pd.read_csv('../input/elo-merchant-category-recommendation/historical_transactions.csv')
df_new_merchant_trans = pd.read_csv('../input/elo-merchant-category-recommendation/new_merchant_transactions.csv')
```

### Cell 6 code

```
test_df = pd.read_csv('../input/elofederatedlearningdataetltraintest/horizontalsplit-0-0-lower_table.table.csv')
```

### Cell 14 code

```
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['month_diff'] += df['month_lag']
```

### Cell 15 code

```
aggs['month_lag'] = ['max','min','mean','var']
aggs['authorized_flag'] = ['sum', 'mean']
    df_hist_trans[col+'_mean'] = df_hist_trans.groupby([col])['purchase_amount'].transform('mean')
df_hist_trans_group = df_hist_trans.groupby('card_id').agg(aggs)
df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')
#df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')
```

### Cell 16 code

```
aggs['month_lag'] = ['max','min','mean','var']
    df_new_merchant_trans[col+'_mean'] = df_new_merchant_trans.groupby([col])['purchase_amount'].transform('mean')
df_hist_trans_group = df_new_merchant_trans.groupby('card_id').agg(aggs)
df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')
#df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')
```

### Cell 22 code

```
df_train = df_train.merge(test_df,on='card_id',how='left')
```

### Cell 26 code

```
df_train.to_csv('elo-ETL-data.csv')
```

### Cell 27 code

```
df_train[df_train['test']==1].drop(['test'],axis=1).to_csv('elo-ETL-data-test.csv')
df_train[df_train['test']!=1].drop(['test'],axis=1).to_csv('elo-ETL-data-train.csv')
```

### Cell 37 code

```
fed_60_v.to_csv('elo-ETL-data-60-vertical.csv')
```

### Cell 38 code

```
fed_60_v_test.to_csv('elo-ETL-data-60-vertical-test.csv')
#fed_60_v_test.to_csv('elo-ETL-data-60-vertical-test-x.csv')
```

### Cell 40 code

```
fed_60_v_train.to_csv('elo-ETL-data-60-vertical-train.csv')
#fed_60_v_train.to_csv('elo-ETL-data-60-vertical-train-x.csv')
```

### Cell 42 code

```
fed_40_v.to_csv('elo-ETL-data-40-vertical.csv')
```
