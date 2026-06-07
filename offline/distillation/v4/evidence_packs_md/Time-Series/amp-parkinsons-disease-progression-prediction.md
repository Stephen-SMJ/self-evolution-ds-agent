# Time-Series / amp-parkinsons-disease-progression-prediction

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1297 quality usable score 29

File: `Time-Series/amp-parkinsons-disease-progression-prediction/rank1297_70pct/predicting-updrs-4.ipynb`
Cells: 17 total, 17 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```

### Cell 2 code

```
train = pd.read_csv("/kaggle/input/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv")
```

### Cell 16 code

```
for (test, test_peptides, test_proteins, sample_submission) in iter_test:
```

## 40pct rank 710 quality usable score 43

File: `Time-Series/amp-parkinsons-disease-progression-prediction/rank710_40pct/uom-cs3111-23-group07-final-submission.ipynb`
Cells: 12 total, 11 code, 1 markdown

### Cell 1 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```

### Cell 3 code

```
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) with an offset of 1.
    metric = np.zeros(len(y_true_plus_1))
    metric[mask_not_zeros] = numerator[mask_not_zeros] / denominator[mask_not_zeros]
    return 100 * np.nanmean(metric)
```

### Cell 4 code

```
train_clinical_data = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv')
supplemental_clinical_data = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/supplemental_clinical_data.csv')
```

### Cell 5 code

```
medians = train_clinical_all.groupby('visit_month')[['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']].median()
```

### Cell 6 code

```
merged_df = pd.merge(train_clinical_all, medians, on='visit_month', suffixes=('_df1', '_df2'))
merged_df
```

### Cell 7 code

```
merged_df['updrs_1'] = merged_df['updrs_1_df1'].fillna(merged_df['updrs_1_df2'])
merged_df['updrs_2'] = merged_df['updrs_2_df1'].fillna(merged_df['updrs_2_df2'])
merged_df['updrs_3'] = merged_df['updrs_3_df1'].fillna(merged_df['updrs_3_df2'])
merged_df['updrs_4'] = merged_df['updrs_4_df1'].fillna(merged_df['updrs_4_df2'])
merged_df = merged_df.drop(['updrs_1_df2', 'updrs_2_df2', 'updrs_3_df2', 'updrs_4_df2'], axis=1)
merged_df = merged_df.drop(['updrs_1_df1', 'updrs_2_df1', 'updrs_3_df1', 'updrs_4_df1'], axis=1)
merged_df
```

### Cell 8 code

```
train_clinical_all = merged_df[~merged_df.visit_month.isin([3, 5, 9])]
```

### Cell 9 code

```
    train_shift = train_clinical_all[['patient_id', 'visit_month', 'pred_month', 'updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']].copy()
    train_shift['visit_month'] -= plus_month
    train_shift.rename(columns={f'updrs_{i}': f'updrs_{i}_plus_{plus_month}' for i in range(1, 5)}, inplace=True)
    train_shift.rename(columns={'pred_month': f'pred_month_plus_{plus_month}'}, inplace=True)
    train_clinical_all = train_clinical_all.merge(train_shift, how='left', on=['patient_id', 'visit_month'])
```

### Cell 10 code

```
#     if target == 'updrs_1': pred_month = pred_month.clip(36, None)
    if target == 'updrs_4': pred_month = pred_month.clip(36, None)
    metric = smape_plus_1(
    return metric
```

### Cell 11 code

```
for test_clinical_data, test_peptides, test_proteins, sample_submission in iter_test:
    sample_submission['patient_id'] = sample_submission['prediction_id'].map(lambda x: int(x.split('_')[0]))
    sample_submission['visit_month'] = sample_submission['prediction_id'].map(lambda x: int(x.split('_')[1]))
    sample_submission['target_name'] = sample_submission['prediction_id'].map(lambda x: 'updrs_' + x.split('_')[3])
    sample_submission['plus_month'] = sample_submission['prediction_id'].map(lambda x: int(x.split('_')[5]))
    sample_submission['pred_month'] = sample_submission['visit_month'] + sample_submission['plus_month']
        mask_target = sample_submission['target_name'] == target
        sample_submission.loc[mask_target, 'rating'] = calculate_predicitons(
            pred_month=sample_submission.loc[mask_target, 'pred_month'],
    env.predict(sample_submission[['prediction_id', 'rating']])
```

## 20pct rank 232 quality strong score 73

File: `Time-Series/amp-parkinsons-disease-progression-prediction/rank232_20pct/pdpp-2-stage-model.ipynb`
Cells: 24 total, 16 code, 8 markdown

### Cell 0 markdown

```
Unfortunately I did not select this submission: it would have placed me top 22 with a score of 69.2 the CV score being 61.9 in 5 fold
```

### Cell 3 code

```
    metric = np.zeros(len(y_true_plus_1))
    metric[mask_not_zeros] = numerator[mask_not_zeros] / denominator[mask_not_zeros]
    return 100 * np.nanmean(metric)
```

### Cell 4 code

```
train_clinical_data = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv')
supplemental_clinical_data = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/supplemental_clinical_data.csv')
```

### Cell 6 code

```
    train_shift = train_clinical_all[['patient_id', 'visit_month', 'pred_month', 'updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']].copy()
    train_shift['visit_month'] -= plus_month
    train_shift.rename(columns={f'updrs_{i}': f'updrs_{i}_plus_{plus_month}' for i in range(1, 5)}, inplace=True)
    train_shift.rename(columns={'pred_month': f'pred_month_plus_{plus_month}'}, inplace=True)
    train_clinical_all = train_clinical_all.merge(train_shift, how='left', on=['patient_id', 'visit_month'])
```

### Cell 8 code

```
    if target == 'updrs_1': pred_month = pred_month.clip(12, 36)
    if target == 'updrs_2': pred_month = pred_month.clip(40, None)
    if target == 'updrs_3': pred_month = pred_month.clip(1, None)
    if target == 'updrs_4': pred_month = pred_month.clip(55, None)
    #pred_month = pred_month.clip(None, c_m)
    metric = smape_plus_1(
    return metric
print("OOF SMAPE+1 : {}".format(smape_plus_1(y_true=np.array(y_to_pred),
```

### Cell 10 code

```
# eval oof pred
print("Stage1 oof pred : {}".format(smape_plus_1(y_gt,preds)))
```

### Cell 14 code

```
proteins = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/train_proteins.csv')
train_clinical_all = train_clinical_all.merge(
# fill with last valid :
train_clinical_all[proteins_features.columns] = train_clinical_all.groupby('patient_id')[proteins_features.columns].fillna(method='ffill')
```

### Cell 18 code

```
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
estimators = [('lr', XGBRegressor()),
    model = StackingRegressor(estimators=estimators,n_jobs=2)
    # oof pred
```

### Cell 21 code

```
# eval oof pred
print("Stage2 oof pred : {}".format(smape_plus_1(y_gt,preds)))
```

### Cell 23 code

```
for test_clinical_data, test_peptides, test_proteins, sample_submission in iter_test:
    sample_submission['patient_id'] = sample_submission['prediction_id'].map(lambda x: int(x.split('_')[0]))
    sample_submission['visit_month'] = sample_submission['prediction_id'].map(lambda x: int(x.split('_')[1]))
    sample_submission['target_name'] = sample_submission['prediction_id'].map(lambda x: 'updrs_' + x.split('_')[3])
    sample_submission['plus_month'] = sample_submission['prediction_id'].map(lambda x: int(x.split('_')[5]))
    sample_submission['pred_month'] = sample_submission['visit_month'] + sample_submission['plus_month']
    sample_submission['visit_id'] = sample_submission['patient_id'].astype(str) + '_' + sample_submission['visit_month'].astype(str)
        mask_target = sample_submission['target_name'] == target
        sample_submission.loc[mask_target, 'target'] = calculate_predictions(
            pred_month=sample_submission.loc[mask_target, 'pred_month'],
    proteins_features_all[proteins_features.columns] = proteins_features_all.groupby('patient_id')[proteins_features.columns].\
    proteins_features = proteins_features_all.groupby('patient_id', as_index=False).last()
    sample_submission = sample_submission.merge(
        # oof pred
#         sample_submission.loc[mask_target, 'rating'] = np.round(sample_submission.loc[mask_target, 'rating'])
    env.predict(sample_submission[['prediction_id', 'rating']])
```

## 10pct rank 188 quality strong score 55

File: `Time-Series/amp-parkinsons-disease-progression-prediction/rank188_10pct/protein-npx-groups.ipynb`
Cells: 27 total, 22 code, 5 markdown

### Cell 2 code

```
    metric = np.zeros(len(y_true_plus_1))
    metric[mask_not_zeros] = numerator[mask_not_zeros] / denominator[mask_not_zeros]
    return 100 * np.nanmean(metric)
```

### Cell 4 code

```
train_clinical_all = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv')
proteins = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/train_proteins.csv')
train_clinical_all = train_clinical_all.merge(
# # # Calculate the exponentially weighted sum for each column
# # # Merge the exponentially weighted sum with the train_clinical_all DataFrame
# train_clinical_all = train_clinical_all.merge(
```

### Cell 5 code

```
# train_clinical_all[proteins_features.columns] = train_clinical_all.groupby('patient_id')[proteins_features.columns].\
# train_clinical_all[proteins_features.columns] = train_clinical_all.groupby('patient_id')[proteins_features.columns].\
train_clinical_all[proteins_features.columns] = train_clinical_all.groupby('patient_id')[proteins_features.columns].\
```

### Cell 6 code

```
#     # Check if the absolute correlation value is less than a threshold (e.g., 0.5)
```

### Cell 7 code

```
# # Set a correlation threshold (e.g., 0.7) to determine strong correlation
# correlation_threshold = 0.2
# # Select the features with correlation above the threshold
# strongly_correlated_features = correlation_with_target[correlation_with_target.abs() > correlation_threshold].index.tolist()
```

### Cell 8 code

```
    train_shift = train_clinical_all[['patient_id', 'visit_month', 'pred_month', 'updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']].copy()
    train_shift['visit_month'] -= plus_month
    train_shift.rename(columns={f'updrs_{i}': f'updrs_{i}_plus_{plus_month}' for i in range(1, 5)}, inplace=True)
    train_shift.rename(columns={'pred_month': f'pred_month_plus_{plus_month}'}, inplace=True)
    train_clinical_all = train_clinical_all.merge(train_shift, how='left', on=['patient_id', 'visit_month'])
```

### Cell 10 code

```
        pred_month = pred_month.clip(60, None)
```

### Cell 11 code

```
def calculate_predicitons_protein(pred_month, protein_shift):
    return np.round(pred_month_trend + protein_shift)
    metric = smape_plus_1(
            protein_shift=x[0]
    return metric
```

### Cell 14 code

```
        item[f'{target}_shift'] = find_best_const(train_clinical_all_filtered0, target)
```

### Cell 15 code

```
        y=f'{target}_shift',
```

### Cell 16 code

```
        item[f'{target}_shift'] = find_best_const(train_clinical_all_filtered1, target)
```

### Cell 17 code

```
        y=f'{target}_shift',
```

## 1st rank 1 quality strong score 78

File: `Time-Series/amp-parkinsons-disease-progression-prediction/rank1_1st/4th-place-solution-public-54-9-private-60-1.ipynb`
Cells: 27 total, 20 code, 7 markdown

### Cell 0 markdown

```
# 4th Place Solution - AMP Parkinson's Disease
This is my 4th place solution to Kaggle's AMP®-Parkinson's Disease Progression Prediction competition. The discussion describing this solution is [here][1]

[1]: https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/discussion/411398
```

### Cell 2 code

```
df1 = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv')
```

### Cell 3 code

```
df2 = pd.read_csv('/kaggle/input/amp-parkinsons-disease-progression-prediction/train_proteins.csv')
```

### Cell 6 markdown

```
# Feature Engineer
We will add "booleans" indicating whether a patient visited doctor on visit month `[0,6,12,18,24,36,48,60,72,84]`. These "booleans" are descibed in discussion [here][1]

[1]: https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/discussion/411398
```

### Cell 9 code

```
# REMOVE DATA BEFORE FIRST BLOOD WORK (because it is not scored by Kaggle)
```

### Cell 12 code

```
    # IS THIS SCORED??
    # IS THIS SCORED ??
```

### Cell 14 markdown

```
# Standardize Features
We use 11 features. Both SVR and MLP performs better if features are standardized
```

### Cell 19 code

```
import tensorflow as tf
import tensorflow.keras.backend as K
```

### Cell 20 code

```
    inp = tf.keras.Input(shape=(len(FEATURES2,)))
    x = tf.keras.layers.Dense(HIDDEN_SIZE,activation=ACT)(inp)
        x = tf.keras.layers.Dense(HIDDEN_SIZE)(x)
        x = tf.keras.layers.Activation(ACT)(x)
    x = tf.keras.layers.Dense(1,activation='linear')(x)
    model = tf.keras.Model(inputs=[inp], outputs=[x])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanAbsoluteError()
```

### Cell 21 code

```
    for fold in range(5):
        print('=> Fold',fold+1)
                #validation_data = (X_valid,y_valid),
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        loss = tf.keras.losses.MeanAbsoluteError()
                #validation_data = (X_valid,y_valid),
        clf.save_weights(f'MLP_{TAR}_f{fold}.h5')
```

### Cell 25 code

```
for (test, test_peptides, test_proteins, sample_submission) in iter_test:
        for fold in range(5):
            clf.load_weights(f'MLP_{t}_f{fold}.h5')
                # INFER 5 FOLDS
```

### Cell 26 code

```
sub = pd.read_csv('submission.csv')
```
