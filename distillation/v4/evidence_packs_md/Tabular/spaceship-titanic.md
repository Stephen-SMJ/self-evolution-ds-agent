# Tabular / spaceship-titanic

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1616 quality strong score 55

File: `Tabular/spaceship-titanic/rank1616_70pct/spaceship-titanic-0527.ipynb`
Cells: 22 total, 16 code, 6 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
```

### Cell 1 code

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
```

### Cell 2 code

```
df_train = pd.read_csv("/kaggle/input/competitions/spaceship-titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/competitions/spaceship-titanic/test.csv")
```

### Cell 4 code

```
member_count= group.groupby("group")["group"].transform("size")
member_count= group.groupby("group")["group"].transform("size")
```

### Cell 9 code

```
df_train_new.groupby("CryoSleep")["Transported"].mean().plot(kind="bar")
```

### Cell 12 code

```
ax = df_train_new.query("CryoSleep==True").groupby(agebin, observed=False)["Transported"].mean().plot(kind="bar")
ax = df_train_new.query("CryoSleep==True").groupby("member_count")["Transported"].mean().plot(kind="bar")
t_rate_4Sleep = df_train_new.query("CryoSleep==True").groupby([agebin, "deck"],observed=False)["Transported"].mean()
ax = sns.heatmap(t_rate_4Sleep.unstack(), annot=True)
```

### Cell 14 code

```
t_rate_4nonSleep = df_train_new.query("CryoSleep==False").groupby([agebin, "Last_init"],observed=False)["Transported"].mean()
ax = sns.heatmap(t_rate_4nonSleep.unstack(), annot=True)
ax = df_train_new.query("CryoSleep==False").groupby(servicebin,observed=False)["Transported"].mean().plot(kind="bar")
```

### Cell 20 code

```
    ("encode", OneHotEncoder(handle_unknown="ignore"))
    #("model", RandomForestClassifier(n_estimators=1000, random_state=0))
    ("model", xgb.XGBClassifier(n_estimators=1000,
    ("encode", OneHotEncoder(handle_unknown="ignore"))
    #("model", RandomForestClassifier(n_estimators=1000, random_state=0))
    ("model", xgb.XGBClassifier(n_estimators=1000,
```

### Cell 21 code

```
y_pred.to_csv("submission.csv", index=False)
```

## 40pct rank 870 quality strong score 73

File: `Tabular/spaceship-titanic/rank870_40pct/beginner-eda-to-clean-cv-gpu-catboost.ipynb`
Cells: 55 total, 30 code, 25 markdown

### Cell 0 markdown

```
  <div style="margin: 0 0 8px 0; color: #233127; font-size: 2.05rem; font-weight: 900; line-height: 1.14;">🚀 Beginner EDA to Clean CV + GPU CatBoost</div>
    Cabin structure, CryoSleep, spending behavior, group-aware validation, GPU-ready CatBoost, OOF blending, and a fold-averaged submission for predicting <span style="color:#2F6B4F; font-weight:900; border-bottom:2px solid #CFE0D0;">Transported</span>.
      <div style="font-size: 11px; text-transform: uppercase; color: #6F7C69; font-weight: 900; letter-spacing: 0.08em; margin-bottom: 6px;">🧪 Validation</div>
      <div style="font-size: 15px; color: #233127; font-weight: 800; line-height: 1.35;">StratifiedGroupKFold</div>
      <div style="font-size: 15px; color: #233127; font-weight: 800; line-height: 1.35;">Fold-averaged probabilities</div>
```

### Cell 1 markdown

```
    <div style="border: 1px solid #E4DAC9; border-radius: 14px; padding: 13px 14px; background: #F7F3EA;"><div style="font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color:#6F7C69; font-weight:900; margin-bottom:6px;">Models</div><div style="font-weight: 900; color: #C7772C; margin-bottom: 6px;">7-9. Validation and Interpretation</div><div style="font-size: 12.5px; line-height: 1.55; color: #233127;">Group-aware CV, ablation, threshold tuning, and feature importance.</div></div>
```

### Cell 2 markdown

```
<div style="background:#FFFCF8; border:1px solid #E2D8C7; border-left:6px solid #2F6B4F; border-radius:14px; padding:13px 15px; margin:10px 0 16px 0; color:#233127; line-height:1.58; box-shadow:0 8px 18px rgba(35,49,39,0.04);">Spaceship Titanic is a <b>binary classification</b> problem. For each passenger, the model predicts whether <span style="color:#2F6B4F; font-weight:900; border-bottom:2px solid #CFE0D0;">Transported</span> is <b>True</b> or <b>False</b>. The public score is accuracy.</div>
```

### Cell 3 code

```
from sklearn.ensemble import (
    RandomForestClassifier,
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
def display_metric_cards(cards, title=None, subtitle=None):
        ratio_like_tokens = ['rate', 'share', 'ratio', 'accuracy', 'auc', 'loss', 'smd', 'correlation', 'drop', 'std', 'mean', 'threshold', 'entropy']
            valid = numeric.dropna()
            if valid.empty:
            is_integer_like = np.all(np.isclose(valid, np.round(valid), atol=1e-9))
                inferred[col] = '{:.2%}' if valid.between(0, 1).all() else '{:.2f}'
            valid = numeric.dropna()
            if valid.empty:
            lo, hi = valid.min(), valid.max()
            normalized = ((numeric - lo) / denom).fillna(0.0).clip(0, 1)
```

### Cell 5 code

```
submission_template_path = resolve_file('sample_submission.csv')
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(submission_template_path)
    '📄 Table': ['train', 'test', 'sample_submission'],
    'Rows': [len(train), len(test), len(sample_submission)],
    'Columns': [train.shape[1], test.shape[1], sample_submission.shape[1]],
    subtitle='The train/test sizes are small enough for fast iteration, but large enough for stable group-aware validation.',
```

### Cell 6 code

```
# COMPUTED HEADLINE METRICS
metric_cards = [
display_metric_cards(
    metric_cards,
```

### Cell 8 code

```
    {'Column': 'Name', 'Family': '👤 Text identifier', 'Plain-English meaning': 'Passenger name; surname can approximate family links.', 'Modeling note': 'Use surname counts; raw surname is best saved for CatBoost.'},
    {'Column': 'Transported', 'Family': '🎯 Target', 'Plain-English meaning': 'Binary label to predict.', 'Modeling note': 'Accuracy is the competition metric.'},
```

### Cell 15 markdown

```
| 🚪 Cabin deck, side, and interactions | `Cabin` | Ship location and deck-side combinations can shift the base rate. |
```

### Cell 16 code

```
    return usable.groupby(key_col)[value_col].agg(lambda values: values.mode().iat[0])
    spend_ratio = spend_for_features.div(data['TotalSpend'].where(data['TotalSpend'].gt(0), np.nan), axis=0).fillna(0).clip(0, 1)
```

### Cell 17 code

```
# OneHotEncoder expects each categorical column to contain a consistent data type.
```

### Cell 19 markdown

```
### 4.1 Rule-Based Imputation Audit

Some missing values can be filled with structural rules before the general model pipeline runs. These rules are intentionally conservative: they use passenger metadata and observed feature relationships, not the target label.
```

### Cell 22 code

```
    .groupby(['CryoSleepLabel', 'NoSpend'], observed=True)[TARGET]
    .agg(Rows='count', TransportedRate='mean')
```

## 20pct rank 440 quality strong score 73

File: `Tabular/spaceship-titanic/rank440_20pct/spaceship-titanic.ipynb`
Cells: 33 total, 32 code, 1 markdown

### Cell 0 markdown

```
## High Accuracy Ensemble Pipeline with Feature Engineering
✅ High Accuracy Ensemble Models
✅ Cross Validation
```

### Cell 1 code

```
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
```

### Cell 2 code

```
train_df = pd.read_csv('/kaggle/input/competitions/spaceship-titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/competitions/spaceship-titanic/train.csv')
```

### Cell 11 code

```
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
```

### Cell 20 code

```
    encoder = LabelEncoder()
```

### Cell 22 code

```
from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
```

### Cell 23 code

```
# TRAIN VALIDATION SPLIT
X_train, X_valid, y_train, y_valid = train_test_split(
print("X_valid Shape :", X_valid.shape)
```

### Cell 24 code

```
# TRAIN CATBOOST MODEL
from catboost import CatBoostClassifier
cat_model = CatBoostClassifier(
    eval_metric='Accuracy',
```

### Cell 25 code

```
# VALIDATION PREDICTIONS
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
preds = cat_model.predict(X_valid)
accuracy = accuracy_score(
    y_valid,
print("Validation Accuracy :", accuracy)
```

### Cell 26 code

```
        y_valid,
```

### Cell 27 code

```
    y_valid,
```

### Cell 30 code

```
test_df = pd.read_csv(
test_df['GroupSize'] = test_df.groupby('Group')['Group'].transform('count')
    encoder = LabelEncoder()
```

## 10pct rank 134 quality strong score 55

File: `Tabular/spaceship-titanic/rank134_10pct/notebooka2b1182843.ipynb`
Cells: 14 total, 14 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```

### Cell 1 code

```
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.utils import set_random_seed
```

### Cell 2 code

```
train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
```

### Cell 3 code

```
test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
```

### Cell 4 code

```
# merge test and train datasets into single dataframe for data cleaning and necessary pre-processing
```

### Cell 5 code

```
Columns with ordinal values (ranks, value_counts, high-cardinality columns, etc) will be processed
to be OneHotEncoded, but we will first convert them into numerical codes (with Ordinal Encoder) so
```

### Cell 6 code

```
# Splitting our merged dataset into original train and test sets,
```

### Cell 11 code

```
# Transforming Categorical features using OneHotEncoder
onehot = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
onehot = make_column_transformer((onehot, categorical), remainder='passthrough')
X_train = onehot.fit_transform(X_train)
X_test = onehot.transform(X_test)
# Creating training and validation split (including corresponding indices from the base set) to evaluate model performances
```

### Cell 13 code

```
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=2, random_state=SEED, )
    # training on shuffled dataset with 0.2 validation_fraction
    model = GradientBoostingClassifier(max_depth=7, max_features='sqrt', n_estimators=600, n_iter_no_change=50, tol=0, subsample=0.75, learning_rate=0.05, validation_fraction=0.2, verbose=0, random_state=SEED+i )
    # Checking OOB scores
#     print(i, 'accuracy', model.score(X_train_full[ts_ids], y_train_full[ts_ids]), '\n')
    # Refit the model using n_estimators = raw_ests - early stopping
    model = GradientBoostingClassifier(max_depth=7, max_features='sqrt', n_estimators=estop_ests, n_iter_no_change=50, tol=0, subsample=0.75, learning_rate=0.05, validation_fraction=0.2, verbose=0, random_state=SEED+i)
    # Checking OOB scores
#     print(i, 'accuracy', model.score(X_train_full[ts_ids], y_train_full[ts_ids]), '\n')
    # predict on test set using model trained on StratifiedShuffleSplit of training data with 0.2 validation_fraction
submission.to_csv('submission.csv', index=False)
```

## 1st rank 119 quality usable score 32

File: `Tabular/spaceship-titanic/rank119_1st/spaceship-titanic-survival-prediction-model.ipynb`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
SUB_PATH = "/kaggle/input/competitions/spaceship-titanic/sample_submission.csv"
test = pd.read_csv(TEST_PATH)
sub = pd.read_csv(SUB_PATH)
sub.to_csv("/kaggle/working/submission.csv", index=False)
```
