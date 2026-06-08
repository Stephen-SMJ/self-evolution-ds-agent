# Tabular / titanic

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 8790 quality strong score 55

File: `Tabular/titanic/rank8790_70pct/titanic-model.ipynb`
Cells: 36 total, 36 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```

### Cell 1 code

```
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
```

### Cell 2 code

```
test_data = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')
train_data = pd.read_csv('/kaggle/input/competitions/titanic/train.csv')
genders = pd.read_csv('/kaggle/input/competitions/titanic/gender_submission.csv')
```

### Cell 10 code

```
model = LogisticRegression(
```

### Cell 13 code

```
accuracy = accuracy_score(Y_test,y_pred)
```

### Cell 15 code

```
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score
print(precision_score(Y_test,y_pred))
print(recall_score(Y_test,y_pred))
print(f1_score(Y_test,y_pred))
```

### Cell 16 code

```
model = RandomForestClassifier(
```

### Cell 19 code

```
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score
accuracy = accuracy_score(Y_test,y_pred)
print(precision_score(Y_test,y_pred))
print(recall_score(Y_test,y_pred))
print(f1_score(Y_test,y_pred))
```

### Cell 20 code

```
model = XGBClassifier(
```

### Cell 23 code

```
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score
accuracy = accuracy_score(Y_test,y_pred)
print(precision_score(Y_test,y_pred))
print(recall_score(Y_test,y_pred))
print(f1_score(Y_test,y_pred))
```

### Cell 31 code

```
model = XGBClassifier(
```

### Cell 35 code

```
submission.to_csv('gender_submission.csv', index=False)
```

## 40pct rank 4820 quality strong score 51

File: `Tabular/titanic/rank4820_40pct/gettingstartedwithtitanic.ipynb`
Cells: 4 total, 4 code, 0 markdown

### Cell 1 code

```
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
```

### Cell 2 code

```
train_data['Age'] = train_data['Age'].fillna(train_data.groupby('Title')['Age'].transform('median'))
test_data['Age'] = test_data['Age'].fillna(test_data.groupby('Title')['Age'].transform('median'))
```

### Cell 3 code

```
from xgboost import XGBClassifier
model = XGBClassifier(
output.to_csv('submission.csv', index=False)
```

## 20pct rank 2445 quality strong score 55

File: `Tabular/titanic/rank2445_20pct/getting-started-with-titanic.ipynb`
Cells: 14 total, 14 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
```

### Cell 1 code

```
train_data = pd.read_csv("/kaggle/input/competitions/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/competitions/titanic/test.csv")
```

### Cell 5 code

```
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
    transformers=[
```

### Cell 6 code

```
    ("classifier", RandomForestClassifier(
X_train, X_valid, y_train, y_valid = train_test_split(
preds = model.predict(X_valid)
accuracy_score(y_valid, preds)
```

### Cell 7 code

```
# submission.to_csv("submission.csv", index=False)
```

### Cell 11 code

```
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
    transformers=[
    ("classifier", RandomForestClassifier(
```

### Cell 12 code

```
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(scores)
print("Média:", scores.mean())
print("Desvio:", scores.std())
```

### Cell 13 code

```
submission.to_csv("submission.csv", index=False)
```

## 10pct rank 1487 quality strong score 73

File: `Tabular/titanic/rank1487_10pct/titanic-78-95-accurancy-eda-ensemble-learning.ipynb`
Cells: 36 total, 19 code, 17 markdown

### Cell 1 markdown

```
The score improved from 78.22 to 78.95 by adding the logistic model and random forest classifier!
```

### Cell 2 markdown

```
### Comprehensive EDA & Soft-Voting Ensemble Pipeline
4. Stratified hyperparameter optimization across cross-validated architectures.
5. Soft-Voting Ensembling integrating tree architectures and geometric classifiers.
```

### Cell 3 code

```
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
df_train = pd.read_csv('/kaggle/input/competitions/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')
```

### Cell 8 markdown

```
## 2. Demographic Exploration & Categorical Encoding (Sex)
### Analysis & Domain Insights
Encoding the `Sex` element into binary vectors allows our distance matrices to map demographic dependencies. The resulting bar chart highlights a substantial rift: **the survival probability of female passengers overwhelmingly exceeds that of males**. This empirical finding confirms the traditional "women and children first" maritime protocol. We anticipate this feature to carry dominant weight across our models.
```

### Cell 9 code

```
le = LabelEncoder()
Sex_Survived = df_train.groupby('Sex')['Survived'].mean().reset_index()
```

### Cell 11 code

```
Age_Survived = df_train.groupby('Age')['Survived'].mean().reset_index()
```

### Cell 12 markdown

```
Replacing missing continuous values with standard parameters (like mean or median) spikes the kurtosis artificially and flattens variances, altering the natural data shape. To protect the original age distribution, we draw random observations directly from the valid subset of training observations, ensuring our models process consistent statistical shapes.
```

### Cell 13 code

```
title_medians_dict = df_train.groupby('Title')['Age'].median().to_dict()
```

### Cell 15 code

```
embark_perc = df_train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()
```

### Cell 16 markdown

```
## 6. Multi-Feature Correlation Diagnosis
### Matrix Insights
Constructing a linear correlation matrix shows the deep dependencies within the dataset. We note strong inverse interactions between `Pclass` and `Fare`, indicating that higher class tiers command significantly higher pricing structures. Identifying these interactions helps ensure stable parameter weights across our downstream algorithms.
```

### Cell 19 code

```
df_plot = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')
```

### Cell 20 markdown

```
- **Family Structures**: Combining `SibSp` and `Parch` into a single `Family` metric lets us evaluate social group dynamics, while the `Alone` flag captures the survival differences between solo passengers and family units.
```

## 1st rank 4 quality strong score 78

File: `Tabular/titanic/rank4_1st/titanic-cv-v2.ipynb`
Cells: 17 total, 15 code, 2 markdown

### Cell 0 markdown

```
# Titanic — Clean Feature Engineering + CV Ensemble (CatBoost / LightGBM)
- Uses Stratified K-Fold CV with OOF probabilities
- Tunes a classification threshold on OOF
- Blends CatBoost + LightGBM and outputs `/kaggle/working/submission.csv`
```

### Cell 1 code

```
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
```

### Cell 2 code

```
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
```

### Cell 4 code

```
    g1 = df.groupby(["Title", "Pclass", "Sex"])["Age"].median()
    g2 = df.groupby(["Pclass", "Sex"])["Age"].median()
```

### Cell 8 code

```
axes[0].bar(tmp.groupby("Sex")["Survived"].mean().index, tmp.groupby("Sex")["Survived"].mean().values)
axes[1].bar(tmp.groupby("Pclass")["Survived"].mean().index, tmp.groupby("Pclass")["Survived"].mean().values)
axes[2].bar(tmp.groupby("Embarked")["Survived"].mean().index, tmp.groupby("Embarked")["Survived"].mean().values)
```

### Cell 9 code

```
def tune_threshold(y_true, proba):
        a = accuracy_score(y_true, (proba >= t).astype(int))
```

### Cell 10 code

```
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_cb = np.zeros(len(X), dtype=float)
for fold, (trn, val) in enumerate(skf.split(X, y), 1):
    m = CatBoostClassifier(
        eval_metric="Accuracy",
        early_stopping_rounds=400,
    oof_cb[val] = m.predict_proba(va_pool)[:, 1]
t_cb, a_cb = tune_threshold(y, oof_cb)
```

### Cell 11 code

```
pred_cb = (oof_cb >= t_cb).astype(int)
```

### Cell 12 code

```
plt.title("CatBoost OOF Confusion Matrix")
```

### Cell 13 code

```
skf2 = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_lgb = np.zeros(len(X), dtype=float)
for fold, (trn, val) in enumerate(skf2.split(X, y), 1):
    m = lgb.LGBMClassifier(
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(400, verbose=False)]
    oof_lgb[val] = m.predict_proba(X_va)[:, 1]
t_lgb, a_lgb = tune_threshold(y, oof_lgb)
```

### Cell 14 code

```
    p = w * oof_cb + (1 - w) * oof_lgb
    t, a = tune_threshold(y, p)
```

### Cell 15 code

```
oof_blend = w * oof_cb + (1 - w) * oof_lgb
test_blend = w * test_cb + (1 - w) * test_lgb
acc_blend = accuracy_score(y, (oof_blend >= t).astype(int))
acc_blend, w, t
```
