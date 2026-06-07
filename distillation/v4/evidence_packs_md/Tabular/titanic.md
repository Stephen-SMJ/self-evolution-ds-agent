# Tabular / titanic

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 8833 quality strong score 55

File: `Tabular/titanic/rank8833_70pct/titanic-model.ipynb`
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

## 40pct rank 4868 quality strong score 51

File: `Tabular/titanic/rank4868_40pct/gettingstartedwithtitanic.ipynb`
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

## 20pct rank 2509 quality strong score 55

File: `Tabular/titanic/rank2509_20pct/getting-started-with-titanic.ipynb`
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

## 10pct rank 1500 quality strong score 73

File: `Tabular/titanic/rank1500_10pct/titanic-78-95-accurancy-eda-ensemble-learning.ipynb`
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

## 1st rank 874 quality strong score 78

File: `Tabular/titanic/rank874_1st/ml-titanic-competition.ipynb`
Cells: 23 total, 18 code, 5 markdown

### Cell 1 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, classification_report
```

### Cell 2 markdown

```
# 01: Exploratory Data Analysis and Features creation
First, let's dive into the data, change them into numerical values, plot some of them and create our own features
```

### Cell 3 code

```
unprocessed_data = pd.read_csv("/kaggle/input/competitions/titanic/train.csv")
```

### Cell 12 markdown

```
# 02: Processing and scaling of the features
Now let's process our features and create our training and test sets
```

### Cell 15 markdown

```
# 03: Definition of the model and training
Let's define the model and analyse the outputs on the training and test set
```

### Cell 16 code

```
titanic_model = LogisticRegression(
                     # l2 (Ridge) keeps all features. l1 (Lasso) can zero out some weights.
```

### Cell 17 code

```
    LogisticRegression(penalty='l2', max_iter=10000, solver='lbfgs', random_state=42),
    cv=5,           # cross-validation on the dataset divided into 5 (4 for train, 1 for test)
```

### Cell 18 code

```
print(f"Accuracy train : {accuracy_score(Y_train, y_pred_train):.4f}")
print(f"Accuracy val   : {accuracy_score(Y_test,  y_pred_val):.4f}")
```

### Cell 20 markdown

```
# 04: Predictions on the Titanic survivors
Now lets's apply our model and predict the survivors of the competition set
```

### Cell 21 code

```
predictions_data = pd.read_csv("/kaggle/input/competitions/titanic/test.csv")
```

### Cell 22 code

```
submission.to_csv("submission.csv", index=False)
```
