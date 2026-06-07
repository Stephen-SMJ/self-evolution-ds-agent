# Tabular / house-prices-advanced-regression-techniques

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 3619 quality strong score 53

File: `Tabular/house-prices-advanced-regression-techniques/rank3619_70pct/house-price-prediction.ipynb`
Cells: 24 total, 22 code, 2 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```

### Cell 1 markdown

```
# Business Problem Statement

The objective of this project is to predict residential house sale prices using property characteristics, construction details, and location-related information.

This project applies supervised machine learning regression techniques to analyze housing data and build predictive models capable of estimating house sale prices for unseen properties.
```

### Cell 2 code

```
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score
```

### Cell 3 code

```
train_df = pd.read_csv(r'/kaggle/input/competitions/house-prices-advanced-regression-techniques/train.csv')
```

### Cell 7 code

```
le = LabelEncoder()
```

### Cell 10 code

```
model = RandomForestRegressor(random_state=42)
```

### Cell 12 code

```
r2 = r2_score(y_test, y_pred)
print("R2 Score :",r2)
```

### Cell 15 code

```
test_df = pd.read_csv(r'/kaggle/input/competitions/house-prices-advanced-regression-techniques/test.csv')
```

### Cell 22 code

```
submission.to_csv('submission.csv',index=False)
```

### Cell 23 markdown

```
# Business Conclusion

The Random Forest Regression model successfully predicted residential house sale prices using property characteristics, construction details, and location-related features.

Feature importance analysis revealed that factors such as overall quality, living area size, garage capacity, and neighborhood-related attributes significantly influenced house prices.

The project demonstrates the complete supervised machine learning regression workflow including data preprocessing, feature encoding, model training, evaluation, feature importance analysis, and Kaggle competition submission generation.
```

## 40pct rank 2060 quality strong score 73

File: `Tabular/house-prices-advanced-regression-techniques/rank2060_40pct/house-prices-advanced-regression-techniques-2026.ipynb`
Cells: 75 total, 38 code, 37 markdown

### Cell 0 markdown

```
**Competition metric**: RMSE on log-transformed SalePrice (RMSLE)
3. 🤖 Baseline Models (CART, Random Forest, Gradient Boosting, XGBoost)
```

### Cell 3 code

```
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    StackingRegressor,
from xgboost import XGBRegressor
```

### Cell 5 code

```
train = pd.read_csv(DATA_DIR + 'train.csv')
test = pd.read_csv(DATA_DIR + 'test.csv')
```

### Cell 6 markdown

```
---
# 📊 Phase 1: Exploratory Data Analysis (EDA)

Before building models, we need to deeply understand our data — its distributions, relationships, missing patterns, and outliers. This informs every decision in feature engineering and modeling.
```

### Cell 8 code

```
print('   This is crucial because our metric (RMSLE) operates in log-space.')
```

### Cell 9 markdown

```
## 1.2 Missing Values Analysis

Understanding *why* data is missing is as important as knowing *what* is missing. In this dataset, many 'missing' values actually mean 'feature not present' (e.g., no pool, no garage, no basement).
```

### Cell 12 markdown

```
## 1.3 Correlation Analysis

Which numeric features are most strongly correlated with SalePrice? This helps us identify the most predictive features and spot multicollinearity.
```

### Cell 14 markdown

```
## 1.4 Key Feature Deep-Dives

Let's examine the most important features in detail.
```

### Cell 16 code

```
qual_medians = train.groupby('OverallQual')['SalePrice'].median()
```

### Cell 17 markdown

```
### 1.4.2 GrLivArea — Above Grade Living Area & Outlier Detection

GrLivArea is the 2nd strongest predictor. The scatter plot reveals **2 famous outliers**: houses with >4,000 sq ft but very low prices. These are well-documented anomalies that should be removed before modeling.
```

### Cell 20 code

```
neighborhood_order = train.groupby('Neighborhood')['SalePrice'].median().sort_values().index
print('💡 Price ranges from ~$90K (MeadowV, BrDale) to ~$335K (NoRidge, NridgHt, StoneBr)')
```

### Cell 21 markdown

```
### 1.4.4 Year Effects — Age Matters

Newer houses and recently remodeled houses tend to sell for more.
```

## 20pct rank 987 quality strong score 73

File: `Tabular/house-prices-advanced-regression-techniques/rank987_20pct/house-price-prediction-eda-feature-engineering.ipynb`
Cells: 19 total, 10 code, 9 markdown

### Cell 0 markdown

```
| Model | Validation R2 | Kaggle Score (RMSLE) |
| Ridge Regression (Pipeline) | 0.9057 | 0.13538 |
```

### Cell 2 code

```
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
```

### Cell 4 code

```
train = pd.read_csv('/kaggle/input/competitions/house-prices-advanced-regression-techniques/train.csv')
test  = pd.read_csv('/kaggle/input/competitions/house-prices-advanced-regression-techniques/test.csv')
```

### Cell 10 markdown

```
## 4. Data Preprocessing & Feature Engineering
```

### Cell 13 code

```
preprocessor = ColumnTransformer(transformers=[
    ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_cols)
    ('model', Ridge(alpha=10))
```

### Cell 14 markdown

```
## 6. Model Training & Evaluation
```

### Cell 15 code

```
# Train-Validation Split
r2   = r2_score(y_val, val_pred)
print(f'  R2 Score : {r2:.4f}')
```

### Cell 17 code

```
submission.to_csv('/kaggle/working/submission.csv', index=False)
```

### Cell 18 markdown

```
- **Encoding**: OrdinalEncoder for quality columns, OneHotEncoder for nominal columns
- **Model**: Ridge Regression (alpha=10) wrapped inside a clean sklearn Pipeline
| Metric | Score |
| Validation R2 | 0.9057 |
```

## 10pct rank 508 quality strong score 73

File: `Tabular/house-prices-advanced-regression-techniques/rank508_10pct/house-prices-xgb-cat-lgb-etc-0-12111.ipynb`
Cells: 35 total, 16 code, 19 markdown

### Cell 0 markdown

```
# House Prices | XGB+Cat+LGB etc. | LB 0.12111
Most public 0.12-range notebooks rely on either heavy stacking (Serigne-style meta-learner) or a single-seed boosting blend. On a small dataset (1,458 rows × ~250 features), these are prone to fold-level variance that doesn't generalize cleanly to the leaderboard. Our insight is to **average each tree model across 3 random seeds** (XGBoost, CatBoost, LightGBM) and then **hedge-blend** the resulting 5-way ensemble with a previously-validated submission — averaging two different preprocessing pipelines.
This took our LB through `0.12247 → 0.12142 → 0.12111` across three iterations. The final improvement came from adding LightGBM as a genuinely diverse signal (its predictions correlate with XGB/CatBoost at ~0.93, not the ~0.99 you typically see between variants of the same boosting method).
The `D_safe_lin` blend weights (Lasso 0.35, Ridge 0.10, XGB 0.20, CatBoost 0.20, LGB 0.15) were selected via OOF from 5 candidate schemes. Linear weight was consistently the strongest single contributor — a useful reminder that regularized linear models remain very competitive on small tabular regression.
- Engineers 14 new features (totals, ratios, presence flags, quality × area, neighborhood target encoding).
- Trains 5 models with 5-fold OOF: Lasso, Ridge, XGBoost (GPU, 3-seed), CatBoost (GPU, 3-seed), LightGBM (CPU, 3-seed).
- Blends with weights chosen by OOF and writes the submission.
```

### Cell 2 code

```
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
N_FOLDS = 5
```

### Cell 3 markdown

```
The original Ames dataset author (Dean De Cock) flagged a handful of houses that are very large (`GrLivArea > 4000`) but sold for very little (`< $300K`) — almost certainly non-market sales. Removing them helps the model learn the normal market.
We also `log1p` the target. The competition metric is RMSE on `log(SalePrice)`, so training on log-prices aligns the loss with the score and stabilizes residuals across the price range.
```

### Cell 4 code

```
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test  = pd.read_csv(f'{DATA_DIR}/test.csv')
# Neighborhood target encoding (train-only stats)
nbhd_median = train.groupby('Neighborhood')['SalePrice'].median()
```

### Cell 7 markdown

```
## 2. Missing values — "absence has meaning"

A key insight in this dataset: many NA values don't mean *unknown*, they mean *the feature isn't there*. `PoolQC = NA` means the house has no pool, not that the pool quality is missing. Same for `Alley`, `Fence`, `GarageType`, `BsmtQual`, etc.

So we fill these with `'None'` rather than imputing. For genuinely missing categoricals we use the mode; `LotFrontage` is filled with the median for the same `Neighborhood` (houses on the same block tend to have similar frontage).
```

### Cell 8 code

```
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
```

### Cell 9 markdown

```
Quality columns have a natural order: `Po < Fa < TA < Gd < Ex`. The naïve `LabelEncoder` sorts alphabetically (`Ex=0, Fa=1, Gd=2, None=3, Po=4, TA=5`), which destroys that order. A manual mapping `{'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}` preserves it, letting both linear and tree models use the rank directly.
```

### Cell 11 markdown

```
- **Presence flags**: `HasPool`, `HasGarage`, `Has2ndFloor`, `HasBsmt`, `HasFireplace` — a clean 0/1 even when the area column is 0.
- **Neighborhood target encoding**: replace the categorical name with the median SalePrice from the training set.
```

### Cell 12 code

```
# Presence flags
# Neighborhood target encoding
```

### Cell 13 markdown

```
Red bars are features we engineered. The fact that `QualArea`, `QualTotalSF`, `TotalSF`, and `NeighborhoodPrice` all rank near the top validates that the engineered features carry real signal.
```

### Cell 15 markdown

```
Many area/size columns are right-skewed (most houses are average, a few are huge). Linear models prefer roughly symmetric distributions, so we apply `log1p` wherever `|skew| > 0.75`. Then we one-hot encode the remaining categoricals — the final design matrix has around 265 columns.
```

### Cell 16 code

```
    all_data[col] = np.log1p(all_data[col].clip(lower=0))
```

## 1st rank 86 quality strong score 72

File: `Tabular/house-prices-advanced-regression-techniques/rank86_1st/house-prices-two-stage-model.ipynb`
Cells: 42 total, 20 code, 22 markdown

### Cell 0 markdown

```
5. **Stage 2: XGBoost** — Predicts dollar residuals with regularization (alpha=0.5, lambda=10)
```

### Cell 2 code

```
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
```

### Cell 4 code

```
train_raw = pd.read_csv(DATA_PATH + "train.csv")
test_raw  = pd.read_csv(DATA_PATH + "test.csv")
# Merge for unified cleaning
```

### Cell 6 code

```
# ─── 3.3 HasGarage flag ───
```

### Cell 12 code

```
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_knn = np.hstack([num_scaled, cat_encoded])
# Stack LotFrontage as first column for KNN imputation
full_knn = np.hstack([lf_col, X_knn])
```

### Cell 17 markdown

```
## 4. Feature Engineering v2

### 4.0 Pre-compute Derived Features
```

### Cell 18 code

```
# Re-merge for unified FE processing
```

### Cell 20 code

```
for col, threshold in sparse_candidates:
    if zero_pct < threshold:
```

### Cell 26 code

```
print(f"After OneHot: {all_fe.shape[1]} cols")
```

### Cell 30 code

```
# ─── 6c: Ultra-sparse OneHot (n < 5) ───
print(f"Dropped {len(sparse_drop)} ultra-sparse OneHot")
```

### Cell 33 markdown

```
— quality-to-size ratios, age-quality interactions, neighborhood deviation z-scores, etc.
```

### Cell 34 code

```
        nbh_med_qual = nbhd.map(cf.groupby('Neighborhood')['OverallQual'].median())
        nbh_med_sf = nbhd.map(pd.Series(total_sf.values, index=cf.index).groupby(nbhd).median())
        nbh_med_age = nbhd.map(pd.Series(house_age.values, index=cf.index).groupby(nbhd).median())
    # Group 5: Extreme combination flags
            'nbh_med_qual': cf.groupby('Neighborhood')['OverallQual'].median(),
            'nbh_med_sf': pd.Series(total_sf.values, index=cf.index).groupby(nbhd).median(),
            'nbh_med_age': pd.Series(house_age.values, index=cf.index).groupby(nbhd).median(),
X_s2_train = np.column_stack([X_base, X_dev_tr])
X_s2_test  = np.column_stack([X_test_base, X_dev_te])
```
