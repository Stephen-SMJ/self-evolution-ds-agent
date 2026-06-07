# Tabular / competitive-data-science-predict-future-sales

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 12343 quality strong score 53

File: `Tabular/competitive-data-science-predict-future-sales/rank12343_70pct/model-advanced-regression-predict-future-sales.ipynb`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
    def __init__(self, train_path, test_path, sample_submission_path):
        self.sample_submission_path = sample_submission_path
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        sample_submission = pd.read_csv(self.sample_submission_path)
        return train, test, sample_submission
        monthly_sales = self.train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({
    def add_lag_features(df, lags, col):
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ['date_block_num', 'shop_id', 'item_id', f'{col}_lag_{i}']
            shifted['date_block_num'] += i
            df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    def __init__(self, X_train, y_train, X_valid, y_valid):
        self.X_valid = X_valid
        self.y_valid = y_valid
        y_pred = model.predict(self.X_valid)
        mae = mean_absolute_error(self.y_valid, y_pred)
        r2_s = r2_score(self.y_valid, y_pred)
        print(f"R2_SCORE: {r2_s}")
    def prepare_test_set(test, sampled_data, lags, features):
        test = pd.merge(test, sampled_data[['shop_id', 'item_id', 'item_price']], on=['shop_id', 'item_id'], how='left')
        for lag in lags:
            lag_col_name = f'item_cnt_month_lag_{lag}'
            lag_data = sampled_data[sampled_data['date_block_num'] == (34 - lag)][['shop_id', 'item_id', 'item_cnt_month']]
            lag_data.columns = ['shop_id', 'item_id', lag_col_name]
            test = pd.merge(test, lag_data, on=['shop_id', 'item_id'], how='left')
    def create_submission(model, X_test, sample_submission):
        samp
...[truncated]
```

## 40pct rank 7678 quality strong score 73

File: `Tabular/competitive-data-science-predict-future-sales/rank7678_40pct/advanced-regression-predict-sales.ipynb`
Cells: 55 total, 42 code, 13 markdown

### Cell 0 markdown

```
1. **Feature Engineering**: A key part of this project involves creating meaningful features from historical sales data, such as monthly sales aggregates, lag-based features (e.g., previous month sales), and price trends. We will also explore combining supplemental data such as item categories and shop information to enrich our predictions.
2. **Handling Time Series Data**: Since the dataset consists of daily sales data from January 2013 to October 2015, we need to model temporal dependencies effectively. Strategies such as using **rolling windows** or **lag features** will help capture past trends, while features like the **date_block_num** will be used to track time progression.
   - **Gradient Boosting Machines (GBM)**: XGBoost or LightGBM, which are well-suited for tabular data and can handle large datasets efficiently.
5. **Evaluation Metric**: The competition uses Root Mean Squared Error (RMSE) as the evaluation metric, which penalizes larger errors more heavily. As the target values are clipped between 0 and 20, our model needs to focus on predicting values within this range accurately, avoiding extreme predictions.
- **Modeling and Hyperparameter Tuning**: Experimenting with various machine learning algorithms, tuning hyperparameters, and leveraging cross-validation techniques will be crucial for model improvement.
```

### Cell 1 code

```
        data[name] = pd.read_csv(path)
```

### Cell 6 code

```
grouped_sales = sales_data.groupby(['date', 'shop_id', 'item_id']).agg({'item_cnt_day': 'sum'}).reset_index()
```

### Cell 8 code

```
datewise_sales = grouped_sales.groupby('date')['total_items_sold'].sum().reset_index()
```

### Cell 10 code

```
shopwise_sales = grouped_sales.groupby('shop_id')['total_items_sold'].sum().reset_index()
```

### Cell 13 code

```
monthly_sales = sales_data.groupby(['year', 'month']).agg({'item_cnt_day': 'sum'}).reset_index()
```

### Cell 15 code

```
top_items = sales_data.groupby('item_id').agg({'item_cnt_day': 'sum'}).sort_values(by='item_cnt_day', ascending=False).head(10)
top_items = top_items.merge(items[['item_id', 'item_name']], on='item_id')
```

### Cell 17 code

```
top_shops = sales_data.groupby('shop_id').agg({'item_cnt_day': 'sum'}).sort_values(by='item_cnt_day', ascending=False).head(10)
top_shops = top_shops.merge(shops[['shop_id', 'shop_name']], on='shop_id')
```

### Cell 20 code

```
# Merge with test_data
merged_data = pd.merge(test_data, dataset, on=['item_id', 'shop_id'], how='left')
# Display the merged dataset
print(merged_data.head())
```

### Cell 21 markdown

```
# Purpose of the Merge
* on=['item_id', 'shop_id']: This tells the merge() function to combine the two DataFrames based on the common columns item_id and shop_id.
```

### Cell 22 code

```
merged_data.head()
```

### Cell 23 code

```
merged_data.fillna(0,inplace=True)
merged_data.head()
```

## 20pct rank 3268 quality weak score 15

File: `Tabular/competitive-data-science-predict-future-sales/rank3268_20pct/notebook5f4ce4366a.ipynb`
Cells: 14 total, 14 code, 0 markdown

### Cell 1 code

```
train = pd.read_csv(
```

### Cell 5 code

```
train_agg = train[agg_features].groupby(["date_block_num", "shop_id", "item_id"]).sum().reset_index()
```

### Cell 11 code

```
train_agg_total = train_agg.groupby("date_block_num")["item_cnt_day"].sum().reset_index()
```

## 10pct rank 1657 quality strong score 61

File: `Tabular/competitive-data-science-predict-future-sales/rank1657_10pct/ch9-modeling.ipynb`
Cells: 84 total, 54 code, 30 markdown

### Cell 1 code

```
sales_train = pd.read_csv(data_path + 'sales_train.csv')
shops = pd.read_csv(data_path + 'shops.csv')
items = pd.read_csv(data_path + 'items.csv')
item_categories = pd.read_csv(data_path + 'item_categories.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')
```

### Cell 14 code

```
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
```

### Cell 18 code

```
items['첫 판매월'] = sales_train.groupby('상품ID').agg({'월ID': 'min'})['월ID']
```

### Cell 26 code

```
label_encoder = LabelEncoder()
```

### Cell 29 code

```
train = pd.DataFrame(np.vstack(train), columns=idx_features)
```

### Cell 31 code

```
group = sales_train.groupby(idx_features).agg({'판매량': 'sum',
train = train.merge(group, on=idx_features, how='left')
```

### Cell 33 code

```
group = sales_train.groupby(idx_features).agg({'판매량': 'count'})
train = train.merge(group, on=idx_features, how='left')
```

### Cell 38 code

```
all_data = all_data.merge(shops, on='상점ID', how='left')
all_data = all_data.merge(items, on='상품ID', how='left')
all_data = all_data.merge(item_categories, on='상품분류ID', how='left')
```

### Cell 42 code

```
    group = df.groupby(idx_features).agg({'월간 판매량': 'mean'})
    df = df.merge(group, on=idx_features, how='left')
```

### Cell 48 code

```
def add_lag_features(df, lag_features_to_clip, idx_features,
                     lag_feature, nlags=3, clip=False):
    df_temp = df[idx_features + [lag_feature]].copy()
    for i in range(1, nlags+1):
        lag_feature_name = lag_feature +'_시차' + str(i)
        df_temp.columns = idx_features + [lag_feature_name]
        df = df.merge(df_temp.drop_duplicates(),
        df[lag_feature_name] = df[lag_feature_name].fillna(0)
        # 0 ~ 20 사이로 제한할 시차 피처명을 lag_features_to_clip에 추가
            lag_features_to_clip.append(lag_feature_name)
    return df, lag_features_to_clip
```

### Cell 50 code

```
lag_features_to_clip = [] # 0 ~ 20 사이로 제한할 시차 피처명을 담을 리스트
all_data, lag_features_to_clip = add_lag_features(df=all_data,
                                                  lag_features_to_clip=lag_features_to_clip,
                                                  lag_feature='월간 판매량',
                                                  nlags=3,
```

### Cell 52 code

```
lag_features_to_clip
```

## 1st rank 98 quality usable score 38

File: `Tabular/competitive-data-science-predict-future-sales/rank98_1st/extcopy-of-future-sales-with-automated-ensembling.ipynb`
Cells: 9 total, 5 code, 4 markdown

### Cell 2 code

```
submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
```

### Cell 4 code

```
            df = pd.read_csv(os.path.join(dirname, filename))
```

### Cell 6 code

```
# Compute weighted sum
weighted_predictions = np.average(preds_array, axis=0, weights=weights)
submission[targetName] = weighted_predictions
submission.to_csv("submission.csv", index=False)
```

### Cell 8 code

```
# submission.to_csv("submission.csv", index=False)
```
