# Time-Series / tabular-playground-series-jan-2022

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1105 quality strong score 47

File: `Time-Series/tabular-playground-series-jan-2022/rank1105_70pct/tpg-0122.ipynb`
Cells: 45 total, 45 code, 0 markdown

### Cell 1 code

```
train = pd.read_csv('../input/tabular-playground-series-jan-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-jan-2022/test.csv')
```

### Cell 24 code

```
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
```

### Cell 35 code

```
from sklearn.ensemble import RandomForestRegressor
```

### Cell 36 code

```
model = RandomForestRegressor(n_estimators = 1000, max_depth = 10, random_state = 42)
```

### Cell 40 code

```
submission = pd.read_csv('../input/tabular-playground-series-jan-2022/sample_submission.csv')
```

### Cell 44 code

```
submission.to_csv('submission.csv', index=False)
```

## 40pct rank 637 quality strong score 59

File: `Time-Series/tabular-playground-series-jan-2022/rank637_40pct/basic-xgbregressor.ipynb`
Cells: 21 total, 17 code, 4 markdown

### Cell 1 code

```
train = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2022/train.csv')
test = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2022/test.csv')
```

### Cell 9 code

```
from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
```

### Cell 16 markdown

```
The graphs had shown that linear methods won't work. It won't handle these ups in the end of any year. Tree-based models could hangle that.
```

### Cell 17 code

```
from xgboost import XGBRegressor
model = XGBRegressor(eta=0.05)
```

### Cell 19 code

```
result = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2022/sample_submission.csv')
```

### Cell 20 code

```
result.to_csv('submission.csv', index=False)
```

## 20pct rank 308 quality strong score 55

File: `Time-Series/tabular-playground-series-jan-2022/rank308_20pct/holiday-kernel.ipynb`
Cells: 24 total, 14 code, 10 markdown

### Cell 0 markdown

```
The effect of holidays seems complicated but they are actually simple superpositions of a single kernel (and an additional one for Christmas). This notebook is the 5th-place solution with some visualizations.

Minimum linear regression (5th-place):
https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/304369
```

### Cell 2 code

```
train = pd.read_csv(di + 'train.csv')
test = pd.read_csv(di + 'test.csv')
submit = pd.read_csv(di + 'sample_submission.csv')
```

### Cell 4 code

```
holiday_table = pd.read_csv(io.StringIO(
```

### Cell 6 code

```
             'country': country,  # binary flag if
```

### Cell 8 code

```
    g = pd.read_csv('/kaggle/input/gdp-20152019-finland-norway-and-sweden/'
    # Convert to year, country, GDP format for pd.merge
    df = pd.merge(df, gdp, how='left', on=['year', 'country'])
```

### Cell 9 markdown

```
   * 10 binary flags for "today is n (0 ≦ n < 10) days after holiday"
   * Similar flag for 10 days after Christmas
```

### Cell 10 code

```
    # Holiday flag for ith day after holiday
    # Additional flag for Christmas
```

### Cell 19 markdown

```
The difference among countries can be explaind by the difference in official holidays, e.g.,

- Norway does not have Chrismas Eve and has smaller peak,
- Finland and Sweden have Epiphany (6 Jan) while Norway do not,

and only 2 kernels are necessary.

Points are data corrected for basic factors (weekday, store, and product including sinusoidal) and averaged over store and product. Lines are the model.

December and January are still easy because the holiday dates are fixed.
```

### Cell 23 code

```
submit.to_csv('submission.csv', index=False)
```

## 10pct rank 139 quality usable score 41

File: `Time-Series/tabular-playground-series-jan-2022/rank139_10pct/tps-jan-2022-pycaret.ipynb`
Cells: 15 total, 15 code, 0 markdown

### Cell 2 code

```
train = cudf.read_csv('../input/tabular-playground-series-jan-2022/train.csv', index_col = 'row_id').to_pandas()
test = cudf.read_csv('../input/tabular-playground-series-jan-2022/test.csv', index_col = 'row_id').to_pandas()
```

### Cell 3 code

```
# Credit to https://www.kaggle.com/ranjeetshrivastav/tps-jan-21-base-xgb
```

### Cell 8 code

```
blend = blend_models(top)
predict_model(blend)
```

### Cell 9 code

```
final_blend = finalize_model(blend)
predict_model(final_blend)
```

### Cell 11 code

```
#tuned_blend = blend_models(tuned_top)
#predict_model(tuned_blend);
```

### Cell 12 code

```
#final_tuned_blend = finalize_model(tuned_blend)
#predict_model(final_tuned_blend);
```

### Cell 13 code

```
unseen_predictions_blend = predict_model(final_blend, data=test)
unseen_predictions_blend.head()
```

### Cell 14 code

```
assert(len(test.index)==len(unseen_predictions_blend))
sub = pd.DataFrame(list(zip(test.index, unseen_predictions_blend.Label)),columns = ['row_id', 'num_sold'])
sub.to_csv('submission.csv', index = False)
```

## 1st rank 3 quality strong score 60

File: `Time-Series/tabular-playground-series-jan-2022/rank3_1st/tpsjan22-fb-prophet-6-12-private-score.ipynb`
Cells: 9 total, 9 code, 0 markdown

### Cell 2 code

```
from prophet.diagnostics import cross_validation,performance_metrics
```

### Cell 3 code

```
train = pd.read_csv(os.path.join(DIR,'train.csv'))
test = pd.read_csv(os.path.join(DIR,'test.csv'))
sample = pd.read_csv(os.path.join(DIR,'sample_submission.csv'))
gdp = pd.read_csv('./GDP_data_2015_to_2019_Finland_Norway_Sweden.csv').set_index('year')
```

### Cell 4 code

```
gdp_map = scaler.stack().to_dict()
```

### Cell 8 code

```
            res = pd.merge(res, d, on=['date','country','store','product'], how='outer')
sub.to_csv("submission.csv", index=None)
```
