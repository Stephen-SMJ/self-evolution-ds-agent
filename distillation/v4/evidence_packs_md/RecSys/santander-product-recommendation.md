# RecSys / santander-product-recommendation

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1246 quality usable score 31

File: `RecSys/santander-product-recommendation/rank1246_70pct/data-visualization-2.ipynb`
Cells: 29 total, 29 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
```

### Cell 1 code

```
df = pd.read_csv('../input/train_ver2.csv',usecols=['ncodpers'])
```

### Cell 3 code

```
customer_count = df.groupby('ncodpers').agg('size').value_counts()
```

### Cell 4 code

```
train = pd.read_csv("../input/train_ver2.csv", dtype='float16',
```

### Cell 7 code

```
train = pd.read_csv('../input/train_ver2.csv',usecols=['fecha_dato'],parse_dates=['fecha_dato'])
```

### Cell 10 code

```
train = pd.read_csv('../input/train_ver2.csv',usecols=['fecha_alta'],parse_dates=['fecha_alta'])
```

### Cell 14 code

```
train = pd.read_csv('../input/train_ver2.csv',usecols=['age'])
```

### Cell 21 code

```
train = pd.read_csv('../input/train_ver2.csv',usecols = ['renta'])
```

## 40pct rank 707 quality strong score 48

File: `RecSys/santander-product-recommendation/rank707_40pct/when-less-is-more-asdsd268.py`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
import xgboost as xgb
from sklearn import preprocessing, ensemble
def getTarget(row):
			target_list = getTarget(row)
			target_list = getTarget(row)
def runXGB(train_X, train_y, seed_val=0):
	param['eval_metric'] = "mlogloss"
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	model = xgb.train(plst, xgtrain, num_rounds)
	model = runXGB(train_X, train_y, seed_val=0)
	xgtest = xgb.DMatrix(test_X)
	preds = np.argsort(preds, axis=1)
	test_id = np.array(pd.read_csv(data_path + "test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
	out_df.to_csv('sub_xgb_new.csv', index=False)
```

## 20pct rank 356 quality strong score 48

File: `RecSys/santander-product-recommendation/rank356_20pct/test-srk.py`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
import xgboost as xgb
from sklearn import preprocessing, ensemble
def getTarget(row):
    'GUADALAJARA': 100635,  'HUELVA': 75534,  'HUESCA': 80324,  'JAEN': 67016,  'LEON': 76339,  'LERIDA': 59191,  'LUGO': 68219,  'MADRID': 141381,  'MALAGA': 89534,  'MELILLA': 116469, 'GIPUZKOA': 101850,
def processDataMK(in_file_name, cust_dict, lag_cust_dict):
            target_list = getTarget(row)
            lag_cust_dict[cust_id] =  target_list[:]
            target_list = getTarget(row)
            lag_target_list = lag_cust_dict.get(cust_id, [0]*22)
            x_vars_list.append(x_vars + prev_target_list + lag_target_list)
            lag_target_list = lag_cust_dict.get(cust_id, [0]*22)
            target_list = getTarget(row)
                        x_vars_list.append(x_vars+prev_target_list+lag_target_list)
    return x_vars_list, y_vars_list, cust_dict, lag_cust_dict
			target_list = getTarget(row)
			target_list = getTarget(row)
def runXGB(train_X, train_y, seed_val=42):
	param['eval_metric'] = "mlogloss"
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	model = xgb.train(plst, xgtrain, num_rounds)
	x_vars_list, y_vars_list, cust_dict, lag_cust_dict = processDataMK(train_file, {}, {})
	x_vars_list, y_vars_list, cust_dict, lag_cust_dict = processDataMK(test_file, cust_dict, lag_cust_dict)
	model = runXGB(train_X, train_y, seed_val=0)
	xgtest = xgb.DMatrix(test_X)
	test_id = np.array(pd.read_csv(data_path + "test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
	preds = np.argsort(np.array(new_products), axis=1)
	out_df.to_csv('sub_xgb_new.csv', index=False)
```

## 10pct rank 183 quality weak score 23

File: `RecSys/santander-product-recommendation/rank183_10pct/exploratory-data-analysis-in-korean-180801.ipynb`
Cells: 12 total, 12 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
```

### Cell 1 code

```
trn = pd.read_csv('../input/train_ver2.csv')
```

### Cell 10 code

```
    label_sum = trn.groupby(['fecha_dato'])[label_cols[i]].agg('sum')
```

## 1st rank 5 quality strong score 53

File: `RecSys/santander-product-recommendation/rank5_1st/when-less-is-more-1.py`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
import xgboost as xgb
from sklearn import preprocessing, ensemble
def getTarget(row):
			target_list = getTarget(row)
			target_list = getTarget(row)
def runXGB(train_X, train_y, seed_val=123):
	param['eval_metric'] = "mlogloss"
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	model = xgb.train(plst, xgtrain, num_rounds)
	model = runXGB(train_X, train_y, seed_val=0)
	xgtest = xgb.DMatrix(test_X)
	preds = np.argsort(preds, axis=1)
	test_id = np.array(pd.read_csv(data_path + "test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
	out_df.to_csv('sub_xgb_new.csv', index=False)
```
