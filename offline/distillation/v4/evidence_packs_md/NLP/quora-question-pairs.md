# NLP / quora-question-pairs

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 21 quality reject score -10

File: `NLP/quora-question-pairs/rank21_70pct/kaggle-book-gokui-chapter5-lightgbm.ipynb`
Cells: 5 total, 4 code, 1 markdown

### Cell 4 code

```
!python experiments/003_gbm_calibration.py --needs_predict --fold_all
```

## 40pct rank 1316 quality strong score 54

File: `NLP/quora-question-pairs/rank1316_40pct/xgb-starter-23-clean-text.py`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
import xgboost as xgb
from sklearn.cross_validation import train_test_split
input_folder = '../input/'
def train_xgb(X, y, params):
    print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
    xg_train = xgb.DMatrix(x, label=y_train)
    xg_val = xgb.DMatrix(X_val, label=y_val)
    return xgb.train(params, xg_train, ROUNDS, watchlist)
def predict_xgb(clr, X_test):
    return clr.predict(xgb.DMatrix(X_test))
    outfile = open('xgb.fmap', 'w')
    params['eval_metric'] = 'logloss'
    df_train = pd.read_csv(input_folder + 'train.csv')
    df_test = pd.read_csv(input_folder + 'test.csv')
            R1 = 0  # tfidf share
            R1 = np.sum(shared_weights) / sumtw  # tfidf share
    x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
    x['cross_idf'] = x['word_match'] *x['tfidf_word_match'] ;
    x['sum_idf'] = x['word_match'] + x['tfidf_word_match'] ;
    x['dif_idf'] = x['word_match'] - x['tfidf_word_match'] ;
    clr = train_xgb(x_train, y_train, params)
    preds = predict_xgb(clr, x_test)
    sub.to_csv("xgb_seed{}_n{}.csv".format(RS, ROUNDS), index=False)
    importance = clr.get_fscore(fmap='xgb.fmap')
    ft = pd.DataFrame(importance, columns=['feature', 'fscore'])
    ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
```

## 20pct rank 664 quality strong score 49

File: `NLP/quora-question-pairs/rank664_20pct/quoras-question-pairs-sentiment-analysis.ipynb`
Cells: 15 total, 11 code, 4 markdown

### Cell 0 markdown

```
# Quoras Question Pairs Sentiment Analysis

Is sentiment a useful feature in determining similarity of question pairs?
```

### Cell 1 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score
#text cleaning borrowed from https://www.kaggle.com/currie32/quora-question-pairs/an-ensemble-approach-cnn-and-timedistributed
```

### Cell 2 code

```
train = pd.read_csv('../input/train.csv')
```

### Cell 5 code

```
train['senti1'] = train['question1'].apply(lambda x: sid.polarity_scores(x)['compound'])
train['senti2'] = train['question2'].apply(lambda x: sid.polarity_scores(x)['compound'])
```

### Cell 7 markdown

```
15% correlation is not nothing.  Seems like a useful feature.
```

### Cell 13 code

```
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(raw)
```

## 10pct rank 332 quality weak score 11

File: `NLP/quora-question-pairs/rank332_10pct/data-analysis-xgboost-starter-0-35460-lb-12d8b0.irnb`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
{"cells":[{"cell_type":"markdown","metadata":{"_cell_guid":"9c355c2c-044d-2b87-e1cf-a6732fb9e802"},"source":"# Identifying Duplicate Questions\n\nWelcome to the Quora Question Pairs competition! Here, our goal is to identify which questions asked on [Quora](https://www.quora.com/), a quasi-forum website with over 100 million visitors a month, are duplicates of questions that have already been asked. This could be useful, for example, to instantly provide answers to questions that have already been answered. We are tasked with predicting whether a pair of questions are duplicates or not, and submitting a binary prediction against the logloss metric.\n\nIf you have any questions or want to discuss competitions/hardware/games/anything with other Kagglers, then join the KaggleNoobs Slack channel [here](https://goo.gl/gGWFXe). We also have regular AMAs with top Kagglers there.\n\n**And as always, if this helped you, some upvotes would be very much appreciated - that's where I get my motivation! :D**\n\nLet's dive right into the data!"},{"cell_type":"code","execution_count":null,"metadata":{"_cell_guid":"d9f5b5bf-b8a8-a88a-a16c-7666be39bd7e"},"outputs":[],"source":"import numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\nimport os\nimport gc\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n%matplotlib inline\n\npal = sns.color_palette()\n\nprint('# File sizes')\nfor f in os.listdir('../input'):\n    if 'zip' not in f:\n        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')"},{"cell_type":"markdown","metadata":{"_cell_guid":"e9b03ade-149a-7d6a-d954-2db4ebb7806f"},"source":"Looks like we are simply given two files this time round, one for the training set and one for the test set.
...[truncated]
```

## 1st rank 21 quality reject score -5

File: `NLP/quora-question-pairs/rank21_1st/kaggle-book-gokui-chapter5-lightgbm.ipynb`
Cells: 5 total, 4 code, 1 markdown

### Cell 4 code

```
!python experiments/003_gbm_calibration.py --needs_predict --fold_all
```
