# RecSys / otto-recommender-system

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1962 quality weak score 20

File: `RecSys/otto-recommender-system/rank1962_70pct/ott-submission.ipynb`
Cells: 44 total, 23 code, 21 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```

### Cell 9 markdown

```
### Recommendation Systems Overview -  One common architecture for recommendation systems consists of the following components: candidate generation , scoring , re-ranking .
Scoring - Next, another model scores and ranks the candidates in order to select the set of items (on the order of 10) to display to the user .
Re-ranking - Finally, the system must take into account additional constraints for the final ranking. For example, the system removes items that the user explicitly disliked or boosts the score of fresher content. Re-ranking can also help ensure diversity, freshness, and fairness.
```

### Cell 10 markdown

```
### Candidate Generation Overview -


Candidate generation is the first stage of recommendation. Given a query, the system generates a set of relevant candidates. The following table shows two common candidate generation approaches:

content-based filtering - 	Uses similarity between items to recommend items similar to what the user likes.

collaborative filtering - Uses similarities between queries and items simultaneously to provide recommendations.
```

### Cell 14 markdown

```
# Content-based Filtering -
Content-based filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback.


# Content-based Filtering Advantages -
The model doesn't need any data about other users, since the recommendations are specific to this user. This makes it easier to scale to a large number of users.

The model can capture the specific interests of a user, and can recommend niche items that very few other users are interested in.


# Content-based Filtering Disadvantages -
Since the feature representation of the items are hand-engineered to some extent, this technique requires a lot of domain knowledge. Therefore, the model can only be as good as the hand-engineered features.

The model can only make recommendations based on existing interests of the user. In other words, the model has limited ability to expand on the users' existing interests.
```

### Cell 15 markdown

```
# Collaborative Filtering -
To address some of the limitations of content-based filtering, collaborative filtering uses similarities between users and items simultaneously to provide recommendations. This allows for serendipitous recommendations; that is, collaborative filtering models can recommend an item to user A based on the interests of a similar user B. Furthermore, the embeddings can be learned automatically, without relying on hand-engineering of features.
```

### Cell 28 code

```
from sklearn.metrics.pairwise import linear_kernel
```

### Cell 37 code

```
df = pd.DataFrame(R_estimated).to_csv("sample.csv")
df = pd.read_csv("sample.csv")
```

### Cell 38 code

```
submission = pd.read_csv('/kaggle/input/otto-recommender-system/sample_submission.csv')
```

### Cell 43 code

```
submission.to_csv('submission.csv', index=False, header=True)
```

## 40pct rank 931 quality strong score 47

File: `RecSys/otto-recommender-system/rank931_40pct/otto-lightgbm-brief-sample-code-with-cudf.ipynb`
Cells: 20 total, 11 code, 9 markdown

### Cell 1 markdown

```
- https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575
- https://www.kaggle.com/code/greenwolf/lightgbm-fast-recall-20/notebook
```

### Cell 3 markdown

```
- LightGBM ranking training & validation
- Brief commentary for LightGBM parameter `group`
```

### Cell 5 code

```
import lightgbm as lgb
```

### Cell 6 markdown

```
    📌For description of loading data, read following notebook: <a href="https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575" target="_blank" style="color:#ff7f7f;">Candidate ReRank Model - [LB 0.575] by cdeotte</a>
```

### Cell 9 code

```
        df['n'] = df.groupby('session').cumcount()
```

### Cell 11 code

```
fold_size = len(df_concat) // 5
```

### Cell 13 code

```
val_df = df_concat.iloc[:fold_size]
train_df = df_concat.iloc[fold_size:]
```

### Cell 14 code

```
train_query = train_df.groupby(by='session', sort=True).size()
val_query = val_df.groupby(by='session', sort=True).size()
```

### Cell 18 code

```
    'objective': 'lambdarank',
#     'metric': '"None"', # if use custom metric, specify "None"
    valid_sets=val_dataset,
#     feval=custom_metric_func,
```

## 20pct rank 636 quality usable score 25

File: `RecSys/otto-recommender-system/rank636_20pct/easy-recommendation-pandas.ipynb`
Cells: 6 total, 3 code, 3 markdown

### Cell 0 markdown

```
The only product that needs to be predicted for the click prediction is the **next one** the user will click, for instances: aid1. However, we can submit a list of up to 20 products for prediction; **as long as aid1 is on the list, the score will be awarded.**
The maximum size of a prediction is 20, so **even though there are 30 products in the cart, the full score can still be obtained as long as every item on the prediction list appears in the cart**.
Since 3 products were ordered, I can still **receive a full score as long as all three appear on the prediction list.**
```

### Cell 5 code

```
df = df.sort_values(["session", "type", "ts"]).groupby(["session"]).apply(
submission.to_csv("submission.csv", index=False)
```

## 10pct rank 227 quality usable score 37

File: `RecSys/otto-recommender-system/rank227_10pct/otto-tr-matrixv2-tail40-top404050-w136.ipynb`
Cells: 12 total, 7 code, 5 markdown

### Cell 0 markdown

```
1. This notebook is originally derived from Chris Deotte notebook at https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575
2. Few modifications were made in the forked version at https://www.kaggle.com/code/hoangnguyen719/candidate-rerank-model-lb-0-575
3. This notebook, forked from the forked version (2), provides covisitation matrices built using only `train` set. They will be used in training and validation.
```

### Cell 2 code

```
CLICK_TIME_TOP = 50 # Top candidates for matrix 3 - time-weighted click
```

### Cell 7 code

```
    # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
            df['n'] = df.groupby('session').cumcount()
            df = df.merge(df,on='session')
            df = df.groupby(['aid_x','aid_y']).wgt.sum()
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
```

### Cell 9 code

```
    # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
            df['n'] = df.groupby('session').cumcount()
            df = df.merge(df,on='session')
            df = df.groupby(['aid_x','aid_y']).wgt.sum()
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
```

### Cell 11 code

```
    # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
            df['n'] = df.groupby('session').cumcount()
            df = df.merge(df,on='session')
            df = df.groupby(['aid_x','aid_y']).wgt.sum()
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
```

## 1st rank 3 quality strong score 52

File: `RecSys/otto-recommender-system/rank3_1st/3rd-place-team-g-b-d-t-0-604.ipynb`
Cells: 10 total, 8 code, 2 markdown

### Cell 0 markdown

```
This notebook is the 3rd place final submission for Team G&B&D&T for Kaggle's OTTO RecSys competition. Team members are Giba ( @titericz ), Benny ( @benediktschifferer ), Chris Deotte ( @cdeotte ), and Theo ( @theoviel ). This solution is an ensemble of 3 single XGB reranker models. Discussion explaining this solution is [here][1], [here][2], and [here][3].
```

### Cell 3 code

```
    pd.read_csv(f).sort_values(['session_type']).reset_index(drop=True) for f in FILES
```

### Cell 6 code

```
blend = []
    blend.append(aid)
```

### Cell 7 code

```
sub['labels'] = blend
sub.to_csv('submission.csv', index=False)
```

### Cell 8 code

```
        print(f'Similarity of row {subs[0]["session_type"][idx]} between sub {j} & blend : {sim :.3f}')
```
