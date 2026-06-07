# RecSys / h-and-m-personalized-fashion-recommendations

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 10 quality strong score 49

File: `RecSys/h-and-m-personalized-fashion-recommendations/rank10_70pct/samp-hm-1st-stage-demo-with-polars-gpu.ipynb`
Cells: 13 total, 10 code, 3 markdown

### Cell 0 markdown

```
4. Rank candidates with a weighted linear `candidate_score`.
```

### Cell 2 code

```
# Teaching/demo parameters. Set MAX_VALID_CUSTOMERS=None for a fuller run.
VALIDATION_CUTOFF = date(2020, 9, 8)
MAX_VALID_CUSTOMERS = 5_000
RANK_WEIGHT = 2.0
LABEL_END = VALIDATION_CUTOFF + timedelta(days=LABEL_DAYS)
EARLIEST_NEEDED_DATE = VALIDATION_CUTOFF - timedelta(days=max(HISTORY_DAYS, max(GLOBAL_POPULARITY_DAYS), AGE_POPULARITY_DAYS, CATEGORY_HISTORY_DAYS, CATEGORY_POPULARITY_DAYS))
DATA_DIR, EARLIEST_NEEDED_DATE, VALIDATION_CUTOFF, LABEL_END
```

### Cell 5 code

```
        .agg(pl.len().alias("cnt"))
        .agg(pl.col("article_idx").unique().alias("actual"))
        .agg(pl.len().alias("cnt"))
        .agg(pl.len().alias("cnt"), pl.max("t_dat").alias("last_date"))
```

### Cell 6 code

```
            .agg(pl.len().alias("cnt"), pl.max("t_dat").alias("last_date"))
```

### Cell 7 code

```
actuals_all = actuals_for_window(transactions, VALIDATION_CUTOFF, LABEL_DAYS)
valid_customers = sorted(actuals_all)
if MAX_VALID_CUSTOMERS is not None:
    valid_customers = valid_customers[:MAX_VALID_CUSTOMERS]
actuals = {customer_idx: actuals_all[customer_idx] for customer_idx in valid_customers}
    "history": build_customer_history(transactions, VALIDATION_CUTOFF, HISTORY_DAYS, HISTORY_TOP_N),
    "global_top": {days: build_top_list(transactions, VALIDATION_CUTOFF, days, GLOBAL_TOP_N) for days in GLOBAL_POPULARITY_DAYS},
    "age_top": build_age_top(transactions, customers, VALIDATION_CUTOFF, AGE_POPULARITY_DAYS, AGE_TOP_N),
    "category_top": build_category_top(transactions, articles, VALIDATION_CUTOFF, CATEGORY_POPULARITY_DAYS, CATEGORY_COLS, CATEGORY_TOP_N),
    "category_preferences": build_category_preferences(transactions, articles, VALIDATION_CUTOFF, CATEGORY_HISTORY_DAYS, CATEGORY_COLS, CATEGORY_PREF_TOP_N),
len(valid_customers), len(actuals_all)
```

### Cell 8 code

```
RANK_COLUMNS = ["history_rank", "age_rank", "global7_rank", "global14_rank", "global30_rank", "global60_rank", "category_rank"]
MISSING_RANK = 999
        "candidate_score": 0.0,
        **{col: MISSING_RANK for col in RANK_COLUMNS},
def score_piece(source, rank, count, recency_days=None):
    score = SOURCE_WEIGHTS[source] + RANK_WEIGHT / max(rank, 1) + COUNT_WEIGHT * math.log1p(max(count, 0))
        score += RECENCY_WEIGHT / (1.0 + max(recency_days, 0))
    return score
def add_candidate(candidates, article_idx, source, rank, count, recency_days=None):
    rank_col = "age_rank" if source == "age14" else f"{source}_rank"
    row[rank_col] = min(row[rank_col], rank)
    row["candidate_score"] += score_piece(source, rank, count, recency_days)
    for rank, (article_idx, count, recency_days) in enumerate(components["history"].get(customer_idx, []), start=1):
        add_candidate(candidates, article_idx, "history", rank, count, recency_days)
    for rank, (article_idx, count) in enumerate(components["age_top"].get(bucket, []), start=1):
        add_candidate(candidates, article_idx, "age14", rank, count)
    for rank, (article_idx, count) in enumerate(components["global_top"].get(7, []), start=1):
        add_candidate(candidates, article_idx, "global7", rank, count)
            for rank, (article_idx, count) in enumerate(components["category_top"][col].get(category, []), start=1):
                add_candidate(candidates, article_idx, "category", rank, count)
        for rank, (article_idx, count) in enumerate(components["global_top"].get(days, []), start=1):
            add_candidate(candidates, article_idx, source, rank, count)
    return sorted(candidates.items(), key=lambda item: (-item[1]["candidate_score"], min(item[1][col] for col in RANK_COLUMNS), item[0]))
```

### Cell 9 code

```
    score = 0.0
    for rank, article_idx in enumerate(predicted[:k], start=1):
            score += hits / rank
    return score / min(len(actual_set), k)
metric_acc = {k: {"customer_recall": [], "map": []} for k in EVAL_CUTOFFS}
for customer_idx in valid_customers:
        metric_acc[k]["customer_recall"].append(hits / len(actual_set) if actual_set else 0.0)
        metric_acc[k]["map"].append(apk(actual_set, prediction, k))
            "candidate_score": features["candidate_score"],
metrics = pl.DataFrame([
        "recall_customer_mean": sum(metric_acc[k]["customer_recall"]) / len(metric_acc[k]["customer_recall"]),
        "map": sum(metric_acc[k]["map"]) / len(metric_acc[k]["map"]),
metrics
```

### Cell 10 code

```
ks = metrics["k"].to_list()
recall_values = metrics["recall_customer_mean"].to_list()
map_values = metrics["map"].to_list()
```

### Cell 12 markdown

```
- MAP is sensitive to ordering, so `candidate_score` matters even before the 2nd-stage reranker.
- The 1st stage does not need to be a heavy model. A transparent weighted linear score is enough to teach the candidate-generation idea.
- The 2nd stage should receive this fixed candidate set and only rerank it.
```

## 40pct rank 2264 quality usable score 27

File: `RecSys/h-and-m-personalized-fashion-recommendations/rank2264_40pct/recommender-system.ipynb`
Cells: 4 total, 3 code, 1 markdown

### Cell 1 code

```
data = pd.read_csv(fname_tran, usecols=['customer_id', 'article_id', 'price'])
articles = pd.read_csv(fname_article)
data = data.groupby(['customer_id', 'article_id'], as_index=False).sum()
```

## 20pct rank 585 quality usable score 37

File: `RecSys/h-and-m-personalized-fashion-recommendations/rank585_20pct/notebookfirstrl.ipynb`
Cells: 39 total, 38 code, 0 markdown

### Cell 0 code

```
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```

### Cell 3 code

```
# create_folds.py
```

### Cell 4 code

```
import torch
from torch import nn
from torch.nn import functional as F
```

### Cell 7 code

```
df_transactions = pd.read_csv(transactions_train,
```

### Cell 9 code

```
df_customers = pd.read_csv(customers,
```

### Cell 12 code

```
df_products = pd.read_csv(products,
```

### Cell 14 code

```
embed_prod = torch.randn(len(df_products.index), len_embed)
embed_prod = embed_prod.to(device, dtype=torch.float)
```

### Cell 16 code

```
# embed_prod(torch.LongTensor([3]))
```

### Cell 17 code

```
# temp = embed_prod(torch.LongTensor([3,4])).squeeze()
```

### Cell 20 code

```
# input1 = torch.randn(100, 128)
```

### Cell 21 code

```
# input2 = torch.randn(100, 128)
```

### Cell 23 code

```
embed_custom = embed_custom.to(device, dtype=torch.float)
```

## 10pct rank 299 quality usable score 29

File: `RecSys/h-and-m-personalized-fashion-recommendations/rank299_10pct/lb-0-0236-ensemble-gives-you-bronze-medal.ipynb`
Cells: 10 total, 8 code, 2 markdown

### Cell 1 markdown

```
# Models from the following notebooks are used to create an ensemble
```

### Cell 2 code

```
sub2 = pd.read_csv('../input/handmbestperforming/hnm-exponential-decay-with-alternate-items.csv').sort_values('customer_id').reset_index(drop=True)
sub5 = pd.read_csv('../input/handmbestperforming/time-is-our-best-friend-v2.csv').sort_values('customer_id').reset_index(drop=True)
sub3 = pd.read_csv('../input/handmbestperforming/lstm-sequential-modelwith-item-features-tutorial.csv').sort_values('customer_id').reset_index(drop=True)
sub4 = pd.read_csv('../input/hm-00224-solution/submission.csv').sort_values('customer_id').reset_index(drop=True)
sub0 = pd.read_csv('../input/hm-00231-solution/submission.csv').sort_values('customer_id').reset_index(drop=True)
sub1 = pd.read_csv('../input/handmbestperforming/h-m-trending-products-weekly-add-test.csv').sort_values('customer_id').reset_index(drop=True)
sub6 = pd.read_csv('../input/handmbestperforming/rule-based-by-customer-age.csv').sort_values('customer_id').reset_index(drop=True)
```

### Cell 4 code

```
def cust_blend2(dt, weights):
```

### Cell 5 code

```
result = cust_blend2(sub0, w)
```

### Cell 9 code

```
sub0.to_csv('submission.csv', index=False)
```

## 1st rank 10 quality strong score 54

File: `RecSys/h-and-m-personalized-fashion-recommendations/rank10_1st/samp-hm-1st-stage-demo-with-polars-gpu.ipynb`
Cells: 13 total, 10 code, 3 markdown

### Cell 0 markdown

```
4. Rank candidates with a weighted linear `candidate_score`.
```

### Cell 2 code

```
# Teaching/demo parameters. Set MAX_VALID_CUSTOMERS=None for a fuller run.
VALIDATION_CUTOFF = date(2020, 9, 8)
MAX_VALID_CUSTOMERS = 5_000
RANK_WEIGHT = 2.0
LABEL_END = VALIDATION_CUTOFF + timedelta(days=LABEL_DAYS)
EARLIEST_NEEDED_DATE = VALIDATION_CUTOFF - timedelta(days=max(HISTORY_DAYS, max(GLOBAL_POPULARITY_DAYS), AGE_POPULARITY_DAYS, CATEGORY_HISTORY_DAYS, CATEGORY_POPULARITY_DAYS))
DATA_DIR, EARLIEST_NEEDED_DATE, VALIDATION_CUTOFF, LABEL_END
```

### Cell 5 code

```
        .agg(pl.len().alias("cnt"))
        .agg(pl.col("article_idx").unique().alias("actual"))
        .agg(pl.len().alias("cnt"))
        .agg(pl.len().alias("cnt"), pl.max("t_dat").alias("last_date"))
```

### Cell 6 code

```
            .agg(pl.len().alias("cnt"), pl.max("t_dat").alias("last_date"))
```

### Cell 7 code

```
actuals_all = actuals_for_window(transactions, VALIDATION_CUTOFF, LABEL_DAYS)
valid_customers = sorted(actuals_all)
if MAX_VALID_CUSTOMERS is not None:
    valid_customers = valid_customers[:MAX_VALID_CUSTOMERS]
actuals = {customer_idx: actuals_all[customer_idx] for customer_idx in valid_customers}
    "history": build_customer_history(transactions, VALIDATION_CUTOFF, HISTORY_DAYS, HISTORY_TOP_N),
    "global_top": {days: build_top_list(transactions, VALIDATION_CUTOFF, days, GLOBAL_TOP_N) for days in GLOBAL_POPULARITY_DAYS},
    "age_top": build_age_top(transactions, customers, VALIDATION_CUTOFF, AGE_POPULARITY_DAYS, AGE_TOP_N),
    "category_top": build_category_top(transactions, articles, VALIDATION_CUTOFF, CATEGORY_POPULARITY_DAYS, CATEGORY_COLS, CATEGORY_TOP_N),
    "category_preferences": build_category_preferences(transactions, articles, VALIDATION_CUTOFF, CATEGORY_HISTORY_DAYS, CATEGORY_COLS, CATEGORY_PREF_TOP_N),
len(valid_customers), len(actuals_all)
```

### Cell 8 code

```
RANK_COLUMNS = ["history_rank", "age_rank", "global7_rank", "global14_rank", "global30_rank", "global60_rank", "category_rank"]
MISSING_RANK = 999
        "candidate_score": 0.0,
        **{col: MISSING_RANK for col in RANK_COLUMNS},
def score_piece(source, rank, count, recency_days=None):
    score = SOURCE_WEIGHTS[source] + RANK_WEIGHT / max(rank, 1) + COUNT_WEIGHT * math.log1p(max(count, 0))
        score += RECENCY_WEIGHT / (1.0 + max(recency_days, 0))
    return score
def add_candidate(candidates, article_idx, source, rank, count, recency_days=None):
    rank_col = "age_rank" if source == "age14" else f"{source}_rank"
    row[rank_col] = min(row[rank_col], rank)
    row["candidate_score"] += score_piece(source, rank, count, recency_days)
    for rank, (article_idx, count, recency_days) in enumerate(components["history"].get(customer_idx, []), start=1):
        add_candidate(candidates, article_idx, "history", rank, count, recency_days)
    for rank, (article_idx, count) in enumerate(components["age_top"].get(bucket, []), start=1):
        add_candidate(candidates, article_idx, "age14", rank, count)
    for rank, (article_idx, count) in enumerate(components["global_top"].get(7, []), start=1):
        add_candidate(candidates, article_idx, "global7", rank, count)
            for rank, (article_idx, count) in enumerate(components["category_top"][col].get(category, []), start=1):
                add_candidate(candidates, article_idx, "category", rank, count)
        for rank, (article_idx, count) in enumerate(components["global_top"].get(days, []), start=1):
            add_candidate(candidates, article_idx, source, rank, count)
    return sorted(candidates.items(), key=lambda item: (-item[1]["candidate_score"], min(item[1][col] for col in RANK_COLUMNS), item[0]))
```

### Cell 9 code

```
    score = 0.0
    for rank, article_idx in enumerate(predicted[:k], start=1):
            score += hits / rank
    return score / min(len(actual_set), k)
metric_acc = {k: {"customer_recall": [], "map": []} for k in EVAL_CUTOFFS}
for customer_idx in valid_customers:
        metric_acc[k]["customer_recall"].append(hits / len(actual_set) if actual_set else 0.0)
        metric_acc[k]["map"].append(apk(actual_set, prediction, k))
            "candidate_score": features["candidate_score"],
metrics = pl.DataFrame([
        "recall_customer_mean": sum(metric_acc[k]["customer_recall"]) / len(metric_acc[k]["customer_recall"]),
        "map": sum(metric_acc[k]["map"]) / len(metric_acc[k]["map"]),
metrics
```

### Cell 10 code

```
ks = metrics["k"].to_list()
recall_values = metrics["recall_customer_mean"].to_list()
map_values = metrics["map"].to_list()
```

### Cell 12 markdown

```
- MAP is sensitive to ordering, so `candidate_score` matters even before the 2nd-stage reranker.
- The 1st stage does not need to be a heavy model. A transparent weighted linear score is enough to teach the candidate-generation idea.
- The 2nd stage should receive this fixed candidate set and only rerank it.
```
