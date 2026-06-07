# NLP / feedback-prize-english-language-learning

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1872 quality strong score 73

File: `NLP/feedback-prize-english-language-learning/rank1872_70pct/compe-nlp-feedback-english.ipynb`
Cells: 23 total, 17 code, 6 markdown

### Cell 2 code

```
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
```

### Cell 4 code

```
    use_k_fold = True
```

### Cell 6 code

```
train = pd.read_csv(f"{config.dataset_path}/train.csv")
test = pd.read_csv(f"{config.dataset_path}/test.csv")
```

### Cell 12 code

```
vectorizor = keras.layers.TextVectorization(
```

### Cell 13 markdown

```
## building model
```

### Cell 14 code

```
    model = keras.Sequential([
        keras.Input(shape=(), dtype="string"),
        keras.layers.Dense(64, kernel_initializer='he_uniform', activation='swish'),
        keras.layers.Dense(32, kernel_initializer='he_uniform', activation='swish'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(len(config.target_columns))
    rmse = tf.keras.metrics.RootMeanSquaredError(name="rmse")
    model.compile(loss="mse", optimizer="adam", metrics=[rmse])
```

### Cell 16 code

```
keras.utils.plot_model(model, show_shapes=True)
```

### Cell 17 code

```
keras.backend.clear_session()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_indices, valid_indices) in enumerate(kfold.split(train_1)):
    x_val = train_1.iloc[valid_indices]["text"]
    y_val = train_1.iloc[valid_indices][config.target_columns]
    rmse = tf.keras.metrics.RootMeanSquaredError(name="rmse")
    checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor="val_rmse", mode="min", save_best_only=True, save_weights_only=True)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_rmse", mode="min", patience=5)
    model.compile(loss="mse", optimizer="adam", metrics=[rmse])
        validation_data=(x_val, y_val),
    if not config.use_k_fold:
```

### Cell 18 code

```
submission.to_csv("submission.csv", index=False)
```

## 40pct rank 1131 quality strong score 59

File: `NLP/feedback-prize-english-language-learning/rank1131_40pct/feedback-prize-ell-baseline.ipynb`
Cells: 15 total, 15 code, 0 markdown

### Cell 0 code

```
import tensorflow as tf
```

### Cell 2 code

```
train_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_data = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
```

### Cell 4 code

```
def train_test_split(data, validation_split=0.1):
    n_val = int(len(text) * validation_split)
```

### Cell 5 code

```
validation_split = 0.1
train_text, train_targets, val_text, val_targets = train_test_split(train_data, validation_split)
```

### Cell 6 code

```
# Create training and validation tf datasets.
validation_dataset = tf.data.Dataset.from_tensor_slices((val_text, val_targets)).shuffle(BUFFER_SIZE_VAL)
validation_dataset = validation_dataset.batch(BATCH_SIZE)
```

### Cell 7 code

```
text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens,
```

### Cell 8 code

```
val_ds = validation_dataset.map(lambda text, targets: (text_vectorizer(text), targets / 5))
```

### Cell 10 code

```
    model = tf.keras.Sequential([tf.keras.layers.Input((output_sequence_length,)),
                                tf.keras.layers.Embedding(text_vectorizer.vocabulary_size(),
                                tf.keras.layers.GlobalAveragePooling1D(),
                                tf.keras.layers.Dense(len(target_columns))])
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
```

### Cell 12 code

```
es_callback = tf.keras.callbacks.EarlyStopping(patience=5)
validation_split = 0.1
model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=2, callbacks=[es_callback])
```

### Cell 14 code

```
submission = sample_submission.copy()
submission.to_csv('submission.csv', index=False)
```

## 20pct rank 415 quality strong score 57

File: `NLP/feedback-prize-english-language-learning/rank415_20pct/english.ipynb`
Cells: 6 total, 4 code, 2 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.linear_model import Ridge, Perceptron, BayesianRidge, ElasticNet, Lasso
import sklearn.ensemble as ensemble
```

### Cell 3 code

```
train_df = pd.read_csv('/kaggle/input/feedback-prize-english-language-learning/train.csv')
test_df = pd.read_csv('/kaggle/input/feedback-prize-english-language-learning/test.csv')
# for bayesian ridge
# model = Ridge(alpha=10000)
# model = MultiOutputRegressor(ensemble.GradientBoostingRegressor(learning_rate=0.1))
# model = BayesianRidge()
    ### see https://stackoverflow.com/questions/45346550/valueerror-unknown-label-type-unknown
    cross_validate(pipe, X_train, np.multiply(y_train, 2).astype('int'), return_train_score=True)
```

### Cell 4 markdown

```
Now we need to export this dataframe to csv.

Use the model and get results.
```

### Cell 5 code

```
results.to_csv('/kaggle/working/submission.csv', index=False)
```

## 10pct rank 238 quality strong score 73

File: `NLP/feedback-prize-english-language-learning/rank238_10pct/fb3-deberta-family-inference-weight-tune.ipynb`
Cells: 18 total, 9 code, 9 markdown

### Cell 1 code

```
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import DataCollatorWithPadding
%env TOKENIZERS_PARALLELISM=false
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Cell 3 markdown

```
Deberta Family ver. 13
* CFG1 : 10 fold deberta-v3-base CV/LB: 0.4595/0.44
* CFG2 : 10 fold deberta-v3-large CV/LB: 0.4553/0.44
* CFG3 : 10 fold deberta-v2-xlarge CV/LB: 0.4604/0.44
* CFG4 : 10 fold deberta-v3-base FGM CV/LB: 0.4590/0.44
* CFG5 : 10 fold deberta-v3-large FGM CV/LB: 0.4564/0.44
* CFG6 : 10 fold deberta-v2-xlarge CV/LB: 0.4666/0.44
* CFG7 : 10 fold deberta-v2-xlarge-mnli CV/LB: 0.4675/0.44
* CFG8 : 10 fold deberta-v3-large unscale CV/LB: 0.4616/0.43
* CFG9 : 10 fold deberta-v3-large unscale CV/LB: 0.4548/0.43
* CFG10 : 10 fold deberta-v3-large unscale CV/LB: 0.4569/0.43
```

### Cell 4 code

```
    model = "microsoft/deberta-v3-base"
    path = "../input/0911-deberta-v3-base/"
    base = "../input/fb3models/microsoft-deberta-v3-base/"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    n_fold=10
    trn_fold=list(range(n_fold))
    model = "microsoft/deberta-v3-large"
    path = "../input/0911-deberta-v3-large/"
    base = "../input/fb3models/microsoft-deberta-v3-large/"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    n_fold=10
    trn_fold=list(range(n_fold))
    model = "microsoft/deberta-v2-xlarge"
    path = "../input/0911-deberta-v2-xlarge/"
    base = "../input/fb3models/microsoft-deberta-v2-xlarge/"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    n_fold=10
    trn_fold=list(range(n_fold))
    model = "microsoft/deberta-v3-base"
    path = "../input/0913-deberta-v3-base-fgm/"
    base = "../input/fb3models/microsoft-deberta-v3-base/"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    n_fold=10
    trn_fold=list(range(n_fold))
    model = "microsoft/deberta-v3-large"
    path = "../input/0914-deberta-v3-large-fgm/"
    base = "../input/fb3models/microsoft-deberta-v3-large/"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    n_fold=10
    trn_fold=list(range(n_fold))
    model = "microsoft/deberta-v2-xlarge"
    path = "../input/0919-deberta-v2-xlarge/"
    base = "../input/fb3models/microsoft-deberta-v2-xlarge/"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    n_fold=10
    trn_fold=list(range(n_fold))
    model = "microsoft/deberta-v2-xlarge-mnli"
    path = "../input/0919-deberta-v2-xlarge-mnli/"
    base = "../input/fb3models/microsoft-deberta-v2-xlarge/"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    n_
...[truncated]
```

### Cell 6 code

```
    scores = []
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores
def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
```

### Cell 8 code

```
# oof
    oof_df = pd.read_pickle(CFG.path+'oof_df.pkl')
    labels = oof_df[CFG.target_cols].values
    preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
    score, scores = get_score(labels, preds)
    LOGGER.info(f'Model: {CFG.model} Score: {score:<.4f}  Scores: {scores}')
```

### Cell 10 code

```
    inputs = cfg.tokenizer.encode_plus(
        inputs[k] = torch.tensor(v, dtype=torch.long)
```

### Cell 11 markdown

```
# Model
```

### Cell 12 code

```
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        min_embeddings, _ = torch.min(embeddings, dim = 1)
```

### Cell 14 code

```
        with torch.no_grad():
```

### Cell 15 code

```
    test = pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')
    submission = pd.read_csv('../input/feedback-prize-english-language-learning/sample_submission.csv')
    test['tokenize_length'] = [len(CFG.tokenizer(text)['input_ids']) for text in test['full_text'].values]
                             collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, padding='longest'),
    for fold in CFG.trn_fold:
        state = torch.load(CFG.path+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                           map_location=torch.device('cpu'))
        torch.cuda.empty_cache()
    submission = submission.drop(columns=CFG.target_cols).merge(test[['text_id'] + CFG.target_cols], on='text_id', how='left')
    submission[['text_id'] + CFG.target_cols].to_csv(f'submission_{_idx + 1}.csv', index=False)
    torch.cuda.empty_cache()
```

### Cell 17 code

```
test = pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')
submission = pd.read_csv('../input/feedback-prize-english-language-learning/sample_submission.csv')
sub1 = pd.read_csv(f'submission_1.csv')[CFG1.target_cols] * CFG1.weight
sub2 = pd.read_csv(f'submission_2.csv')[CFG2.target_cols] * CFG2.weight
sub3 = pd.read_csv(f'submission_3.csv')[CFG3.target_cols] * CFG3.weight
sub4 = pd.read_csv(f'submission_4.csv')[CFG4.target_cols] * CFG4.weight
sub5 = pd.read_csv(f'submission_5.csv')[CFG5.target_cols] * CFG5.weight
sub6 = pd.read_csv(f'submission_6.csv')[CFG6.target_cols] * CFG6.weight
sub7 = pd.read_csv(f'submission_7.csv')[CFG7.target_cols] * CFG7.weight
sub8 = pd.read_csv(f'submission_8.csv')[CFG8.target_cols] * CFG8.weight
sub9 = pd.read_csv(f'submission_9.csv')[CFG9.target_cols] * CFG9.weight
sub10 = pd.read_csv(f'submission_10.csv')[CFG10.target_cols] * CFG10.weight
submission.to_csv('submission.csv', index=False)
```

## 1st rank 8 quality reject score -11

File: `NLP/feedback-prize-english-language-learning/rank8_1st/llm-detect-pip.ipynb`
Cells: 2 total, 2 code, 0 markdown

### Cell 0 code

```
!pip download transformers
```
