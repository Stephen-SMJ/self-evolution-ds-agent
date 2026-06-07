# NLP / google-quest-challenge

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1068 quality usable score 41

File: `NLP/google-quest-challenge/rank1068_70pct/google-quest-spelling-feature.ipynb`
Cells: 20 total, 13 code, 7 markdown

### Cell 0 markdown

```
In this notebook i show one approach for predicting the 'question_type_spelling' label
by building a custom feature using L2distance between USE embeddings of
* a magic sentence and
* input 'answer' data
```

### Cell 1 code

```
import tensorflow_hub as hub
```

### Cell 3 code

```
df_train = pd.read_csv(PATH+'train.csv')
```

### Cell 4 markdown

```
We find how many rows match for this feature 'question_type_spelling' in training data
```

### Cell 8 code

```
    embeddings_train[text + '_embedding'] = np.vstack(curr_train_emb)
```

### Cell 11 code

```
  spell_embeddings_train['spell_embedding'] = np.vstack(curr_train_emb)
```

### Cell 13 code

```
df_comp = pd.DataFrame(np.hstack([dist_features_train]),
```

## 40pct rank 275 quality usable score 43

File: `NLP/google-quest-challenge/rank275_40pct/eda-and-feature-engineering.ipynb`
Cells: 64 total, 33 code, 31 markdown

### Cell 1 code

```
from sklearn.feature_extraction.text import TfidfVectorizer
```

### Cell 4 code

```
train_data = pd.read_csv('../input/google-quest-challenge/train.csv')
test_data = pd.read_csv('../input/google-quest-challenge/test.csv')
sample_submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
```

### Cell 5 code

```
print('Size of sample_submission', sample_submission.shape)
```

### Cell 15 code

```
sample_submission.head()
```

### Cell 17 code

```
targets = list(sample_submission.columns[1:])
```

### Cell 48 markdown

```
# <a id='6'>6. Data Preparation & Feature Engineering</a>
```

### Cell 50 code

```
#https://www.kaggle.com/urvishp80/quest-encoding-ensemble
```

### Cell 56 markdown

```
## <a id='6-3'>6.3 Feature Engineering</a>
```

### Cell 57 markdown

```
### <a id='6-3-1'>6.3.1 Text based features</a>
```

### Cell 58 markdown

```
Text based features are :
 * Number of characters in the question_title
 * Number of characters in the question_body
 * Number of characters in the answer
 * Number of words in the question_title
 * Number of words in the question_body
 * Number of words in the answer
 * Number of unique words in the question_title
 * Number of unique words in the question_body
 * Number of unique words in the answer
```

### Cell 60 markdown

```
### <a id='6-3-2'>6.3.2 TF-IDF Features</a>
```

### Cell 61 markdown

```
#### TF-IDF :
  *  Term Frequency (TF) and Inverse Document Frequency (IDF)
  *  TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
  *  IDF(t) = log_e(Total number of documents / Number of documents with term t in it)

**TF-IDF based features are :**

* Character Level N-Gram TF-IDF of question_title
* Character Level N-Gram TF-IDF of question_body
* Character Level N-Gram TF-IDF of answer
* Word Level N-Gram TF-IDF of question_title
* Word Level N-Gram TF-IDF of question_body
* Word Level N-Gram TF-IDF of answer
```

## 20pct rank 275 quality usable score 43

File: `NLP/google-quest-challenge/rank275_20pct/eda-and-feature-engineering.ipynb`
Cells: 64 total, 33 code, 31 markdown

### Cell 1 code

```
from sklearn.feature_extraction.text import TfidfVectorizer
```

### Cell 4 code

```
train_data = pd.read_csv('../input/google-quest-challenge/train.csv')
test_data = pd.read_csv('../input/google-quest-challenge/test.csv')
sample_submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
```

### Cell 5 code

```
print('Size of sample_submission', sample_submission.shape)
```

### Cell 15 code

```
sample_submission.head()
```

### Cell 17 code

```
targets = list(sample_submission.columns[1:])
```

### Cell 48 markdown

```
# <a id='6'>6. Data Preparation & Feature Engineering</a>
```

### Cell 50 code

```
#https://www.kaggle.com/urvishp80/quest-encoding-ensemble
```

### Cell 56 markdown

```
## <a id='6-3'>6.3 Feature Engineering</a>
```

### Cell 57 markdown

```
### <a id='6-3-1'>6.3.1 Text based features</a>
```

### Cell 58 markdown

```
Text based features are :
 * Number of characters in the question_title
 * Number of characters in the question_body
 * Number of characters in the answer
 * Number of words in the question_title
 * Number of words in the question_body
 * Number of words in the answer
 * Number of unique words in the question_title
 * Number of unique words in the question_body
 * Number of unique words in the answer
```

### Cell 60 markdown

```
### <a id='6-3-2'>6.3.2 TF-IDF Features</a>
```

### Cell 61 markdown

```
#### TF-IDF :
  *  Term Frequency (TF) and Inverse Document Frequency (IDF)
  *  TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
  *  IDF(t) = log_e(Total number of documents / Number of documents with term t in it)

**TF-IDF based features are :**

* Character Level N-Gram TF-IDF of question_title
* Character Level N-Gram TF-IDF of question_body
* Character Level N-Gram TF-IDF of answer
* Word Level N-Gram TF-IDF of question_title
* Word Level N-Gram TF-IDF of question_body
* Word Level N-Gram TF-IDF of answer
```

## 10pct rank 217 quality strong score 67

File: `NLP/google-quest-challenge/rank217_10pct/end-to-end-bert-training.ipynb`
Cells: 12 total, 6 code, 6 markdown

### Cell 1 code

```
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from transformers import BertModel, BertPreTrainedModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
```

### Cell 2 markdown

```
## Utils for loading data and calculating score
```

### Cell 3 code

```
    train = pd.read_csv(BASE_PATH + 'train.csv')
    test = pd.read_csv(BASE_PATH + 'test.csv')
    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv(BASE_PATH + 'sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(
        sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, sample_submission
def mean_spearmanr_correlation_score(y_true, y_pred):
    score = np.nanmean([spearmanr(y_pred[:, col], y_true[:, col]).correlation
    return score
```

### Cell 5 code

```
def compute_input_arrays(df, columns, tokenizer, max_sequence_length,
        t, q, a = _trim_input(t, q, a, tokenizer, max_sequence_length, t_max_len, q_max_len, a_max_len, head_tail)
        ids, masks, segments = _convert_to_bert_inputs(stoken, tokenizer, max_sequence_length)
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    # TODO: if label is int, dtype is torch.long
    return torch.tensor(np.asarray(df[columns]), dtype=torch.float32)
def _trim_input(title, question, answer, tokenizer, max_sequence_length,
    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
def _convert_to_bert_inputs(stoken, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
```

### Cell 6 markdown

```
## Dataset and model class
```

### Cell 7 code

```
class QuestDataset(torch.utils.data.Dataset):
class CustomBertForSequenceClassification(BertPreTrainedModel):
        super(CustomBertForSequenceClassification, self).__init__(config)
        self.bert = BertModel(config)
        # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel.forward
        outputs = self.bert(
        pooled_output = torch.cat([
```

### Cell 9 code

```
class BertRunner:
    def train(self, model, criterion, optimizer, loaders, scheduler=None, logdir=None,
              num_epochs=5, score_func=None):
        valid_loader = loaders['valid']
        best_score = -1.0
            torch.cuda.empty_cache()
            avg_loss = self._train_model(model, criterion, optimizer, train_loader, scheduler)
            # evaluate on validation set
            avg_val_loss, score = self._validate_model(model, criterion, valid_loader, score_func)
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t score={:.6f} \t time={:.2f}s'.format(
                epoch + 1, num_epochs, avg_loss, avg_val_loss, score, elapsed_time))
            if score is None:
                    torch.save(best_param_loss, save_path)
                if best_score < score:
                    best_score = score
                    best_param_score = model.state_dict()
                    torch.save(best_param_score, save_path)
        model.load_state_dict(torch.load(resume))
        with torch.no_grad():
                output_valid = model(
                logits = output_valid[0]  # output preds
            preds = torch.sigmoid(torch.tensor(preds)).numpy()
    def _train_model(self, model, criterion, optimizer, train_loader, scheduler=None):
            # bert training
            # the position of this depends on the scheduler you use
            if scheduler is not None:
                scheduler.step()
    def _validate_model(self, model, criterion, valid_loader, score_func=None):
        valid_preds = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                output_valid = model(
                logits = output_valid[0]  # output preds
                # calc scor
...[truncated]
```

### Cell 11 code

```
num_folds = 5
bert_model = 'bert-base-uncased'
train, test, sample_submission = read_data(base_dataset_path)
# init Bert
tokenizer = BertTokenizer.from_pretrained(bert_model)
kf = GroupKFold(n_splits=num_folds)
fold_scores = []
for fold, (train_idx, valid_idx) in enumerate(ids):
    print("Current Fold: ", fold + 1)
    logdir = os.path.join(base_logdir, 'fold_{}'.format(fold + 1))
    train_df, val_df = train.iloc[train_idx], train.iloc[valid_idx]
    print("Train and Valid Shapes are", train_df.shape, val_df.shape)
    inputs_train = compute_input_arrays(train_df, input_cols, tokenizer, max_sequence_length=512)
    print("Preparing valid datasets....")
    inputs_valid = compute_input_arrays(val_df, input_cols, tokenizer, max_sequence_length=512)
    outputs_valid = compute_output_arrays(val_df, columns=target_cols)
    lengths_valid = np.argmax(inputs_valid[0] == 0, axis=1)
    lengths_valid[lengths_valid == 0] = inputs_valid[0].shape[1]
    valid_set = QuestDataset(inputs=inputs_valid, lengths=lengths_valid, labels=outputs_valid)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    model = CustomBertForSequenceClassification.from_pretrained(
        bert_model, num_labels=num_labels, output_hidden_states=True)
    scheduler = get_linear_schedule_with_warmup(
    runner = BertRunner(device=device)
    loaders = {'train': train_loader, 'valid': valid_loader}
    runner.train(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                 score_func=mean_spearmanr_correlation_score)
    # calc valid score
    val_preds = runner.predict_loader(model, loaders['valid'], resume=best_model_path)
    val_truth = train[target_cols].iloc[valid_idx].values
    # TODO: set your score function
    cv_score = mean_spearman
...[truncated]
```

## 1st rank 5 quality strong score 54

File: `NLP/google-quest-challenge/rank5_1st/postprocessing-for-google-quest.ipynb`
Cells: 10 total, 8 code, 2 markdown

### Cell 3 code

```
target_columns=pd.read_csv(f'../input/ensemble-data/fold_0_labels.csv').iloc[:,1:].columns.tolist()
```

### Cell 5 code

```
        # deal with every column but skip the columns that post-processing won't improve the score
def cal(arr1, arr2): # calculate column-wise scores
```

### Cell 6 code

```
for FOLD in range(5):
    labels=pd.read_csv(f'../input/ensemble-data/fold_{FOLD}_labels.csv').iloc[:,1:]
    base=pd.read_csv(f'../input/ensemble-data/bert_base_uncased_fold_{FOLD}_preds.csv').iloc[:,1:]
    wwm_uncased=pd.read_csv(f'../input/ensemble-data/wwm_uncased_fold_{FOLD}_preds.csv').iloc[:,1:]
    wwm_cased=pd.read_csv(f'../input/ensemble-data/wwm_cased_fold_{FOLD}_preds.csv').iloc[:,1:]
    large_uncased=pd.read_csv(f'../input/ensemble-data/large_uncased_fold_{FOLD}_preds.csv').iloc[:,1:]
    roberta=pd.read_csv(f'../input/ensemble-data/roberta_large_fold_{FOLD}_preds.csv').iloc[:,1:]
    ps=[base.values, wwm_uncased.values, wwm_cased.values, large_uncased.values, roberta.values]
    original_scores=cal(labels.values,mv)
    relative_scores=cal(labels.values,mv_1)-original_scores
    row = pd.DataFrame(relative_scores).T
diffs.index=[f'fold-{n}' for n in range(5)]
```

### Cell 8 code

```
# apply post-processing to the following columns will lower the scores. The numbers are the indices of the column in target_columns
scores,post_scores,post_ban_scores=[],[],[]
for FOLD in range(5):
    labels=pd.read_csv(f'../input/ensemble-data/fold_{FOLD}_labels.csv').iloc[:,1:]
    base=pd.read_csv(f'../input/ensemble-data/bert_base_uncased_fold_{FOLD}_preds.csv').iloc[:,1:]
    wwm_uncased=pd.read_csv(f'../input/ensemble-data/wwm_uncased_fold_{FOLD}_preds.csv').iloc[:,1:]
    wwm_cased=pd.read_csv(f'../input/ensemble-data/wwm_cased_fold_{FOLD}_preds.csv').iloc[:,1:]
    large_uncased=pd.read_csv(f'../input/ensemble-data/large_uncased_fold_{FOLD}_preds.csv').iloc[:,1:]
    roberta=pd.read_csv(f'../input/ensemble-data/roberta_large_fold_{FOLD}_preds.csv').iloc[:,1:]
    ps=[base.values, wwm_uncased.values, wwm_cased.values, large_uncased.values, roberta.values]
    scores.append(compute_spearmanr(labels.values,mv))
    post_scores.append(compute_spearmanr(labels.values,mv_1))
    post_ban_scores.append(compute_spearmanr(labels.values,mv_2))
```

### Cell 9 code

```
print(f"original score: {np.mean(scores)}\npost without ban: {np.mean(post_scores)}\npost with ban: {np.mean(post_ban_scores)}")
```
