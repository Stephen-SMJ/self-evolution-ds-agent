# GenAI / feedback-prize-effectiveness

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1059 quality strong score 55

File: `GenAI/feedback-prize-effectiveness/rank1059_70pct/feedback-prize-eda-bert-baseline.ipynb`
Cells: 27 total, 20 code, 7 markdown

### Cell 1 code

```
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             recall_score,
# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
# BERT
from transformers import BertTokenizer, BertModel, BertConfig
import transformers
transformers.logging.set_verbosity_error()
```

### Cell 3 code

```
submission_csv = os.path.join(input_dir, 'sample_submission.csv')
train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)
submission = pd.read_csv(submission_csv)
```

### Cell 10 code

```
    #model_cfg = BertConfig()
    tokenizer = BertTokenizer.from_pretrained('../input/huggingface-bert-variants/bert-base-uncased/bert-base-uncased')
    #model = BertModel(model_cfg)
    model = BertModel.from_pretrained('../input/huggingface-bert-variants/bert-base-uncased/bert-base-uncased', return_dict=True)
    def __init__(self, data, max_len, tokenizer, data_path):
        self.tokenizer = tokenizer
                         self.tokenizer.sep_token,
                         self.tokenizer.sep_token,
        inputs = self.tokenizer.encode_plus(text.lower(),
        targets = torch.FloatTensor(self.targets[index])
```

### Cell 12 markdown

```
## Creating Train-Validation Sets
```

### Cell 13 code

```
print(f'Validation: {val_df.shape} \n')
```

### Cell 14 code

```
                                     tokenizer = config.tokenizer,
valid_dataset = FeedbackPrizeDataset(val_df,
                                     tokenizer = config.tokenizer,
val_data_loader = DataLoader(valid_dataset,
```

### Cell 15 markdown

```
## Building Model
```

### Cell 16 code

```
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```

### Cell 17 code

```
class FeedbackPrizeModel(torch.nn.Module):
        self.bert_model = config.model
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, len(target_list))
        output = self.bert_model(input_ids,
```

### Cell 18 code

```
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()),
```

### Cell 19 markdown

```
## Train Model
```

### Cell 20 code

```
            input_ids = batch['input_ids'].to(device, dtype = torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
            targets = batch['targets'].to(device, dtype = torch.float)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f' Epoch: {epoch + 1} - Validation Set '.center(50, '='))
        with torch.no_grad():
                input_ids = data['input_ids'].to(device, dtype = torch.long)
                attention_mask = data['attention_mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f} \n'.format(
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

## 40pct rank 628 quality strong score 55

File: `GenAI/feedback-prize-effectiveness/rank628_40pct/train-detectai-deberta.ipynb`
Cells: 27 total, 26 code, 1 markdown

### Cell 1 code

```
import transformers
```

### Cell 2 code

```
model_checkpoint = "/kaggle/input/deberta-v3-large" #base model
```

### Cell 3 code

```
df_ai = pd.read_csv('/kaggle/input/daigt-external-dataset/daigt_external_dataset.csv')
```

### Cell 4 code

```
df_ai = pd.read_csv('/kaggle/input/daigt-external-dataset/daigt_external_dataset.csv')
df_valid = df_ai[-200:]
v1['text'] = df_valid.source_text
v2['text'] = df_valid.text
valid = pd.concat([v1,v2]).sample(frac=1).reset_index(drop=True)
```

### Cell 5 code

```
llm7prompt = pd.read_csv("/kaggle/input/llm-7-prompt-training-dataset/train_essays_RDizzl3_seven_v1.csv")
```

### Cell 6 code

```
feedback_1 = pd.read_csv("/kaggle/input/feedback21-and-fb-effectiveness/feedback-prize-2021.csv")
feedback_2 = pd.read_csv("/kaggle/input/feedback21-and-fb-effectiveness/feedback-prize-effectiveness.csv")
```

### Cell 7 code

```
h2o = pd.read_csv("/kaggle/input/h2oai-predict-the-llm/train.csv")[np.invert(pd.read_csv("/kaggle/input/h2oai-predict-the-llm/train.csv").Response.isnull())].rename(columns={"Response":"text"})[["text"]]
```

### Cell 9 code

```
train2 = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=',')
```

### Cell 10 code

```
                   valid],
                   valid],
```

### Cell 11 code

```
palm = pd.read_csv("/kaggle/input/llm-generated-essay-using-palm-from-google-gen-ai/LLM_generated_essay_PaLM.csv")[["text"]]
mistral = pd.read_csv("/kaggle/input/llm-mistral-7b-instruct-texts/Mistral7B_CME_v7_15_percent_corruption.csv")[["text"]]
llama = pd.read_csv("/kaggle/input/daigt-data-llama-70b-and-falcon180b/llama_falcon_v3.csv")[["text"]]
claude = pd.read_csv("/kaggle/input/hello-claude-1000-essays-from-anthropic/persuade15_claude_instant1.csv")[["essay_text"]]
```

### Cell 13 code

```
from sklearn.model_selection import StratifiedKFold
```

### Cell 14 code

```
sk = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
```

## 20pct rank 339 quality strong score 67

File: `GenAI/feedback-prize-effectiveness/rank339_20pct/copetition1.ipynb`
Cells: 35 total, 35 code, 0 markdown

### Cell 0 code

```
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from transformers import TFBertModel
import transformers
```

### Cell 1 code

```
df = pd.read_csv("../input/feedback-prize-effectiveness/train.csv")
```

### Cell 3 code

```
eff = pd.read_csv('../input/compet00/ef.csv')
ineff = pd.read_csv('../input/compet00/inef.csv')
adq = pd.read_csv('../input/compet00/ad (1).csv')
```

### Cell 17 code

```
def bert_encode(texts, tokenizer, max_len=MAX_LEN):
        token = tokenizer(text, max_length=max_len, truncation=True, padding='max_length',
```

### Cell 18 code

```
# First load the real tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained('../input/huggingface-bert-variants/bert-base-cased/bert-base-cased')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
```

### Cell 20 code

```
sep = tokenizer.sep_token
```

### Cell 22 code

```
X_train, X_valid, y_train, y_valid = train_test_split(df['inputs'], df['label'], test_size=0.01, random_state=42)
```

### Cell 23 code

```
X_train = bert_encode(X_train.astype(str), tokenizer)
X_valid = bert_encode(X_valid.astype(str), tokenizer)
y_valid = y_valid.values
```

### Cell 24 code

```
valid_dataset = (
    .from_tensor_slices((X_valid, y_valid))
```

### Cell 25 code

```
def build_model(bert_model, max_len=MAX_LEN):
    sequence_output = bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
    model.compile(Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### Cell 26 code

```
transformer_layer = (TFBertModel.from_pretrained('../input/huggingface-bert-variants/bert-base-cased/bert-base-cased'))
```

### Cell 27 code

```
    validation_data=valid_dataset,
```

## 10pct rank 174 quality usable score 41

File: `GenAI/feedback-prize-effectiveness/rank174_10pct/all-essays-for-3-feedback-competitions.ipynb`
Cells: 24 total, 17 code, 7 markdown

### Cell 4 code

```
train_1 = pd.read_csv('../input/feedback-prize-2021/train.csv')
train_2 = pd.read_csv('../input/feedback-prize-effectiveness/train.csv')
train_3 = pd.read_csv('../input/feedback-prize-english-language-learning/train.csv')
```

### Cell 22 code

```
all_data.to_csv('./feedbackprize_full_essays.csv', index=False)
```

## 1st rank 1 quality reject score -9

File: `GenAI/feedback-prize-effectiveness/rank1_1st/exp08-del-dataset-part-1.ipynb`
Cells: 4 total, 4 code, 0 markdown

### Cell 1 code

```
model = "exp-08-del-8folds-kd-part-1"
shutil.copytree('../input/exp-08-deberta-large-kd-part-1', f'./{model}/')
```

### Cell 2 code

```
# model = "a-prod-fpe-dexl-8-folds"
# shutil.copytree("../input/a-prod-fpe-dexl-8-folds", f'./{model}/')
```
