# NLP / commonlitreadabilityprize

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 2492 quality strong score 67

File: `NLP/commonlitreadabilityprize/rank2492_70pct/nn-notebook-lstm.ipynb`
Cells: 29 total, 23 code, 6 markdown

### Cell 2 code

```
train = pd.read_csv(input_path+"train.csv", usecols = ["excerpt","target"])
test = pd.read_csv(input_path+"test.csv", usecols=["excerpt"])
sub = pd.read_csv(input_path+"sample_submission.csv")
```

### Cell 5 code

```
    return [token.text for token in tok.tokenizer(nopunct)]
```

### Cell 11 code

```
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
```

### Cell 13 code

```
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx] #, self.X[idx][1]
```

### Cell 15 code

```
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
val_dataset = CommonLitReadabiltyDataset(X_valid, y_valid)
```

### Cell 19 code

```
        inputs=inputs.to('cuda').type(torch.long)
        labels=labels.to('cuda').type(torch.float) #передаем батч на GPU(cuda)
```

### Cell 20 code

```
def valid_epoch(model,criterion,optimizer,dataset,epoch):
    print(f"Epoch#{epoch}. Validation")
    with torch.no_grad():
            inputs=inputs.to('cuda').type(torch.long)
            labels=labels.to('cuda').type(torch.float) #передаем батч на GPU(cuda)
    print(f"Epoch#{epoch} (Validation) completed. ")
```

### Cell 24 code

```
    model_ft, val_loss = valid_epoch(model_ft,criterion,optimizer,val_dataset,epoch)
```

### Cell 26 code

```
X_test = torch.LongTensor(X_test).to('cuda')
```

### Cell 28 code

```
sub.to_csv("submission_lstm.csv", index=False)
```

## 40pct rank 1483 quality strong score 53

File: `NLP/commonlitreadabilityprize/rank1483_40pct/commonlit-baseline-model.ipynb`
Cells: 13 total, 10 code, 3 markdown

### Cell 0 markdown

```
This notebook is an example of a quick and (relatively) simple regression model for the CommonLit dataset.

```

### Cell 1 code

```
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
```

### Cell 2 code

```
df = pd.read_csv(train_file, index_col='id')
```

### Cell 7 markdown

```
Split the data into subsets for training and validation to see how well the model will generalize (i.e. how well will it be able to make new predictions).
```

### Cell 9 markdown

```
Fit the random forest model.
```

### Cell 10 code

```
rf = RandomForestRegressor(n_estimators=100, random_state=1)
```

### Cell 11 code

```
valid_mse = mean_squared_error(y_val_predicted, y_val_actual)
print(f"baseline model validation MSE = {valid_mse:.6f}")
```

### Cell 12 code

```
ax2.set_title("Validation data")
```

## 20pct rank 645 quality usable score 29

File: `NLP/commonlitreadabilityprize/rank645_20pct/roberta-base-pretrained.ipynb`
Cells: 6 total, 6 code, 0 markdown

### Cell 0 code

```
from transformers import (AutoModel,AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
```

### Cell 1 code

```
train_data = pd.read_csv('../input/commonlitreadabilityprize/train.csv')
test_data = pd.read_csv('../input/commonlitreadabilityprize/test.csv')
```

### Cell 2 code

```
model_name = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('./roberta-base-pretrained');
```

### Cell 3 code

```
    tokenizer=tokenizer,
valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    tokenizer=tokenizer, mlm=True, mlm_probability=0.3)
    metric_for_best_model='eval_loss',
    eval_dataset=valid_dataset)
```

### Cell 4 code

```
trainer.save_model(f'./roberta-base-pretrained')
```

## 10pct rank 217 quality strong score 73

File: `NLP/commonlitreadabilityprize/rank217_10pct/pre-trained-roberta-solution-in-pytorch.ipynb`
Cells: 19 total, 15 code, 4 markdown

### Cell 0 markdown

```
This is kernel is almost the same as [Lightweight Roberta solution in PyTorch](https://www.kaggle.com/andretugan/lightweight-roberta-solution-in-pytorch), but instead of "roberta-base", it starts from [Maunish's pre-trained model](https://www.kaggle.com/maunish/clrp-roberta-base).
Acknowledgments: some ideas were taken from kernels by [Torch](https://www.kaggle.com/rhtsingh) and [Maunish](https://www.kaggle.com/maunish).
```

### Cell 1 code

```
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold
```

### Cell 2 code

```
NUM_FOLDS = 3
ROBERTA_PATH = "../input/clrp-roberta-base/clrp_roberta_base"
TOKENIZER_PATH = "../input/clrp-roberta-base/clrp_roberta_base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Cell 3 code

```
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
```

### Cell 4 code

```
train_df = pd.read_csv("/kaggle/input/commonlitreadabilityprize/train.csv")
test_df = pd.read_csv("/kaggle/input/commonlitreadabilityprize/test.csv")
submission_df = pd.read_csv("/kaggle/input/commonlitreadabilityprize/sample_submission.csv")
```

### Cell 5 code

```
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
```

### Cell 7 code

```
            self.target = torch.tensor(df.target.values, dtype=torch.float32)
        self.encoded = tokenizer.batch_encode_plus(
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        attention_mask = torch.tensor(self.encoded['attention_mask'][index])
```

### Cell 8 markdown

```
The model is inspired by the one from [Maunish](https://www.kaggle.com/maunish/clrp-roberta-svm).
```

### Cell 9 code

```
        config = AutoConfig.from_pretrained(ROBERTA_PATH)
        self.roberta = AutoModel.from_pretrained(ROBERTA_PATH, config=config)
        roberta_output = self.roberta(input_ids=input_ids,
        # 1 for the embedding layer, and 12 for the 12 Roberta layers.
        # We take the hidden states from the last Roberta layer.
        last_layer_hidden_states = roberta_output.hidden_states[-1]
        # The size of the hidden state of each cell is 768 (for roberta-base).
        # we compute a weighted average of the hidden states of all cells.
        # Now we compute context_vector as the weighted average.
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)
        # Now we reduce the context vector to the prediction score.
```

### Cell 10 code

```
    with torch.no_grad():
```

### Cell 11 code

```
    with torch.no_grad():
```

### Cell 12 code

```
          optimizer, scheduler=None, num_epochs=NUM_EPOCHS):
            if scheduler:
                scheduler.step()
                    torch.save(model.state_dict(), model_path)
```

## 1st rank 12 quality strong score 54

File: `NLP/commonlitreadabilityprize/rank12_1st/fork-of-commonlitreadability-r3-2a94de.ipynb`
Cells: 51 total, 51 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
```

### Cell 1 code

```
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
```

### Cell 2 code

```
if torch.cuda.is_available():
    device = torch.device("cuda")
    device = torch.device("cpu")
```

### Cell 5 code

```
from torch import nn
def predict_fast(model_name=None, data=None, init_model=None, tokenizer=None, num_labels=1, is_multilabel=False, output_logits=False, use_softmax=False):
  tokenizer = AutoTokenizer.from_pretrained(model_name) if model_name else tokenizer
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
```

### Cell 6 code

```
def postprocess_predictions(predictions, bin_predictions, bin_averages, threshold=0.58):
    if abs(p - bin_averages[bin_predictions[idx][0]]) > 0.5 and np.argmax(bin_predictions[idx][1]) > threshold:
```

### Cell 7 code

```
def perform_bin_postprocessing(predictions, bin_dirs, averages, data):
    final_preds = np.mean(np.vstack(bin_preds), axis=0)
    new_preds = postprocess_predictions(predictions, zipped, averages)
```

### Cell 8 code

```
def make_ridge_predictions(df, ridge_dirs, model_dirs, model_bin_dirs):
    Y = np.column_stack([logits_arr, preds_arr])
    clf = load(ridge_dirs[idx])
  preds = np.vstack(predictions)
```

### Cell 9 code

```
def make_ensembler_predictions(fold_predictions, ensembler_dirs, return_mean=True):
  for idx, predictions in enumerate(fold_predictions):
    clf = load(ensembler_dirs[idx])
    Y = np.column_stack(predictions)
    preds = np.vstack(final_predictions)
```

### Cell 10 code

```
  preds = np.vstack(preds)
```

### Cell 11 code

```
test_df = pd.read_csv('../input/commonlitreadabilityprize/test.csv')
```

### Cell 13 code

```
#predictions = predict_fast('../input/robertabase051/roberta-clrp-target-v2/best', test_values)
```

### Cell 14 code

```
ridge_dirs = [
    '../input/largeridges/ridge-robertas-large/model_fold_0/ridge_model.joblib',
    '../input/largeridges/ridge-robertas-large/model_fold_1/ridge_model.joblib',
    '../input/largeridges/ridge-robertas-large/model_fold_2/ridge_model.joblib',
    '../input/largeridges/ridge-robertas-large/model_fold_3/ridge_model.joblib',
    '../input/largeridges/ridge-robertas-large/model_fold_4/ridge_model.joblib',
    '../input/largeridges/ridge-robertas-large/model_fold_5/ridge_model.joblib'
    '../input/rolargef0/model_fold_0/best',
    '../input/roblargeaugf1/model_fold_1/best',
    '../input/rolargef2/model_fold_2/best',
    '../input/roblargeaugf4/model_fold_4/best',
    '../input/roblargef5/model_fold_5/best',
    '../input/roblargeaugf3/model_fold_3/best'
    '../input/binsf0/model_fold_0/best',
    '../input/binsf1/model_fold_1/best',
    '../input/binsf2/model_fold_2/best',
    '../input/binsf3/model_fold_3/best',
    '../input/binsf4/model_fold_4/best',
    '../input/binsf5/model_fold_5/best',
```
