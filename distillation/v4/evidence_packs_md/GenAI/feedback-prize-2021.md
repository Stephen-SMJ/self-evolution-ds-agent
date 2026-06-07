# GenAI / feedback-prize-2021

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1408 quality strong score 67

File: `GenAI/feedback-prize-2021/rank1408_70pct/all-about-bert-you-need-to-know.ipynb`
Cells: 111 total, 33 code, 78 markdown

### Cell 1 markdown

```
In this tutorial I'll show you how to use BERT with the huggingface PyTorch library to quickly and efficiently fine-tune a model to get near state of the art performance in sentence classification. More broadly, I describe the practical application of transfer learning in NLP to create high performance models with minimal effort on a range of NLP tasks.
```

### Cell 2 markdown

```
- [1. What is BERT?](#1)
- [3. A Shift in NLP](#3)
    - [6.1.BERT Tokenizer ](#5.1)
    - [6.8.Training & Validation Split](#5.8)
    - [6.9.Converting to PyTorch Data Types](#5.9)
    - [7.1. BertForSequenceClassification](#6.1)
    - [7.2. Optimizer & Learning Rate Scheduler](#6.2)
```

### Cell 3 markdown

```
# **<center><span style="color:#00BFC4;">What is BERT? </span></center>**
BERT (Bidirectional Encoder Representations from Transformers), released in late 2018, is the model we will use in this tutorial to provide readers with a better understanding of and practical guidance for using transfer learning models in NLP. BERT is a method of pretraining language representations that was used to create models that NLP practicioners can then download and use for free. You can either use these models to extract high quality language features from your text data, or you can fine-tune these models on a specific task (classification, entity recognition, question answering, etc.) with your own data to produce state of the art predictions.
This post will explain how you can modify and fine-tune BERT to create a powerful NLP model that quickly gives you state of the art results.
```

### Cell 5 markdown

```
In this tutorial, we will use BERT to train a text classifier. Specifically, we will take the pre-trained BERT model, add an untrained layer of neurons on the end, and train the new model for our classification task. Why do this rather than train a train a specific deep learning model (a CNN, BiLSTM, etc.) that is well suited for the specific NLP task you need?
    * First, the pre-trained BERT model weights already encode a lot of information about our language. As a result, it takes much less time to train our fine-tuned model - it is as if we have already trained the bottom layers of our network extensively and only need to gently tune them while using their output as features for our classification task. In fact, the authors recommend only 2-4 epochs of training for fine-tuning BERT on a specific NLP task (compared to the hundreds of GPU hours needed to train the original BERT model or a LSTM from scratch!).
    * In addition and perhaps just as important, because of the pre-trained weights this method allows us to fine-tune our task on a much smaller dataset than would be required in a model that is built from scratch. A major drawback of NLP models built from scratch is that we often need a prohibitively large dataset in order to train our network to reasonable accuracy, meaning a lot of time and energy had to be put into dataset creation. By fine-tuning BERT, we are now able to get away with training a model to good performance on a much smaller amount of training data.
    * Finally, this simple fine-tuning procedure (typically adding one fully-connected layer on top of BERT and training for a few epochs) was shown to achieve state of the art results with minimal task-specific adjustments for a wide variety of tasks: classification, language inference, semantic 
...[truncated]
```

### Cell 6 markdown

```
# **<center><span style="color:#00BFC4;"> A Shift in NLP </span></center>**
This shift to transfer learning parallels the same shift that took place in computer vision a few years ago. Creating a good deep learning network for computer vision tasks can take millions of parameters and be very expensive to train. Researchers discovered that deep networks learn hierarchical feature representations (simple features like edges at the lowest layers with gradually more complex features at higher layers). Rather than training a new network from scratch each time, the lower layers of a trained network with generalized image features could be copied and transfered for use in another network with a different task. It soon became common practice to download a pre-trained deep network and quickly retrain it for the new task or add additional layers on top - vastly preferable to the expensive process of training a network from scratch. For many, the introduction of deep pre-trained language models in 2018 (ELMO, BERT, ULMFIT, Open-GPT, etc.) signals the same shift to transfer learning in NLP that computer vision saw.
```

### Cell 7 code

```
import tensorflow as tf
```

### Cell 9 code

```
import torch
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    device = torch.device("cpu")
```

### Cell 11 markdown

```
Next, let's install the [transformers](https://github.com/huggingface/transformers) package from Hugging Face which will give us a pytorch interface for working with BERT. (This library contains interfaces for other pretrained language models like OpenAI's GPT and GPT-2.) We've selected the pytorch interface because it strikes a nice balance between the high-level APIs (which are easy to use but don't provide insight into how things work) and tensorflow code (which contains lots of details but often sidetracks us into lessons about tensorflow, when the purpose here is BERT!).
At the moment, the Hugging Face library seems to be the most widely accepted and powerful pytorch interface for working with BERT. In addition to supporting a variety of different pre-trained transformer models, the library also includes pre-built modifications of these models suited to your specific task. For example, in this tutorial we will use `BertForSequenceClassification`.
The library also includes task-specific classes for token classification, question answering, next sentence prediciton, etc. Using these pre-built classes simplifies the process of modifying BERT for your purposes.
```

### Cell 12 code

```
# !pip install transformers
```

### Cell 13 markdown

```
The code in this notebook is actually a simplified version of the [run_glue.py](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py) example script from huggingface.
`run_glue.py` is a helpful utility which allows you to pick which GLUE benchmark task you want to run on, and which pre-trained model you want to use (you can see the list of possible models [here](https://github.com/huggingface/transformers/blob/e6cff60b4cbc1158fbd6e4a1c3afda8dc224f566/examples/run_glue.py#L69)). It also supports using either the CPU, a single GPU, or multiple GPUs. It even supports using 16-bit precision if you want further speed up.
```

### Cell 15 markdown

```
We'll use [The Corpus of Linguistic Acceptability (CoLA)](https://nyu-mll.github.io/CoLA/) dataset for single sentence classification. It's a set of sentences labeled as grammatically correct or incorrect. It was first published in May of 2018, and is one of the tests included in the "GLUE Benchmark" on which models like BERT are competing.
```

### Cell 24 markdown

```
We can't use the pre-tokenized version because, in order to apply the pre-trained BERT, we *must* use the tokenizer provided by the model. This is because (1) the model has a specific, fixed vocabulary and (2) the BERT tokenizer has a particular way of handling out-of-vocabulary words.
```

## 40pct rank 823 quality usable score 43

File: `GenAI/feedback-prize-2021/rank823_40pct/corrected-train-csv-feedback-prize.ipynb`
Cells: 29 total, 15 code, 14 markdown

### Cell 0 markdown

```
### This is a process that standardizes the examples in the training set. The main issue I see is `discourse_start` and `discourse_end` being slightly off.  By standardizing and cleaning the examples, we can hopefully create consistent training data that causes an improvement in model performance.

#### If `text` is the text from the file, then it would be expected that `file_text[discourse_start:discourse_end]=discourse_text` but in 44k rows, this is not the case.

#### Here is an example where `file_text[discourse_start:discourse_end]!=discourse_text`
```
discourse_id = 1622992280991.0
discourse_text = "First, cell phones are a benefit and allows everyone to have access to a telephone at all times of the day.\n"
file_text[discourse_start:discourse_end]="rst, cell phones are a benefit and allows everyone to have access to a telephone at all times of the day.

K"
```
After it goes through my script:
```
'First, cell phones are a benefit and allows everyone to have access to a telephone at all times of the day.\n'
```

#### Here is an example straight from the csv where there is a bit at the beginning that looks unecessary:
```
discourse_id = 1622489430075.0
discourse_text = '. Drivers should not be able to use cell phones in any capacity while operating a motor vehicle. '
file_text[discourse_start:discourse_end] = '. Drivers should not be able to use cell phones in any capacity while operating a motor vehicle.\n'
```
After it goes through my script:
```
'Drivers should not be able to use cell phones in any capacity while operating a motor vehicle.\n'
```

# Please look at the following discussions for more information about this topic!
- [Mystery Solved - Discrepancy Between PredictionString and DiscourseText](https://www.kaggle.com/c/feedback-prize-2021/discussion/2975
...[truncated]
```

### Cell 2 code

```
df = pd.read_csv("../input/feedback-prize-2021/train.csv")
```

### Cell 8 markdown

```
## My approach to fix the incorrect `discourse_start` and `discourse_end` values

The first step is to check if we can use the entire `discourse_text` to find the starting index in the file text.


If a match with the entire `discourse_text` string is not found, I'll take the first ~20 or so characters from `discourse_text` and see where it starts in the file text and use that as the starting point.

If the span starts with punctuation and then text, I'll keep increasing the start index until it isn't whitespace or punctuation. This eliminates the examples that begin with a period or comma.

If the span ends in whitespace, I'll keep it. This whitespace could be beneficial for the model.

If the span does not end in whitespace and the character after the span is punctuation or whitespace, extend the span to include it. Extending it can hopefully add useful information and it also standardizes the examples to have trailing whitespace but not leading whitespace. Adding the punctuation or whitespace would ***not*** change the `predictionstring` but it ***would*** change how the NER labeling is done.
```

### Cell 27 code

```
df.to_csv("corrected_train.csv", index=False)
```

### Cell 28 markdown

```
### Hopefully this makes our models better! I know it can be confusing, so please comment and I'll do my best to answer your quesions!

<p style="font-size: 40px">😊</p>
```

## 20pct rank 535 quality strong score 55

File: `GenAI/feedback-prize-2021/rank535_20pct/q-a-inference.ipynb`
Cells: 29 total, 27 code, 2 markdown

### Cell 0 code

```
from transformers import *
```

### Cell 2 code

```
# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
```

### Cell 4 code

```
             'Counterclaim', 'Rebuttal']
```

### Cell 8 code

```
#model_ckpt = "distilbert-base-cased-distilled-squad"
model_checkpoint = "../input/q-a-pytorch/model.h5"
# model_checkpoint = "distilbert-base-cased-distilled-squad"
config_model = "../input/q-a-pytorch/model.h5/config.json"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
             'Counterclaim':5, 'Rebuttal':6}
```

### Cell 9 code

```
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
# tokenizer.save_pretrained('model')
```

### Cell 10 code

```
max_length = tokenizer.model_max_length
```

### Cell 11 code

```
def preprocess_validation_examples(examples):
    inputs = tokenizer(
```

### Cell 14 code

```
validation_dataset = dataset.map(
    preprocess_validation_examples,
len(validation_dataset)
```

### Cell 15 markdown

```
# Build Model
We will use LongFormer backbone and add our own NER head using one hidden layer of size 256 and one final layer with softmax. We use 15 classes because we have a `B` class and `I` class for each of 7 labels. And we have an additional class (called `O` class) for tokens that do not belong to one of the 14 classes.
```

### Cell 16 code

```
import torch
from transformers import AutoModelForQuestionAnswering
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
```

### Cell 19 code

```
validation_dataset
```

### Cell 20 code

```
from torch.utils.data import DataLoader
from transformers import default_data_collator
dataloader = DataLoader(validation_dataset, collate_fn = default_data_collator, batch_size = 8)
```

## 10pct rank 202 quality strong score 73

File: `GenAI/feedback-prize-2021/rank202_10pct/feedback-3-eda-sentence-transformers.ipynb`
Cells: 45 total, 29 code, 16 markdown

### Cell 3 code

```
sys.path.append("../input/sentencetransformers/sentence-transformers-2.2.2/sentence-transformers-2.2.2")
import sentence_transformers
```

### Cell 4 code

```
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
```

### Cell 5 code

```
# !pip install ../input/spacy306installwhlfiles/spacy_transformers-1.0.2-py2.py3-none-any.whl
```

### Cell 6 markdown

```
This time we find that we have a column of IDs, a column with the student texts - written by students who are learning English as a second language. To the right you will see 6 columns with scores in what seems to be a 1 to 5 scale at first sight.
```

### Cell 7 code

```
df = pd.read_csv('../input/feedback-prize-english-language-learning/train.csv')
```

### Cell 9 markdown

```
Let's skip the texts for now and let's instead try to understand what are the target measures we'll be identifying. Let's do them one by one starting by the distributions of the scores:
```

### Cell 10 code

```
print("\n All the values are scored from 1 to 5 in 0.5 increments. Let's double check the histograms as they stack next to each other\n\n")
```

### Cell 13 markdown

```
Another question to ask is whether there are just **bad** texts or if it's possible to score excellent in one target measure and then poorly in another one. When we look at the differences inside a row, we find that most of the texts have a difference of 1 point between its best and its worst scores. There are almost never differences in a row higher than 1.5 - only 12 texts have a difference of 2 between their best and worst scores.
```

### Cell 18 markdown

```
## SYNTAX:
# From wikipedia:
In linguistics, syntax (/ˈsɪntæks/) is the study of how words and morphemes combine to form larger units such as phrases and sentences. Central concerns of syntax include word order, grammatical relations, hierarchical sentence structure (constituency), agreement, the nature of crosslinguistic variation, and the relationship between form and meaning (semantics). There are numerous approaches to syntax that differ in their central assumptions and goals.

# Analysis:

Let's choose a bad syntax example text. You will see that this is characterized by the non-existence of punctuation. We can force this into the dependency parser server and see the mess it creates. It seems like the dependency parser goes from to and all directions. Don't forget to uncomment the line in the middle to be able to see it. Scroll to the right so you can see the full sentence analysis.
DONT FORGET TO CLICK STOP TO THE LEFT TO STOP THE SERVER
```

### Cell 20 markdown

```
It seems like this score is all about using words that exist and that are used correctly. Made up words or not using the words properly will get the student a bad score. Other than transformers I would probably just check that the words exist for this part of the competition
```

### Cell 22 markdown

```
I think you would score high in this metric if you are able to use some idioms correctly. Note in this example the use of multi-word expressions such as "in contrast" or "as a result"
```

### Cell 30 code

```
model = SentenceTransformer("../input/sentencetransformers/multi-qa-MiniLM-L6-cos-v1/multi-qa-MiniLM-L6-cos-v1")
```

## 1st rank 1 quality strong score 58

File: `GenAI/feedback-prize-2021/rank1_1st/feedback-lgb-train.ipynb`
Cells: 5 total, 2 code, 3 markdown

### Cell 0 markdown

```
##### There are no lstm and pca features in this training code, because these two features are too cumbersome and only improve by 0.001
##### Also, for memory reasons, you need to copy to your own machine for training
```

### Cell 2 code

```
df = pickle.load(open('./data/data_6model_offline712_online704_ensemble.pkl','rb'))
train_df = pd.read_csv('./data/train.csv')
             5:'Counterclaim', 6:'Rebuttal', 7:'blank'}
    "Rebuttal": 0.02,
valid_pred = pd.DataFrame(all_predictions, columns=['id', 'class', 'pos'])
for cache in tqdm(valid_pred.values):
valid_pred['predictionstring'] = predictionstring
    "Rebuttal": 0.6,
    flag_list = []
        flag = 0
                flag = min(left_thre, right_thre)
        flag_list.append(flag)
    df_new['flag'] = flag_list
    df_new = pd.concat([df_new, df.loc[df_new[(df_new.flag>=thre) & (df_new.flag<0.95)].index]])
    df_new['flag'].fillna(0,inplace=True)
valid_pred = deal_predictionstring(valid_pred)
valid_oof = train_df.copy()
tmp = valid_oof.predictionstring.map(lambda x:x.split())
valid_oof['predictionstring'] = tmp1
    valid_oof[["id", "discourse_type", "predictionstring"]]
pred_df = valid_pred[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
joined = pred_df.merge(
valid_pred['label'] = 0
valid_true_id = joined[joined.potential_TP==True]['pred_id']
valid_pred.loc[valid_true_id, 'label'] = 1
overlap = overlap.groupby('pred_id')['min_overlap'].max().reset_index()
valid_pred = valid_pred.merge(overlap, left_index=True, right_on='pred_id', how='left')
valid_pred = valid_pred.drop('pred_id',axis=1)
pickle.dump(valid_pred, open('./data/recall_data.pkl','wb+'))
```

### Cell 4 code

```
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
train_df = pd.read_csv(data_path+'train.csv')
             5:'Counterclaim', 6:'Rebuttal', 7:'blank'}
df_wf = pickle.load(open('./data/data_6model_offline712_online704_ensemble.pkl','rb'))
preds1_5fold = {}
preds2_5fold = {}
    preds1_5fold[row.id] = t1
    preds2_5fold[row.id] = t2
valid_pred = pickle.load(open('./data/recall_data.pkl','rb'))
kfold_ids = pickle.load(open('./data/kfold_ids.pkl','rb'))
logging.info(f'valid_pred num:{len(valid_pred)}')
preds2_5fold_type = {}
for k,t in preds2_5fold.items():
    preds2_5fold_type[k] = np.array(t).argmax(axis=-1)
        dic['post_flag'] = cache[4]
        preds1_all = np.array(preds1_5fold[id])
        preds2_all = np.array(preds2_5fold[id])[:,label2id[typ]]
        preds4_all = np.array(preds2_5fold[id])[:,other_type].max(axis=-1)
        sort_idx = preds1[1:].argsort()[::-1]
        sort_idx = preds2.argsort()
                dic[f'before_other_type{i}'] = preds2_5fold_type[id][start-i]
                dic[f'after_other_type{i}'] = preds2_5fold_type[id][end+i]
data_splits = np.array_split(valid_pred.values, num_jobs)
          'metric': {'auc'},
#           'metric': {'l2'},
valid_pred = df_feat[['id', 'class','word_start','word_end']].copy()
valid_pred['class'] = valid_pred['class'].map(lambda x:id2label[x])
valid_pred['lgb_prob'] = -1
for fold in range(5):
    df_feat_train = df_feat[df_feat.id.isin(kfold_ids[fold][0])].copy()
    df_feat_val = df_feat[df_feat.id.isin(kfold_ids[fold][1])].copy()
                    valid_sets=[lgb_train, lgb_val],
                    early_stopping_rounds=100)
    valid_pred.loc[df_feat_val.index, 'lgb_prob'] = lgb_preds
    pickle.dump(clf, open(f'./result/lgb_fold{fold}.pkl','
...[truncated]
```
