# NLP / nlp-getting-started

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 557 quality weak score 21

File: `NLP/nlp-getting-started/rank557_70pct/notebook76d50be751.ipynb`
Cells: 2 total, 2 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
```

### Cell 1 code

```
from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
# 1. Load the data (Kaggle automatically stores it in the input folder)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_df.to_csv('submission.csv', index=False)
```

## 40pct rank 331 quality strong score 73

File: `NLP/nlp-getting-started/rank331_40pct/disaster-tweets-tf-idf-vs-bilstm-vs-bert.ipynb`
Cells: 26 total, 14 code, 12 markdown

### Cell 0 markdown

```
            <span style="font-weight: 300; font-size: 24px; color: #a8a29e;">TF-IDF vs BiLSTM vs BERT</span>
            Three generations of NLP compared on the same dataset. TF-IDF with keyword n-grams as a baseline, BiLSTM with GloVe embeddings and attention pooling for sequence awareness, and fine-tuned BERT for contextual understanding.
            <p style="color: #a8a29e; font-size: 11px; margin: 4px 0 0; text-transform: uppercase; letter-spacing: 1px;">Eval Metric</p>
```

### Cell 3 code

```
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
!pip install -q transformers==4.44.2 2>/dev/null
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
logging.getLogger('tensorflow').setLevel(logging.ERROR)
train_df = pd.read_csv('/kaggle/input/competitions/nlp-getting-started/train.csv')
test_df  = pd.read_csv('/kaggle/input/competitions/nlp-getting-started/test.csv')
sub_df   = pd.read_csv('/kaggle/input/competitions/nlp-getting-started/sample_submission.csv')
```

### Cell 5 code

```
dup_conflicting = dup_texts.groupby('text')['target'].nunique()
```

### Cell 6 code

```
                 .groupby('keyword')['target']
                 .agg(['mean', 'count'])
```

### Cell 7 code

```
def word_freq(texts): # score each word by how much it leans toward one class vs the other
scores = {} # +1 means only in disaster, -1 means only in non-disaster
        scores[w] = (dr - sr) / (dr + sr + 1e-9)
ranked = sorted(scores.items(), key=lambda x: x[1])
top_d = ranked[-15:][::-1]
ax1.set_xlabel('Relative frequency score')
top_s = ranked[:15]
ax2.set_xlabel('Relative frequency score')
ax4 = fig.add_subplot(gs[1, 1]) # words with high frequency in both classes but near-zero score — these fail TF-IDF
ambiguous_words = sorted(scores.items(), key=lambda x: abs(x[1]))[:15]
sorted_idx = np.argsort(amb_freq)[::-1] # sort by total frequency so the most common ambiguous words are on top
for i, (freq, score) in enumerate(zip(amb_freq, amb_vals)):
             f'n={freq}  score={score:+.3f}', va='center', fontsize=8, color='#444')
ax4.spines['right'].set_visible(False) # "like", "people", "new" etc. carry zero signal, this is where BERT should help
```

### Cell 10 code

```
    """Minimal cleaning for BERT."""
        text = f"{keyword} : {text}" # prepend keyword, BERT's attention can learn to use it as context
train_df['text_bert'] = train_df.apply(
test_df['text_bert'] = test_df.apply(
    print(f"  bert  : {row['text_bert'][:90]}")
X_bert = train_df['text_bert'].values
splits = train_test_split(X_text, X_bert, X_text_kw, y, test_size=0.15, random_state=42, stratify=y) # split all text arrays together so indices stay aligned
X_train_bert, X_val_bert = splits[2], splits[3]
```

### Cell 12 code

```
tfidf = TfidfVectorizer(
X_train_tfidf = tfidf.fit_transform(X_train_kw)
X_val_tfidf   = tfidf.transform(X_val_kw)
X_test_tfidf  = tfidf.transform(test_df['text_kw'].values)
print(f"  vocabulary: {len(tfidf.vocabulary_):,} terms")
print(f"  matrix: {X_train_tfidf.shape}")
models_tfidf = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'Linear SVM': CalibratedClassifierCV(
tfidf_results = {}
for name, model in models_tfidf.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_val_tfidf)
    probs = model.predict_proba(X_val_tfidf)[:, 1]
        'f1': f1_score(y_val, preds),
        'accuracy': accuracy_score(y_val, preds),
        'precision': precision_score(y_val, preds),
        'recall': recall_score(y_val, preds),
    tfidf_results[name] = res
# ensemble
ensemble = VotingClassifier(
    estimators=[(k.lower().replace(' ', '_'), v) for k, v in models_tfidf.items()],
ensemble.fit(X_train_tfidf, y_train)
ens_preds = ensemble.predict(X_val_tfidf)
ens_probs = ensemble.predict_proba(X_val_tfidf)[:, 1]
tfidf_results['Ensemble'] = {
    'model': ensemble, 'preds': ens_preds, 'probs': ens_probs,
    'f1': f1_score(y_val, ens_preds),
    'accuracy': accuracy_score(y_val, ens_preds),
    'precision': precision_score(y_val, ens_preds),
    'recall': recall_score(y_val, ens_preds),
print(f"\n  {'Ensemble':<25s}  {tfidf_results['Ensemble']['f1']:>.4f}  {tfidf_results['Ensemble']['accuracy']:>.4f}  {tfidf_results['Ensemble']['precision']:>.4f}  {tfidf_results['Ensemble']['recall']:>.4f}")
best_name = max(tfidf_results, key=lambda k: tfidf_results[k]['f1'])
    ('tfidf', TfidfVectorizer(max_features=25000, ngram_range=(1,3), min_df=2, max_df=0.95, sublinear_tf=True, strip_accents='unicode')),
    ('clf', LogisticRegres
...[truncated]
```

### Cell 13 code

```
for ax, (name, res) in zip(axes.flat, tfidf_results.items()):
```

### Cell 14 markdown

```
<a id="lstm"></a>
<div style="
    margin: 35px auto 15px;
    padding: 12px 20px;
    background: #292524;
    border-radius: 8px;
    border-left: 3px solid #c67d4a;">
    <p style="color: #c67d4a; font-size: 10px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin: 0 0 3px;">Section 05</p>
    <h2 style="color: #ffffff; font-size: 22px; font-weight: 600; margin: 0;">BiLSTM with GloVe and Attention</h2>
</div>

<div style="
    margin: 0 auto 15px;
    padding: 18px 24px;
    background: #faf9f6;
    border-radius: 10px;
    border: 1px solid #e8e4de;">
    <p style="color: #44403c; font-size: 14px; line-height: 1.8; margin: 0;">
        TF-IDF throws away word order entirely. "Man bites dog" and "dog bites man" produce the same vector. An LSTM reads tokens sequentially and maintains a hidden state that accumulates meaning as it moves through the sentence. Bidirectional means two passes, one forward and one backward so every position has context from both sides. GloVe embeddings trained on 2 billion tweets initialize the word vectors so the model starts with real semantic relationships instead of random weights.
    </p>
</div>
```

### Cell 15 code

```
tokenizer_lstm = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tokenizer_lstm.fit_on_texts(X_train_text)
X_train_seq = pad_sequences(tokenizer_lstm.texts_to_sequences(X_train_text), maxlen=MAX_LEN)
X_val_seq   = pad_sequences(tokenizer_lstm.texts_to_sequences(X_val_text), maxlen=MAX_LEN)
X_test_seq  = pad_sequences(tokenizer_lstm.texts_to_sequences(test_df['text_clean'].values), maxlen=MAX_LEN)
word_index = tokenizer_lstm.word_index
GLOVE_PATH = '/kaggle/input/datasets/robertyoung/glove-twitter-pickles-27b-25d-50d-100d-200d/glove.twitter.27B.100d.pkl'
    inp = keras.Input(shape=(MAX_LEN,))
    x = layers.Lambda(lambda z: tf.keras.backend.sum(z, axis=1))(x)
    return keras.Model(inp, out, name='BiLSTM_Attention')
    optimizer=keras.optimizers.Adam(1e-3),
    metrics=['accuracy']
trainable = sum(tf.keras.backend.count_params(w) for w in lstm_model.trainable_weights)
    validation_data=(X_val_seq, y_val),
        callbacks.EarlyStopping(monitor='val_loss', patience=5,
# ── threshold optimization for LSTM too ──
    f1_t = f1_score(y_val, preds_t)
lstm_f1 = f1_score(y_val, lstm_preds)
print(f"\n  BiLSTM val F1={lstm_f1:.4f}  Acc={accuracy_score(y_val, lstm_preds):.4f}")
print(f"  optimal threshold: {best_lstm_thresh:.2f} (default 0.50 would give F1={f1_score(y_val, (lstm_probs >= 0.5).astype(int)):.4f})")
```

### Cell 17 markdown

```
<a id="bert"></a>
    <h2 style="color: #ffffff; font-size: 22px; font-weight: 600; margin: 0;">Fine-tuned BERT</h2>
    GloVe gives every word one fixed vector regardless of context. "Fire" produces the same embedding whether the tweet reports a building burning or someone getting fired from a job. BERT solves this with self-attention where each token's representation is computed from every other token in the sentence simultaneously. The word "fire" in "the building is on fire" attends to "building" and produces a very different embedding than "fire" in "he got fired last week" which attends to "got". The model was pre-trained on billions of words using masked language modeling so it already understands English deeply. Fine-tuning just teaches it what disaster versus not-disaster looks like on top of that existing knowledge.
```

### Cell 18 code

```
BERT_MODEL = 'bert-base-uncased'
BERT_MAX_LEN = 64
BERT_BATCH = 16
BERT_EPOCHS = 6
BERT_LR = 1e-5 # 2e-5 was overfitting by epoch 3, halved it
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
def tokenize_bert(texts, max_len=BERT_MAX_LEN):
    return bert_tokenizer(
train_enc = tokenize_bert(X_train_bert)
val_enc   = tokenize_bert(X_val_bert)
test_enc  = tokenize_bert(test_df['text_bert'].values)
print(f"  model: {BERT_MODEL}")
print(f"  max length: {BERT_MAX_LEN}  |  batch: {BERT_BATCH}  |  lr: {BERT_LR}")
from tf_keras.optimizers import Adam as TFKerasAdam # Keras 3 and tf-keras optimizers are incompatible, HuggingFace TF models use tf-keras internally
optimizer = TFKerasAdam(
    learning_rate=BERT_LR,
bert_model = TFAutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
bert_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
print(f"  parameters: {bert_model.count_params():,}")
history_bert = bert_model.fit(
    validation_data=(dict(val_enc), y_val),
    epochs=BERT_EPOCHS, batch_size=BERT_BATCH,
        callbacks.EarlyStopping(
            monitor='val_loss', patience=2, # patience=2 because BERT loss can fluctuate between epochs on 6k samples
# ── Threshold Optimization ──
bert_logits = bert_model.predict(dict(val_enc), verbose=0).logits
bert_probs  = tf.nn.softmax(bert_logits, axis=1).numpy()[:, 1]
    preds_t = (bert_probs >= t).astype(int)
    f1_t = f1_score(y_val, preds_t)
bert_preds = (bert_probs >= best_thresh).astype(int)
bert_f1 = f1_score(y_val, bert_preds)
print(f"\n  BERT val F1={bert_f1:.4f}  Acc={accuracy_score(y_val, bert_preds):.4f}")
print(f"  optimal threshold: {best_thresh:.2f} (default 0.50 would give F1={f1_score(y_val, (bert_probs >= 0.5).astype(in
...[truncated]
```

## 20pct rank 173 quality strong score 53

File: `NLP/nlp-getting-started/rank173_20pct/83-6-accuracy-bert-nlp.ipynb`
Cells: 15 total, 8 code, 7 markdown

### Cell 0 markdown

```
# Natural Language Processing: Transfer Learning with BERT
### Predicting Disaster Tweets using HuggingFace and PyTorch
Where traditional models (like TF-IDF or basic RNNs) struggle to capture grammatical nuance and long-term context, modern NLP relies on the "Attention Mechanism." In this notebook, we utilize **Transfer Learning**. We import **DistilBERT**, a Transformer architecture pre-trained on the English Wikipedia corpus, and fine-tune its classification head exclusively on the Kaggle tweet set using native PyTorch.
```

### Cell 1 code

```
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
```

### Cell 2 code

```
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device = torch.device("cpu")
```

### Cell 3 markdown

```
Twitter text is inherently noisy. Before feeding data into the Transformer, we load the raw strings. Unlike classical NLP approaches, we intentionally **skip aggressive text cleaning** (e.g., removing stop-words or lemmatization). Transformers map semantic relationships using the literal grammar of the sentence; prematurely deleting stop-words destroys the attention-context the model relies on.
```

### Cell 4 code

```
train_df = pd.read_csv("/kaggle/input/competitions/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/competitions/nlp-getting-started/test.csv")
```

### Cell 5 markdown

```
## Chapter 2: The BERT Tokenizer
Pre-trained Transformers utilize massive, immutable vocabularies. To ensure our data aligns with the model's pre-existing knowledge vectors, we must use the exact Tokenizer designed for our specific BERT checkpoint.
The tokenizer will convert our raw text into integer IDs, while automatically generating an `attention_mask`. This mask explicitly instructs the AI which tokens contain semantic meaning, and which represent empty padding to be ignored during training.
```

### Cell 6 code

```
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
train_encodings = tokenizer(list(X_train_raw), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test_raw), truncation=True, padding=True, max_length=128)
```

### Cell 7 markdown

```
## Chapter 3: PyTorch Data Handling
For maximum computational efficiency during GPU training, PyTorch requires data to be formally wrapped in a `torch.utils.data.Dataset` subclass.
```

### Cell 8 code

```
class TweetDataset(torch.utils.data.Dataset):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
```

### Cell 9 markdown

```
We now securely download the pre-trained `DistilBERT` architecture. By calling `ForSequenceClassification`, the HuggingFace library automatically removes BERT's original output layer (which was designed for masked language modeling) and attaches a completely blank, randomized 2-neuron classification head (Disaster vs. Safe).
```

### Cell 10 code

```
print(f"DistilBERT successfully loaded onto {device.type.upper()}!")
```

### Cell 11 markdown

```
Because we are modifying a pre-trained architectural masterpiece, standard Deep Learning hyperparameters would destroy the model. Using a standard learning rate (e.g., `1e-3`) would cause violent mathematical updates that shatter BERT's understanding of English.
Instead, we use a microscopic learning rate (`2e-5`) to gently "nudge" the weights. The HuggingFace `Trainer` API abstracts away the complex PyTorch training loop, handling evaluation and optimization under the hood.
```

## 10pct rank 81 quality strong score 73

File: `NLP/nlp-getting-started/rank81_10pct/nlp-disaster-tweets-roberta-5-fold-top-5.ipynb`
Cells: 29 total, 15 code, 14 markdown

### Cell 0 markdown

```
# 🌪️ NLP with Disaster Tweets — RoBERTa 5-Fold Ensemble
> **Public Score: 0.84125** | Top ~5% on the Leaderboard
Starting from simple TF-IDF baselines, we progressively improve to a **RoBERTa 5-Fold Ensemble**
achieving a public score of **0.84125**.
| Approach | Val F1 | Public Score |
| DistilBERT (cleaned text) | 0.7787 | — |
| RoBERTa (raw text) | 0.8262 | 0.83266 |
| **RoBERTa 5-Fold Ensemble** | **~0.81 avg** | **0.84125** |
1. **Raw text > cleaned text for BERT** — hashtags like `#wildfire` carry disaster signals
2. **RoBERTa > DistilBERT** — better pre-training leads to better fine-tuning
3. **5-Fold Ensemble > Single model** — averaging probabilities reduces variance
→ DistilBERT Fine-tuning → RoBERTa Fine-tuning → 5-Fold CV Ensemble → Submission
```

### Cell 2 code

```
!pip install transformers -q
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification,
                          RobertaTokenizer, RobertaForSequenceClassification,
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Cell 4 code

```
train = pd.read_csv('/kaggle/input/competitions/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/competitions/nlp-getting-started/test.csv')
```

### Cell 6 code

```
print(train.groupby('target')[['text_length', 'word_count']].mean().round(2))
```

### Cell 9 markdown

```
> For BERT/RoBERTa, we use **raw text** — the model already understands context,
```

### Cell 11 markdown

```
- Rare but frequent words get high scores → better signal
This is why we need BERT later.
```

### Cell 12 code

```
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(train['clean_text'])
X_test_tfidf = tfidf.transform(test['clean_text'])
print(f"Feature matrix shape: {X_train_tfidf.shape}")
print(f"Each tweet is now a vector of {X_train_tfidf.shape[1]} features")
```

### Cell 13 code

```
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
    scores = cross_val_score(model, X_train_tfidf, train['target'],
    results[name] = scores.mean()
    print(f"{name:<25} F1: {scores.mean():.4f} ± {scores.std():.4f}")
scores = list(results.values())
colors = ['#2E75B6' if s == max(scores) else '#AED6F1' for s in scores]
bars = ax.bar(names, scores, color=colors, edgecolor='white')
ax.set_ylabel('F1 Score')
for bar, score in zip(bars, scores):
            f'{score:.4f}', ha='center', fontsize=10)
print(f"\nBest baseline: Naive Bayes with F1={max(scores):.4f}")
print("→ We need BERT to understand context!")
```

### Cell 15 code

```
    """PyTorch Dataset for tweet tokenizations."""
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
def train_epoch(model, loader, optimizer, scheduler, device):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scheduler.step()
    """Evaluate model, return F1 score."""
    with torch.no_grad():
            predictions = torch.argmax(outputs.logits, dim=1)
    return f1_score(true_labels, preds)
```

### Cell 16 markdown

```
## 7. DistilBERT Baseline
### What is BERT?
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained
Same word, different meaning — BERT handles this perfectly.
### Why DistilBERT?
DistilBERT is a lighter version of BERT:
- **40% smaller** and **60% faster** than BERT
- Retains **97%** of BERT's performance
We don't train BERT from scratch — it would require billions of sentences and months of compute.
```

### Cell 17 code

```
tokenizer_distil = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# We use CLEANED text for DistilBERT (to show the difference with RoBERTa later)
train_enc_distil = tokenizer_distil(
test_enc_distil = tokenizer_distil(
tokens = tokenizer_distil(sample)
print(f"Tokens : {tokenizer_distil.convert_ids_to_tokens(tokens['input_ids'])}")
```

### Cell 18 code

```
                                   generator=torch.Generator().manual_seed(42))
model_distil = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2)
# Optimizer and scheduler
scheduler_distil = get_linear_schedule_with_warmup(
print("Training DistilBERT...")
                      optimizer_distil, scheduler_distil, device)
        torch.save(model_distil.state_dict(), 'best_distilbert.pt')
print(f"\nDistilBERT Best Val F1: {best_distil_f1:.4f}")
```

## 1st rank 7 quality reject score 1

File: `NLP/nlp-getting-started/rank7_1st/disaster-tweets-classification-with-distilbert.ipynb`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
import tensorflow as tf
import keras
import keras_nlp
df_test = pd.read_csv("/kaggle/input/competitions/nlp-getting-started/test.csv")
sample_submission = pd.read_csv("/kaggle/input/competitions/nlp-getting-started/sample_submission.csv")
model = keras.models.load_model("/kaggle/input/models/jek1wantaufik/buddy/keras/nlp/1/output/distilbert_model.keras", compile=False)
sample_submission["target"] = np.argmax(preds, axis=1)
sample_submission.to_csv("/kaggle/working/submission.csv", index=False)
```
