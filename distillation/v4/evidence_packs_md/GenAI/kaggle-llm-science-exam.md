# GenAI / kaggle-llm-science-exam

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1916 quality strong score 67

File: `GenAI/kaggle-llm-science-exam/rank1916_70pct/2023kagglellm-deberta-v3-large-model1-inference.ipynb`
Cells: 7 total, 6 code, 1 markdown

### Cell 0 markdown

```
This DeBERTa-V3-Large model was trained using:
```

### Cell 1 code

```
import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
```

### Cell 2 code

```
    tokenized_example = tokenizer(first_sentence, second_sentence, truncation=True)
    tokenizer: PreTrainedTokenizerBase
        batch = self.tokenizer.pad(
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
    sorted_answer_indices = np.argsort(-predictions)
#test_df = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/test.csv')
test_df_all = pd.read_csv('/kaggle/input/more-data/train_3000.csv')
```

### Cell 4 code

```
model_dir = '/kaggle/input/2023kagglellm-deberta-v3-large-model1'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
submission_df.to_csv('submission.csv', index=False)
```

### Cell 5 code

```
def score(prediction,answer,num):
#print(score(subm
score(submission_df,test_df,1251) #  #0.673(original)
```

### Cell 6 code

```
test_df = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/test.csv')
submission_df.to_csv('submission.csv', index=False)
```

## 40pct rank 1096 quality strong score 61

File: `GenAI/kaggle-llm-science-exam/rank1096_40pct/explained-platypus2-70b-wikipedia-rag.ipynb`
Cells: 23 total, 8 code, 15 markdown

### Cell 0 markdown

```
# Explained Platypus2-70B + Wikipedia RAG

## Introduction 🌟
Welcome to this Jupyter notebook developed for Kaggle - LLM Science Exam This notebook is designed to help you participate in the competition and to Use LLMs to answer difficult science questions


### Inspiration and Credits 🙌
This notebook is inspired by the work of simjeg
, available at [this Kaggle project](https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag/notebook). I extend my gratitude to simjeg
 for sharing their insights and code.

🌟 Explore my profile and other public projects, and don't forget to share your feedback!
👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈

🙏 Thank you for taking the time to review my work, and please give it a thumbs-up if you found it valuable! 👍

## Purpose 🎯
The primary purpose of this notebook is to:
- Load and preprocess the competition data 📁
- Engineer relevant features for model training 🏋️‍♂️
- Train predictive models to make target variable predictions 🧠
- Submit predictions to the competition environment 📤

## Notebook Structure 📚
This notebook is structured as follows:
1. **Data Preparation**: In this section, we load and preprocess the competition data.
2. **Feature Engineering**: We generate and select relevant features for model training.
3. **Model Training**: We train machine learning models on the prepared data.
4. **Prediction and Submission**: We make predictions on the test data and submit them for evaluation.



```

### Cell 5 markdown

```
- Libraries include standard Python libraries (e.g., `gc`, `logging`, `time`) and popular data science libraries (e.g., `numpy`, `pandas`, `torch`).
```

### Cell 6 code

```
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
from safetensors.torch import load_file
```

### Cell 9 code

```
    torch.cuda.empty_cache()
df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/test.csv", index_col="id")
# df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv", index_col="id")
```

### Cell 11 markdown

```
- In the `__init__` method, the pre-trained model and tokenizer are loaded based on the specified checkpoint.
- The `transform` method tokenizes and preprocesses sentences using the tokenizer.
```

### Cell 12 code

```
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokens = self.tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt", max_length=MAX_SEQ_LEN)
            with torch.no_grad():
```

### Cell 13 markdown

```
## Processing and Extracting Context for Test Set**

**Explanation:**
- This cell processes and extracts context for the test set if `IS_TEST_SET` is `True`.
- It starts by loading an embedding model (`SentenceTransformer`) and initializing a timer (`start`) to measure the elapsed time.
- The prompts and answer choices are combined into a single string using a lambda function and applied to the DataFrame, resulting in `inputs` for embedding.
- A Faiss index is loaded to perform efficient nearest neighbor search for text matching.
- Text search is performed using Faiss by searching for the closest sentences to the prompt embeddings.
- Context is extracted from a pre-loaded dataset, and each entry in the `df` DataFrame is updated with the extracted context.
- Memory is freed, including resetting the Faiss index, deleting variables, and running the `clean_memory()` function.
- The elapsed time for the entire process is printed.

**Note:**
- The Faiss library is used for efficient similarity search, which can significantly speed up the search for relevant passages in a large dataset.
```

### Cell 15 markdown

```
##  Creating Symlinks from Kaggle Datasets to Cached Model


**Explanation:**
- This cell creates symbolic links (symlinks) from dataset files in Kaggle to a cached model checkpoint directory.
- It starts by defining the `checkpoint_path`, which is the directory where the symlinks will be created.
- The script then loops through two parts (part 1 and part 2).
- For each part, it defines the `source_dir`, which is the directory containing dataset files.
- It then iterates through files in the `source_dir`.
- For each file, it attempts to create a symlink in the `checkpoint_path` directory with the same name, pointing to the original file in the `source_dir`.
- Any exceptions that occur during symlink creation are caught and ignored.

**Note:**
- This code is useful for setting up symlinks between Kaggle dataset files and a cached model directory, potentially reducing the need to download data repeatedly.
```

### Cell 17 markdown

```
 ## 🦙ShardedLlama Class for Language Model
```

### Cell 18 markdown

```
2. `def __init__(self, checkpoint_path, device="cuda:0", dtype=torch.float16):`: Defines the constructor for the `ShardedLlama` class, which initializes an instance of the class.
   - `dtype` (optional): The data type for tensors (default is torch.float16).
   - `self.tokenizer`: Initializes a tokenizer using `AutoTokenizer` from the checkpoint.
   - Sets the tokenizer's `pad_token` to the tokenizer's `eos_token` and sets `padding_side` to "right."
```

### Cell 19 code

```
    def __init__(self, checkpoint_path, device="cuda:0", dtype=torch.float16):
        dtype : torch.dtype, optional
            dtype, by default torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
            assert param.dtype != torch.int8, "int8 not supported (need to add fp16_statistics)"
        suffix_eos = [(suffix != self.tokenizer.pad_token_id).sum(1) - 1 for _, suffix in inputs]
        attention_mask = torch.finfo(self.dtype).min * torch.ones(MAX_LENGTH, MAX_LENGTH)
        position_ids = torch.arange(MAX_LENGTH, dtype=torch.long, device=self.device)[None, :]
        with ThreadPoolExecutor() as executor, torch.inference_mode():
                        batch[j] = (None, layer(suffix[torch.arange(n_suffixes), suffix_eos[j]][:, None]))
        # Get scores
```

### Cell 20 markdown

```
## Running the Model on Multiple GPUs
```

## 20pct rank 365 quality usable score 43

File: `GenAI/kaggle-llm-science-exam/rank365_20pct/retriever-reader-model-with-flexible-sharding.ipynb`
Cells: 17 total, 11 code, 6 markdown

### Cell 0 markdown

```
Retriever-Reader models, particularly the Contextualized Retrieval-Augmented Generation (RAG) approach, represent a breakthrough in Large Language Models (LLMs) such as GPT-3. These models integrate a dual-process system:
### Ranking Criterion
- **BM25 Scoring**: Our system utilizes Lucene's BM25 scoring function for text retrieval. BM25 optimizes document ranking by considering term frequency and document length, providing a balanced approach to information retrieval. This method is crucial for selecting the most relevant passages to inform the LLM's responses.
```

### Cell 5 code

```
    merge_method: str = "none",
                _res["score"] = hits[_k].score
        scores = retrieved_results_df["score"].values
        # Add rank score, which is much bigger than BM25 score.
        scores = scores.reshape(5, top_k) + (np.arange(top_k)[::-1] * 10000.0)
        scores = scores.reshape(5 * top_k)
        retrieved_results_df["rank_bm25_score"] = scores
        retrieved_results_df.sort_values("rank_bm25_score", ascending=False, inplace=True)
    if merge_method == "none":
    elif merge_method == "mergev2":
        merged_results = []
                merged_results.append({"contents": contents})
                # print(f"[DEBUG] merged until {_k=}")
            merged_results.append({"contents": contents})
            # print(f"[DEBUG] merged until {len(retrieved_results)}")
        retrieved_results = merged_results
        raise ValueError(f"[ERROR] Unexpected value merge_method={merge_method}")
```

### Cell 6 code

```
    merge_method: str = "none",
        delayed(retrieval_row)(searcher, row, top_k, jsonl, context_template, max_length, merge_method, search_method)
```

### Cell 7 code

```
df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv", index_col="id")
```

### Cell 13 markdown

```
### Create symlinks from kaggle datasets to fake cached model. This step could be skipped on local machines.
```

### Cell 15 code

```
                        --prompt_pickle input_prompts.pkl --output_file platypus2_output_score.pkl
```

### Cell 16 code

```
with open('platypus2_output_score.pkl', 'rb') as file:
    output_scores = pickle.load(file)
n = len(output_scores)
for i, scores in enumerate(output_scores):
    first_token_scores = scores[:, 0, [5852, 7700]]
    positive_scores = first_token_scores[:, 0] / first_token_scores.sum(axis=-1)
    top3 = np.argsort(positive_scores)[::-1][:3]
```

## 10pct rank 346 quality strong score 61

File: `GenAI/kaggle-llm-science-exam/rank346_10pct/llm-science.ipynb`
Cells: 25 total, 22 code, 3 markdown

### Cell 0 markdown

```
### @CSC494, this is the publicly-shared notebook that you need to add comments and some form of EDA and send as part of your hands-on midterm. it is copied-and-edited from the outstanding work of @mbanaei at https://www.kaggle.com/code/mbanaei/86-2-with-only-270k-articles. Do not modify any other parts of this notebook for your midterm. You are welcome to copy-and-edit another version that you may modify and try to improve upon the score if you like. Do not submit that for your midterm though.
```

### Cell 2 markdown

```
- Although the majority of high-performing public models use DeBERTa-V3 to do the inference in their pipeline, I used a LongFormer Large model, which enables us to have a much longer prefix context given limited GPU memory. More specifically, as opposed to many public notebooks, there's no splitting to sentence level, and the whole paragraph is retrieved and passed to the classifier as a context (the main reason that we don't get OOM and also have relatively fast inference is that in LongFormer full attention is not computed as opposed to standard models like BERT).
```

### Cell 4 code

```
!cp -rf /kaggle/input/sentence-transformers-222/sentence-transformers /kaggle/working/sentence-transformers
!pip install -U /kaggle/working/sentence-transformers
!pip install --no-index --no-deps /kaggle/input/llm-whls/transformers-4.31.0-py3-none-any.whl
```

### Cell 6 code

```
backup_model_predictions = pd.read_csv("submission_backup.csv")
```

### Cell 9 code

```
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
#from transformers import LongformerTokenizer, LongformerForMultipleChoice
import transformers
```

### Cell 11 code

```
def get_relevant_documents_parsed(df_valid):
    for idx in tqdm(range(0, df_valid.shape[0], df_chunk_size)):
        df_valid_ = df_valid.iloc[idx: idx+df_chunk_size]
        articles_indices, merged_top_scores = retrieval(df_valid_, modified_texts)
        all_articles_values.append(merged_top_scores)
def get_relevant_documents(df_valid):
    for idx in tqdm(range(0, df_valid.shape[0], df_chunk_size)):
        df_valid_ = df_valid.iloc[idx: idx+df_chunk_size]
        articles_indices, merged_top_scores = retrieval(df_valid_, modified_texts)
        all_articles_values.append(merged_top_scores)
def retrieval(df_valid, modified_texts):
    corpus_df_valid = df_valid.apply(lambda row:
    vectorizer1 = TfidfVectorizer(ngram_range=(1,2),
    vectorizer1.fit(corpus_df_valid)
    vocab_df_valid = vectorizer1.get_feature_names_out()
    vectorizer = TfidfVectorizer(ngram_range=(1,2),
                                 vocabulary=vocab_df_valid)
    corpus_tf_idf = vectorizer.transform(corpus_df_valid)
        temp_scores = (corpus_tf_idf * wiki_vectors.T).toarray()
        chunk_top_indices = temp_scores.argpartition(-top_per_chunk, axis=1)[:, -top_per_chunk:]
        chunk_top_values = temp_scores[np.arange(temp_scores.shape[0])[:, np.newaxis], chunk_top_indices]
    merged_top_scores = np.sort(top_values_array, axis=1)[:,-top_per_query:]
    merged_top_indices = top_values_array.argsort(axis=1)[:,-top_per_query:]
    articles_indices = top_indices_array[np.arange(top_indices_array.shape[0])[:, np.newaxis], merged_top_indices]
    return articles_indices, merged_top_scores
        tokenizer,
    c_plus_q   = context + ' ' + tokenizer.bos_token + ' ' + question
    tokenized_examples = tokenizer(
```

### Cell 13 code

```
df_valid = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/test.csv")
```

### Cell 14 code

```
df_valid.info()
```

### Cell 15 code

```
df_valid.head()
```

### Cell 16 code

```
retrieved_articles_parsed = get_relevant_documents_parsed(df_valid)
```

### Cell 17 code

```
retrieved_articles = get_relevant_documents(df_valid)
```

### Cell 19 code

```
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## 1st rank 7 quality usable score 42

File: `GenAI/kaggle-llm-science-exam/rank7_1st/d01-wiki-urls.ipynb`
Cells: 61 total, 49 code, 12 markdown

### Cell 30 code

```
    "Category:Mathematical_proofs": 2,
```
