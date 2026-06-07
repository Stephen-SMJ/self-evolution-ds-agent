# Audio / birdclef-2023

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 779 quality strong score 55

File: `Audio/birdclef-2023/rank779_70pct/transform-yamnet-log-mel-to-tfrecord.ipynb`
Cells: 23 total, 23 code, 0 markdown

### Cell 0 code

```
import tensorflow as tf
import tensorflow_io as tfio
```

### Cell 1 code

```
    valid_dir = os.path.join(output, "validation")
```

### Cell 5 code

```
df = pd.read_csv(Config.metada_file)
```

### Cell 8 code

```
import tensorflow_hub as hub
yamnetlayer = hub.KerasLayer('https://tfhub.dev/google/yamnet/1', trainable=False)
```

### Cell 11 code

```
        examples = map(create_example, tf.unstack(fl, axis=0), tf.unstack(ll, axis=0))
    dff.to_csv(os.path.join(Config.output, f"{name}.csv"))
```

### Cell 14 code

```
    df_train, df_valid = None, None
            dff_train, dff_valid = train_test_split(dff, test_size=test_size, random_state=random_state)
            df_valid = dff_valid if df_valid is None else pd.concat([df_valid, dff_valid])
    return df_train, df_valid
```

### Cell 15 code

```
train_df, valid_df = train_split_label_balanced(df, labels=df["primary_label"].unique(), test_size=0.3)
```

### Cell 18 code

```
Config.valid_dir
```

### Cell 19 code

```
print("Validation size | Size: ", len(valid_df))
filepaths = valid_df["filepath"]
labels = valid_df["label"]
frame2records(filepaths, labels, path_records=Config.valid_dir, record_size=Config.records_size, name="validation")
```

### Cell 20 code

```
df_train_metadata = pd.read_csv(os.path.join("/kaggle/working/birdclef-2023", "train.csv"))
```

### Cell 21 code

```
df_train_metadata = pd.read_csv(os.path.join("/kaggle/working/birdclef-2023", "validation.csv"))
```

## 40pct rank 437 quality usable score 43

File: `Audio/birdclef-2023/rank437_40pct/notedta-2.ipynb`
Cells: 13 total, 13 code, 0 markdown

### Cell 0 code

```
import torch
import torch.nn as nn
import torchvision
import torchaudio  # importing torchaudio library for audio processing
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
```

### Cell 1 code

```
cfg.data_folder = ''
cfg.train_data_folder = cfg.data_dir + "train_audio/"
cfg.val_data_folder = cfg.data_dir + "train_audio/"
cfg.test_data_folder = cfg.data_dir + "test_soundscapes/soundscape_29201.ogg"
```

### Cell 2 code

```
        sig, sr = torchaudio.load(audio_file)
          resig = torch.cat([sig, sig, sig])
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), 1)
    def time_shift(aud, shift_limit):
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)
        spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
            aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
            aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
```

### Cell 3 code

```
    shift_pct = 0.4
    shift_aud = AudioUtil.time_shift(dur_aud, shift_pct)
    sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
```

### Cell 4 code

```
    __train_data_path = cfg.train_data_folder
    __train_df = pd.read_csv(cfg.train_df)
    __le=preprocessing.LabelEncoder()
            #RawData.__le = preprocessing.LabelEncoder()
            #RawData.__labels_t = torch.as_tensor(targets)
```

### Cell 5 code

```
        #audio = torch.tensor(audio)
        chunks = audio.unfold(1, frame_length, frame_step).transpose(0, 1)
```

### Cell 8 code

```
train_df = pd.read_csv(cfg.train_df)
```

### Cell 9 code

```
    loadpath = cfg.train_data_folder+ data['filename']
    aud,sr = torchaudio.load(loadpath)
    torch.save(signal,savepath)
```

### Cell 10 code

```
'''train_df = pd.read_csv(cfg.train_df)
    loadpath = cfg.train_data_folder+ data['filename']
    aud,sr = torchaudio.load(loadpath)
        torch.save(signal,savepath)
```

### Cell 11 code

```
pd_lists.to_csv(save_data_path+'lists.csv',index=False)
```

## 20pct rank 197 quality strong score 67

File: `Audio/birdclef-2023/rank197_20pct/baseline-clf.ipynb`
Cells: 56 total, 32 code, 24 markdown

### Cell 3 code

```
!pip install timm
```

### Cell 5 code

```
# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import timm
```

### Cell 6 markdown

```
<img src="https://i.imgur.com/gb6B4ig.png" width="400" alt="Weights & Biases" />

<span style="color: #000508; font-family: Segoe UI; font-size: 1.2em; font-weight: 300;"> Weights & Biases (W&B) is a set of machine learning tools that helps you build better models faster. <strong>Kaggle competitions require fast-paced model development and evaluation</strong>. There are a lot of components: exploring the training data, training different models, combining trained models in different combinations (ensembling), and so on.</span>

> <span style="color: #000508; font-family: Segoe UI; font-size: 1.2em; font-weight: 300;">⏳ Lots of components = Lots of places to go wrong = Lots of time spent debugging</span>

<span style="color: #000508; font-family: Segoe UI; font-size: 1.2em; font-weight: 300;">To learn more about Weights and Biases check out this <strong><a href="https://www.kaggle.com/ayuraj/experiment-tracking-with-weights-and-biases">kernel</a></strong>.</span>
```

### Cell 9 code

```
      "model_name": "tf_efficientnet_b0_ns",
      "valid_batch_size": 128,
      "scheduler": 'CosineAnnealingLR',
      "n_fold": 5,
      "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
```

### Cell 12 code

```
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Cell 17 code

```
df = pd.read_csv(f"{ROOT_DIR}/train_metadata.csv")
```

### Cell 23 code

```
encoder = LabelEncoder()
```

### Cell 25 code

```
skf = StratifiedKFold(n_splits=CONFIG['n_fold'])
for fold, ( _, val_) in enumerate(skf.split(X=df, y=df.primary_label)):
      df.loc[val_ , "kfold"] = fold
```

### Cell 26 code

```
assert(df.groupby("kfold").size().sum() == df.shape[0])
df.groupby("kfold").size()
```

### Cell 28 code

```
        audio, sample_rate = torchaudio.load(filepath)
        label_onehot = torch.zeros(CONFIG['num_classes'])
        label_onehot[self.labels[index]] = 1
        label = torch.tensor(self.labels[index])
        image = torch.stack([mel, mel, mel])
        max_val = torch.abs(image).max()
            label_onehot=label_onehot,
        return torch.mean(audio, axis=0)
```

### Cell 30 code

```
        self.p = nn.Parameter(torch.ones(1)*p)
```

### Cell 31 markdown

```
# <span><h1 style = "font-family: garamond; font-size: 40px; font-style: normal; letter-spcaing: 3px; background-color: #f6f5f5; color :#fe346e; border-radius: 100px 100px; text-align:center">Create Model</h1></span>
```

## 10pct rank 113 quality strong score 55

File: `Audio/birdclef-2023/rank113_10pct/google-bird-model-embeddings-predictions-score.ipynb`
Cells: 16 total, 12 code, 4 markdown

### Cell 0 code

```
import tensorflow_hub as hub
import tensorflow as tf
import torchaudio
import torch
from torch.utils.data import DataLoader, Dataset
df = pd.read_csv('/kaggle/input/birdclef-2023/train_metadata.csv')
model = hub.load('https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/2')
model_labels_df = pd.read_csv(hub.resolve('https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/tensorFlow2/variations/bird-vocalization-classifier/versions/2') + "/assets/label.csv")
```

### Cell 3 code

```
# use a torch dataloader to decode audio in parallel on CPU while GPU is running
        audio = torchaudio.load(AUDIO_PATH / filename)[0].numpy()[0]
        all_embeddings[filename] = np.stack(file_embeddings)
        all_predictions[filename] = np.stack(file_predictions)
torch.save(all_embeddings, 'embeddings.pt')
torch.save(all_predictions, 'predictions.pt')
```

### Cell 4 markdown

```
# Scores of predictions on the first 5 seconds of each recording
```

### Cell 5 code

```
predicted_classes = torch.tensor([row[0].argmax() for row in all_predictions.values()])
actual_classes = torch.tensor([label_to_index[label] for label in df.primary_label])
```

### Cell 6 code

```
logits = torch.stack([torch.tensor(row[0]) for row in all_predictions.values()])
ce_loss = torch.nn.CrossEntropyLoss()(logits, actual_classes)
```

### Cell 7 code

```
actual_probs = torch.eye(len(bc2023_labels))[actual_classes]
bce_loss = torch.nn.BCEWithLogitsLoss()(logits, actual_probs)
```

### Cell 8 code

```
import sklearn.metrics
    score = sklearn.metrics.average_precision_score(
    return score
```

### Cell 9 code

```
    submission=pd.DataFrame(torch.softmax(logits, 1).numpy(), columns=bc2023_labels),
```

### Cell 10 code

```
    submission=pd.DataFrame(torch.sigmoid(logits).numpy(), columns=bc2023_labels),
```

### Cell 15 code

```
import torchaudio
compute_melspec = torchaudio.transforms.MelSpectrogram(
power_to_db = torchaudio.transforms.AmplitudeToDB(
    audio = torchaudio.load(AUDIO_PATH / df.filename[index], start, start+WINDOW)[0][0]
probs = torch.sigmoid(logits)
    rank = 1 + sorted(probs[i], reverse=True).index(predicted_prob_for_correct_label)
    print(f'correct was {correct_class}, given prob {predicted_prob_for_correct_label} and ranked #{rank}')
```

## 1st rank 2 quality strong score 54

File: `Audio/birdclef-2023/rank2_1st/bird-clef-2023-inference-v1.ipynb`
Cells: 39 total, 30 code, 9 markdown

### Cell 3 code

```
sys.path.append('/kaggle/input/bird-clef-2023-code/main_folder/main_folder/')
```

### Cell 4 code

```
import torch
from code_base.models import WaveCNNClasifier, WaveCNNAttenClasifier
from code_base.utils import load_json, compose_submission_dataframe, groupby_np_array, stack_and_max_by_samples
from code_base.utils.metrics import padded_cmap_numpy
```

### Cell 7 code

```
    "run_validation": False,
    "folds":[0],
#     "label_map_data_path": "/kaggle/input/bird-clef-2023-models/xc_birds_202x_only_scored.json",
#     "model_class": WaveCNNAttenClasifier,
if "folds" not in CONFIG:
    CONFIG["folds"] = list(range(CONFIG["n_folds"]))
```

### Cell 10 code

```
if CONFIG["run_validation"]:
    df = pd.read_csv(CONFIG["train_df_path"])
    val_df = [df.iloc[split[fold_id][1]].reset_index(drop=True) for fold_id in CONFIG["folds"]]
```

### Cell 12 code

```
if CONFIG["run_validation"]:
       "validate_sr": 32_000
        "validate_sr": 32_000
```

### Cell 13 code

```
if CONFIG["run_validation"]:
```

### Cell 14 code

```
if CONFIG["run_validation"]:
    loader_val = [torch.utils.data.DataLoader(
    loader_test = torch.utils.data.DataLoader(
```

### Cell 15 markdown

```
# Model
```

### Cell 16 code

```
        t_chkp = torch.load(model_chkp, map_location="cpu")
```

### Cell 17 code

```
#         model_chkp=f"/kaggle/input/bird-clef-2023-models/{CONFIG['exp_name']}/{CONFIG['exp_name']}/fold_{m_i}/checkpoints/{CONFIG['chkp_name']}",
# ) for m_i in CONFIG["folds"]]
```

### Cell 22 code

```
if CONFIG["run_validation"]:
```

### Cell 24 code

```
    sample_submission = pd.read_csv("/kaggle/input/birdclef-2023/sample_submission.csv")
    if test_pred_df.shape[1] > sample_submission.shape[1]:
        test_pred_df = test_pred_df[sample_submission.columns]
```
