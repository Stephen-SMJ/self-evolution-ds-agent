# CV / digit-recognizer

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1062 quality strong score 65

File: `CV/digit-recognizer/rank1062_70pct/digitrecognizer.ipynb`
Cells: 13 total, 7 code, 6 markdown

### Cell 1 code

```
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

### Cell 3 code

```
train = pd.read_csv('/kaggle/input/competitions/digit-recognizer/train.csv')
```

### Cell 4 code

```
# Reshaping is necessary for the Data Augmentation part
```

### Cell 5 markdown

```
## **Data Augmentation and Model Builidng**
```

### Cell 6 code

```
    width_shift_range=0.1,
    height_shift_range=0.1
    Input(shape=(28, 28, 1)), # OG code used to be Flatten(input_shape=(28, 28, 1)) but in newer versions of Tensorflow, a cleaner version exists
              metrics=['accuracy'])
                    validation_data=(X_val, y_val))
```

### Cell 7 markdown

```
## **Plotting a confusion matrix to see where exactly is the model going wrong**
```

### Cell 9 markdown

```
## **We observe that the model is getting confused mainly in the 3/8 and 2/7 numbers**
```

### Cell 12 code

```
test = pd.read_csv('/kaggle/input/competitions/digit-recognizer/test.csv')
submission.to_csv('submission.csv', index=False)
```

## 40pct rank 548 quality strong score 55

File: `CV/digit-recognizer/rank548_40pct/sine-multi-scale-cnn-99-1399-params-11-class.ipynb`
Cells: 11 total, 11 code, 0 markdown

### Cell 0 code

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
```

### Cell 1 code

```
train_df = pd.read_csv("/kaggle/input/competitions/digit-recognizer/train.csv")
test_df  = pd.read_csv("/kaggle/input/competitions/digit-recognizer/test.csv")
# Augmentation makes exploration space even more vast
X_train, X_valid = X, X
y_train, y_valid = y, y
```

### Cell 2 code

```
import torchvision.transforms.functional as TF
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    mat = torch.tensor([[sx * cos, -sy * sin, tx / x.shape[-1]],
    dx = torch.randn(1, 1, h, w) * alpha
    dy = torch.randn(1, 1, h, w) * alpha
    coords = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-coords**2 / (2 * sigma**2))
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1,1,h), torch.linspace(-1,1,w), indexing='ij')
    grid = torch.stack([grid_x + dx.squeeze() / w, grid_y + dy.squeeze() / h], dim=-1).unsqueeze(0)
    def __init__(self, X, y=None, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
        self.augment = augment
        if self.augment:
            x = x + torch.randn_like(x) * 0.2
```

### Cell 3 code

```
sys.path.append('/kaggle/input/models/elvenmonk/sine-multi-scale-cnn-1390-params-mnist-99/pytorch/default/6')
from SinCNN import SinCNN
model = SinCNN().to(device)
```

### Cell 4 code

```
path = f'/kaggle/input/models/elvenmonk/sine-multi-scale-cnn-1390-params-mnist-99/pytorch/default/6/SinRCNN_{total}.pth'
    model.load_state_dict(torch.load(path, map_location=device))
```

### Cell 5 code

```
train_loader = DataLoader(MNISTDataset(X_train, y_train, augment=True), batch_size=512, shuffle=True, num_workers=0)
valid_loader = DataLoader(MNISTDataset(X_valid, y_valid), batch_size=512, num_workers=2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler.step()
    with torch.no_grad():
            for xb, yb in valid_loader
    print(f"Epoch {epoch}: {correct}/{len(X_valid)} ({100*correct/len(X_valid):.2f}%)")
```

### Cell 6 code

```
#torch.save(model.state_dict(), path)
with torch.no_grad():
pd.DataFrame({'ImageId': range(1, len(preds)+1), 'Label': preds}).to_csv('submission.csv', index=False)
```

### Cell 7 code

```
with torch.no_grad():
    for xb, yb in valid_loader:
preds = torch.cat(all_preds)
labels = torch.cat(all_labels)
imgs = torch.cat(all_imgs)
```

### Cell 8 code

```
    with torch.no_grad():
```

### Cell 9 code

```
# label distribution in validation set
axes[0].hist(y_valid, bins=np.arange(11)-0.5, edgecolor='black')
axes[0].set_title('Validation label distribution')
```

## 20pct rank 374 quality strong score 73

File: `CV/digit-recognizer/rank374_20pct/cnn-scratch-vs-lenet-5-digit-classification.ipynb`
Cells: 39 total, 23 code, 16 markdown

### Cell 1 code

```
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten ,Dropout,Conv2D ,MaxPooling2D , BatchNormalization , ReLU,MaxPooling2D,Flatten,AveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
```

### Cell 2 code

```
train = pd.read_csv("/kaggle/input/competitions/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/competitions/digit-recognizer/test.csv")
```

### Cell 9 markdown

```
# **3-Splitting Validation**
```

### Cell 13 markdown

```
# **5-CNN Scratch Model Structure**
```

### Cell 14 code

```
    Conv2D(32 , (3,3),padding="valid",input_shape=(28,28,1)),
    Conv2D(128 , (3,3),padding="valid",),
```

### Cell 15 markdown

```
# **6-Compiling & CallBacks**
```

### Cell 16 code

```
    metrics=["accuracy"]
```

### Cell 17 code

```
early_stop = EarlyStopping(
```

### Cell 18 markdown

```
# **7-Model Training**
```

### Cell 19 code

```
                 validation_data=(x_val, y_val),
```

### Cell 22 code

```
# -- Plot training vs validation accuracy --
# -- Plot training vs validation loss --
```

### Cell 23 markdown

```
# **9-Test Model With Images**
```

## 10pct rank 164 quality strong score 73

File: `CV/digit-recognizer/rank164_10pct/digit-recognizer-cnn.ipynb`
Cells: 27 total, 14 code, 13 markdown

### Cell 1 code

```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print('TensorFlow version:', tf.__version__)
```

### Cell 3 code

```
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```

### Cell 4 code

```
train_df = pd.read_csv('/kaggle/input/competitions/digit-recognizer/train.csv')
test_df  = pd.read_csv('/kaggle/input/competitions/digit-recognizer/test.csv')
```

### Cell 8 code

```
y_cat = keras.utils.to_categorical(y, num_classes=10)
# Split train / validation (90% / 10%)
```

### Cell 12 code

```
    model = keras.Sequential([
```

### Cell 13 markdown

```
## 6. Compilation et callbacks
```

### Cell 14 code

```
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
    keras.callbacks.ReduceLROnPlateau(
    keras.callbacks.EarlyStopping(
```

### Cell 16 code

```
datagen = keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1
print('Data augmentation configurée.')
```

### Cell 18 code

```
    validation_data=(X_val, y_val),
```

### Cell 20 code

```
ax1.plot(history.history['val_accuracy'], label='Validation')
ax2.plot(history.history['val_loss'], label='Validation')
print(f'Meilleure accuracy validation : {val_acc:.4f} ({val_acc*100:.2f}%)')
```

### Cell 22 code

```
from sklearn.metrics import confusion_matrix, classification_report
plt.title('Matrice de confusion (validation)')
```

### Cell 24 code

```
submission.to_csv('submission.csv', index=False)
```

## 1st rank 7 quality strong score 64

File: `CV/digit-recognizer/rank7_1st/digit-recognizer-cnn-public-mnist-variants.ipynb`
Cells: 6 total, 5 code, 1 markdown

### Cell 0 markdown

```
# Digit Recognizer: high-score CNN variants
This notebook trains two compact CNN variants for Digit Recognizer:
- public MNIST 70k training source: submitted public score `0.99835`;
- public MNIST 60k training source: submitted public score `0.99767`.
generated CSV files are CNN predictions, not direct lookup labels.
```

### Cell 1 code

```
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
            if (candidate / 'test.csv').exists() and (candidate / 'sample_submission.csv').exists():
    WORK_DIR = Path(r'C:/kaggle/digit-recognizer/outputs/notebook_cnn_check')
def pick_device() -> torch.device:
    if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
                return torch.device('cuda')
    return torch.device('cpu')
```

### Cell 2 code

```
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
def to_tensor(images: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(images.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0)
class SmallCnn(nn.Module):
```

### Cell 3 code

```
train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df = pd.read_csv(DATA_DIR / 'test.csv')
sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
```

### Cell 4 code

```
@torch.no_grad()
    model = SmallCnn().to(DEVICE)
        TensorDataset(to_tensor(train_x), torch.from_numpy(train_y)),
        pin_memory=torch.cuda.is_available(),
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    out_path = WORK_DIR / f'submission_cnn_{source}.csv'
    submission.to_csv(out_path, index=False)
```

### Cell 5 code

```
    print('CPU fallback smoke run; submitted scores in the markdown are from the full local GPU run.')
```
