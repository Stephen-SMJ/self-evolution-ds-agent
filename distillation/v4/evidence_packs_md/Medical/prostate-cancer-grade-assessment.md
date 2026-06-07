# Medical / prostate-cancer-grade-assessment

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 709 quality usable score 43

File: `Medical/prostate-cancer-grade-assessment/rank709_70pct/getting-started-with-the-panda-dataset.ipynb`
Cells: 48 total, 28 code, 20 markdown

### Cell 2 code

```
train_labels = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv').set_index('image_id')
```

### Cell 8 markdown

```
Running the cell below loads four example biopsies using OpenSlide. Some things you can notice:

- The image dimensions are quite large (typically between 5.000 and 40.000 pixels in both x and y).
- Each slide has 3 levels you can load, corresponding to a downsampling of 1, 4 and 16. Intermediate levels can be created by downsampling a higher resolution level.
- The dimensions of each level differ based on the dimensions of the original image.
- Biopsies can be in different rotations. This rotation has no clinical value, and is only dependent on how the biopsy was collected in the lab.
- There are noticable color differences between the biopsies, this is very common within pathology and is caused by different laboratory procedures.
```

### Cell 9 code

```
    print(f"Gleason score: {train_labels.loc[case_id, 'gleason_score']}\n\n")
```

### Cell 12 markdown

```
Using the `level` argument we can easily load in data from any level that is present in the slide. Coordinates passed to `read_region` are always relative to level 0 (the highest resolution).
```

## 40pct rank 311 quality usable score 43

File: `Medical/prostate-cancer-grade-assessment/rank311_40pct/mi-zero-demo.ipynb`
Cells: 32 total, 32 code, 0 markdown

### Cell 1 code

```
!pip install assets/timm_ctp.tar --no-deps
```

### Cell 3 code

```
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
```

### Cell 6 code

```
def load_ctranspath_clip(ckpt_path, img_size = 224, return_trsforms = True):
    def clean_state_dict_clip(state_dict):
    state_dict = torch.load(ckpt_path, map_location="cpu")['state_dict']
    state_dict = clean_state_dict_clip(state_dict)
ckpt_path = "/kaggle/input/mizerow/ctranspath_448_bioclinicalbert/ctranspath_448_bioclinicalbert/checkpoints/epoch_50.pt"
model, trsforms = load_ctranspath_clip(ckpt_path, img_size = 224)
```

### Cell 15 code

```
with torch.inference_mode():
```

### Cell 16 code

```
features_ = torch.cat(features_)
```

### Cell 17 code

```
def tokenize(tokenizer, texts=["a histopathological image of adipocytes"]):
    tokens = tokenizer.batch_encode_plus(texts,
from transformers import AutoTokenizer
model_name = 'emilyalsentzer/Bio_ClinicalBERT'
tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
```

### Cell 18 code

```
        device: torch.device = torch.device('cpu'),
    with open("./src/model_configs/ctranspath_448_bioclinicalbert.json") as f:
    model = CLIP(**model_cfg)
    state_dict = torch.load(model_checkpoint, map_location='cpu')['state_dict']
```

### Cell 20 code

```
texts, attention_mask = tokenize(tokenizer,texts=["a histopathological image of prostate cancer"]) # Tokenize with custom tokenizer
texts = torch.from_numpy(np.array(texts))#.to(device)
attention_mask = torch.from_numpy(np.array(attention_mask)).to(device)
```

### Cell 21 code

```
texts, attention_mask = tokenize(tokenizer,texts=["a histopathological image of adipose tissue"]) # Tokenize with custom tokenizer
texts = torch.from_numpy(np.array(texts))#.to(device)
attention_mask = torch.from_numpy(np.array(attention_mask)).to(device)
```

### Cell 22 code

```
texts, attention_mask = tokenize(tokenizer,texts=["adipose"]) # Tokenize with custom tokenizer
texts = torch.from_numpy(np.array(texts))#.to(device)
attention_mask = torch.from_numpy(np.array(attention_mask)).to(device)
```

### Cell 23 code

```
texts, attention_mask = tokenize(tokenizer,texts=["stromal"]) # Tokenize with custom tokenizer
texts = torch.from_numpy(np.array(texts))#.to(device)
attention_mask = torch.from_numpy(np.array(attention_mask)).to(device)
```

### Cell 26 code

```
import torch.nn as nn
```

## 20pct rank 202 quality strong score 55

File: `Medical/prostate-cancer-grade-assessment/rank202_20pct/panda-stain-norm-downsample12x64x64-inference.ipynb`
Cells: 20 total, 15 code, 5 markdown

### Cell 5 code

```
SAMPLE = '../input/prostate-cancer-grade-assessment/sample_submission.csv'
```

### Cell 6 markdown

```
# Model
```

### Cell 7 code

```
    model = ResNet(block, layers, **kwargs)
        #m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
```

### Cell 8 code

```
    state_dict = torch.load(path,map_location=torch.device('cpu'))
```

### Cell 10 code

```
def get_tissue_mask(I, luminosity_threshold=0.8):
    mask = L < luminosity_threshold
def get_stain_matrix(I, luminosity_threshold=0.8, angular_percentile=99):
    tissue_mask = get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,))
    # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
```

### Cell 14 code

```
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
#mean = torch.tensor([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304])
#std = torch.tensor([0.36357649, 0.49984502, 0.40477625])
mean = torch.tensor([1.0-0.90008685, 1.0-0.80557228, 1.0-0.88988811])
std = torch.tensor([0.3247047, 0.41417728, 0.33721329])
        self.names = list(pd.read_csv(test).image_id)
        #tiles = torch.Tensor(1.0 - TMP/255.0)
        tiles = torch.Tensor(1.0 - img_resized/255.0)
```

### Cell 18 code

```
sub_df = pd.read_csv(SAMPLE)
    with torch.no_grad():
            #dihedral TTA
            x = torch.stack([x,x.flip(-1),x.flip(-2),x.flip(-1,-2),
            p = torch.stack(p,1)
    probs = torch.cat(probs).numpy()
    preds = torch.cat(preds).numpy()
    sub_df.to_csv('submission.csv', index=False)
    #with torch.no_grad():
    #        #dihedral TTA
    #        x = torch.stack([x,x.flip(-1),x.flip(-2),x.flip(-1,-2),
    #        p = torch.stack(p,1)
    #probs = torch.cat(probs).numpy()
    #preds = torch.cat(preds).numpy()
    #sub_df.to_csv('submission.csv', index=False)
```

### Cell 19 code

```
sub_df.to_csv("submission.csv", index=False)
```

## 10pct rank 103 quality usable score 37

File: `Medical/prostate-cancer-grade-assessment/rank103_10pct/tile-factory.ipynb`
Cells: 15 total, 12 code, 3 markdown

### Cell 0 markdown

```
It aggregates several public tile strategies and provides pipeline for finding optimal SIZE, COLS, ROWS params form the human perspective. It also shows how to save the results.

derives from
* https://www.kaggle.com/iafoss/panda-16x128x128-tiles
* https://www.kaggle.com/akensert/panda-optimized-tiling-tf-data-dataset
* https://www.kaggle.com/debanga/let-s-enhance-the-images
* https://www.kaggle.com/harupy/visualization-panda-16x128x128-tiles
* https://www.kaggle.com/raghaw/panda-medium-resolution-dataset-25x256x256

output dataset examples
* 6x6 256 lafoss_tiles https://www.kaggle.com/dlarionov/prostate-cancer-tiles-medium
* 4x4 312 akensert_tiles https://www.kaggle.com/dlarionov/akensert-1-312-4x4 (this kernel result)
```

### Cell 2 code

```
df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
```

### Cell 3 code

```
    idxs_selected = np.argsort(sums)[:N]
    merged = join_tiles(selected)
    return merged
    # merge tiles to one image
    merged = join_tiles(selected)
        return merged, img
        return merged
```

### Cell 4 code

```
    img_enhanced = cv2.addWeighted(image, contrast, image, 0, brightness)
    return cv2.addWeighted(img, 1.8, img_gaussian, -0.8, 0, img)
def _mask_tissue(image, kernel_size=(7, 7), gray_threshold=220):
    mask = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)
    shift = excess // 2
        i * patch_size + start - shift
        min_decimal_keep : Threshold for decimal point for removing "excessive" patch
```

## 1st rank 6 quality strong score 46

File: `Medical/prostate-cancer-grade-assessment/rank6_1st/panda-submit-clas-twodl.ipynb`
Cells: 3 total, 3 code, 0 markdown

### Cell 0 code

```
!pip install /kaggle/input/pytorchimagemodels120
!pip install /kaggle/input/efficientnet
#!pip install /kaggle/input/pytorch-lightning-075/pytorch_lightning-0.7.5-py3-none-any.whl
!pip install /kaggle/input/pytorch-lightning-081-whl/pytorch_lightning-0.8.1-py3-none-any.whl
!pip install /kaggle/input/albumentations
!mkdir -p /root/.cache/torch/checkpoints/
!cp /kaggle/input/efficientnetb0355c32eb/efficientnet-b0-355c32eb.pth /root/.cache/torch/checkpoints
!cp /kaggle/input/my-efficientnetb008094119/my_efficientnet-b0-08094119.pth /root/.cache/torch/checkpoints
```

### Cell 2 code

```
import torch
from torch.utils.data import DataLoader as DataLoaderPytorch
# This together with tta=6 resulted in error
#df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
df_train = pd.read_csv(data_dir / 'train.csv')
df_test = pd.read_csv(data_dir / 'test.csv')
image_folder = data_dir / 'test_images'
is_test = os.path.exists(image_folder)  # IF test_images is not exists, we will use some train images.
image_folder = image_folder if is_test else data_dir / 'train_images'
# Works with tta_nr=1
# Works with tta_nr=4
# Works with tta_nr=1
tta_nr = 1
#tta_nr = 4
#tta_nr = 6  # might need too much RAM
device = torch.device('cuda')
if tta_nr > 1:
    # Use color augmentations in TTA
dataset_256 = TileDataset(image_folder, df, 256, 256, 144, tile_mode=tile_mode,
                      inference=True, input_type=input_type, overwrite_image_folder=False, tta_nr=tta_nr,
loader_256 = DataLoaderPytorch(dataset_256, batch_size=batch_size, num_workers=num_workers, shuffle=False)
dataset_512 = TileDataset(image_folder, df, 512, 512, 36, tile_mode=tile_mode,
                      inference=True, input_type=input_type, overwrite_image_folder=False, tta_nr=tta_nr,
loader_512 = DataLoaderPytorch(dataset_512, batch_size=batch_size, num_workers=num_workers, shuffle=False)
with torch.no_grad():
            for tta_idx in range(tta_nr):
                if tta_nr > 1:
                    data_tta = data["data"][tta_idx].to(device)
                    data_tta = data["data"].to(device)
                data_raw = model(data_tta)
            for tta_idx in range(tta_nr):
                if tta_nr > 1:
                    data_tta = data["data"][tta_idx].to(device)
                    data_tta = data["data"].to(device)
                data_raw = model(data_tta)
        logits_batch_ens = n
...[truncated]
```
