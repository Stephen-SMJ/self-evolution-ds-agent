# Medical / rsna-breast-cancer-detection

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1184 quality strong score 49

File: `Medical/rsna-breast-cancer-detection/rank1184_70pct/rsna-breast-baseline-inference.ipynb`
Cells: 25 total, 17 code, 8 markdown

### Cell 7 code

```
SAVE_FOLDER = "/kaggle/tmp/output/"
os.makedirs(SAVE_FOLDER, exist_ok=True)
```

### Cell 8 code

```
def process(f, size=512, save_folder="", extension="png"):
    if dicom.PhotometricInterpretation == "MONOCHROME1":
    cv2.imwrite(save_folder + f"{patient}_{image}.{extension}", (img * 255).astype(np.uint8))
```

### Cell 9 code

```
    delayed(process)(uid, size=SIZE, save_folder=SAVE_FOLDER, extension=EXTENSION)
```

### Cell 13 code

```
df = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/test.csv")
df['path'] = SAVE_FOLDER + df["patient_id"].astype(str) + "_" + df["image_id"].astype(str) + ".png"
```

### Cell 15 code

```
EXP_FOLDERS = [
```

### Cell 17 code

```
import torch
import torch.nn as nn
from torchvision import transforms as T
```

### Cell 18 code

```
from torchvision import models
import torch
import torch.nn as nn
from torchvision import transforms as T
        'm': models.efficientnet_b0,
        'm': models.efficientnet_b0,
    'resnet18': {
        'm': models.resnet18,
    'resnet50': {
        'm': models.resnet50,
        max_value, _ = torch.max(X_mask.view(*shape[:2], -1), dim = -1)
        one = torch.ones_like(X)
        checkpoint = torch.load(model_cf['weight'], map_location=self.device)
        imgs = torch.stack([self.to_tensor(img) for img in imgs]).to(device = self.device)
        with torch.no_grad():
        pred_label = [torch.softmax(output, dim = 1).cpu().tolist() for output in outputs]
```

### Cell 19 code

```
config={'weight': '/kaggle/input/weights/resnet50_cut256_2cls.pt',
            'backbone': 'resnet50',
```

### Cell 23 code

```
sub = df[['prediction_id', 'cancer']].groupby("prediction_id").mean().reset_index()
# sub.to_csv('submission.csv', index=False)
sub.to_csv('/kaggle/working/submission.csv', index=False)
```

## 40pct rank 676 quality strong score 61

File: `Medical/rsna-breast-cancer-detection/rank676_40pct/train-pytorch-lightning-gpu-tpu-w-b-kfolds.ipynb`
Cells: 27 total, 15 code, 12 markdown

### Cell 0 markdown

```
# PyTorch Lightning GPU & TPU Trainer with KFolds and F1 Loss + W&B logging ⚡️
```

### Cell 3 code

```
# ! curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# ! python pytorch-xla-env-setup.py --version 1.7 --apt-packages libomp5 libopenblas-dev
# ! python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
# !pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.7-cp36-cp36m-linux_x86_64.whl
# !pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
```

### Cell 5 code

```
!pip install timm
!pip install albumentations
!pip install --upgrade torchmetrics
!pip install --upgrade pytorch-lightning
```

### Cell 6 code

```
import torch_xla
```

### Cell 7 code

```
import timm
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedGroupKFold
from albumentations import (
from albumentations.pytorch import ToTensorV2
```

### Cell 10 code

```
    'VALID_BS': 16,
    'EVAL_METRIC': 'F1',
```

### Cell 11 markdown

```
We will use their tools to log hyperparameters and output metrics from your runs, then visualize and compare results and quickly share findings with your colleagues.<br><br></p>
```

### Cell 13 code

```
#     project='pytorch_lightning',
```

### Cell 14 code

```
        https://github.com/clcarwin/focal_loss_pytorch
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
    https://www.kaggle.com/code/awsaf49/rsna-bcd-efficientnet-tf-tpu-1vm-train
    preds = preds.clip(0, 1)
```

### Cell 16 code

```
    def __init__(self, df, img_folder, augments=None, is_test=False):
        self.augments = augments
        self.img_folder = img_folder
        img_path = os.path.join(self.img_folder, self.df['img_name'][idx])
        if self.augments:
            img = self.augments(image=img)['image']
        img = torch.tensor(img, dtype=torch.float)
            target = torch.tensor(target, dtype=torch.float)
```

### Cell 18 code

```
class Augments:
    Contains Train, Validation Augments
    train_augments = Compose([
    valid_augments = Compose([
```

### Cell 19 markdown

```
## Model Class using `pl.LightningModule` ⚡️
```

## 20pct rank 333 quality usable score 29

File: `Medical/rsna-breast-cancer-detection/rank333_20pct/rsna-eda-and-nii-files.ipynb`
Cells: 14 total, 8 code, 6 markdown

### Cell 2 code

```
df = pd.read_csv(csv_path)
```

### Cell 4 code

```
output_train_folder = '/kaggle/working/train_imgs/'
if os.path.exists(output_train_folder) == False:
    os.mkdir(output_train_folder)
    folder_name = str(dfvalue[2])+'_'+ str(dfvalue[4])
    #print(folder_name)
    output_folder = '/kaggle/working/train_imgs/'+folder_name
    if os.path.exists(output_folder) == False:
        os.mkdir(output_folder)
    dst = output_folder+'/'+str(dfvalue[2])+'.dcm'
```

### Cell 11 code

```
output_nii_folder = '/kaggle/working/nii_img/'
if os.path.exists(output_nii_folder) == False:
    os.mkdir(output_nii_folder)
        sitk.WriteImage(img, output_nii_folder+image_dir+'.nii.gz')
```

### Cell 12 code

```
for img_folder in os.listdir(main_input_dir):
    print(img_folder)
    dicom2nifti(main_input_dir, img_folder)
```

### Cell 13 code

```
len(os.listdir(output_nii_folder))
```

## 10pct rank 125 quality strong score 65

File: `Medical/rsna-breast-cancer-detection/rank125_10pct/logistic-regression-tabular-data.ipynb`
Cells: 25 total, 20 code, 5 markdown

### Cell 0 code

```
from sklearn.linear_model import LogisticRegression
```

### Cell 2 markdown

```
### Create the feature dataset with meta data features
```

### Cell 3 code

```
        meta_df = pd.read_csv(f'{data_path}/train.csv')
        meta_df = pd.read_csv(f'{data_path}/test.csv')
    valid_views = {'MLO', 'CC'}
    is_valid_view = meta_df['view'].isin(valid_views)
    meta_df = meta_df.loc[is_valid_view, :]
    meta_df = meta_df.groupby('patient_id').agg(list)
```

### Cell 5 markdown

```
### NOTE: Here we would add the training image predictions to the train_df when we have them, as an additional feature.
```

### Cell 8 markdown

```
### NOTE: Here we will get the test set image predictions and add them as an additional feature
```

### Cell 9 code

```
# # create tensorflow datasets
#     model = tf.keras.models.load_model(path)
```

### Cell 12 code

```
clf = LogisticRegression(max_iter=500)
```

### Cell 15 code

```
    preds = preds.clip(0, 1)
```

### Cell 23 code

```
submission_df = test_df.groupby('prediction_id')['cancer'].agg(max)
submission_df.to_csv('submission.csv', index=False)
```

## 1st rank 1 quality strong score 72

File: `Medical/rsna-breast-cancer-detection/rank1_1st/rsna-mammo-rejection-ensemble-v2-ishikei.ipynb`
Cells: 43 total, 34 code, 9 markdown

### Cell 2 code

```
%%writefile /opt/conda/lib/python3.7/site-packages/nvidia/dali/plugin/pytorch.py
import torch
import torch.utils.dlpack as torch_dlpack
to_torch_type = {
    types.DALIDataType.FLOAT:   torch.float32,
    types.DALIDataType.FLOAT64: torch.float64,
    types.DALIDataType.FLOAT16: torch.float16,
    types.DALIDataType.UINT8:   torch.uint8,
    types.DALIDataType.INT8:    torch.int8,
    types.DALIDataType.UINT16:  torch.int16,
    types.DALIDataType.INT16:   torch.int16,
    types.DALIDataType.INT32:   torch.int32,
    types.DALIDataType.INT64:   torch.int64
    Copy contents of DALI tensor to PyTorch's Tensor.
    `arr` : torch.Tensor
    `cuda_stream` : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    dali_type = to_torch_type[dali_tensor.dtype]
                                    " doesn't match the element type of the target PyTorch Tensor: "
        ("Shapes do not match: DALI tensor has size {0}, but PyTorch Tensor has size {1}".
    General DALI iterator for PyTorch. It can return any number of
    outputs from the DALI pipeline in the form of PyTorch's Tensors.
                Setting this flag to False will cause the iterator to return
            category_torch_type = dict()
            torch_gpu_device = None
            torch_cpu_device = torch.device('cpu')
                category_torch_type[category] = to_torch_type[category_tensors[category].dtype]
                    if not torch_gpu_device:
                        torch_gpu_device = torch.device('cuda', dev_id)
                    category_device[category] = torch_gpu_device
                    cate
...[truncated]
```

### Cell 6 code

```
J2K_FOLDER = Path("/tmp/j2k/")
```

### Cell 7 code

```
import torch
import torch.nn.functional as F
from nvidia.dali.plugin.pytorch import feed_ndarray, to_torch_type
def convert_dicom_to_j2k(file, save_folder):
        with open(save_folder/f"{patient}_{image}.jp2", "wb") as binary_file:
        with open(save_folder/f"{patient}_{image}.jp2", "wb") as binary_file:
    if dataset.PhotometricInterpretation == "MONOCHROME1":
    if dataset["PhotometricInterpretation"] == "MONOCHROME1":
def torch_voi_lut(img, dataset):
        img = y_range / (1 + torch.exp(-4 * (img - center) / width)) + y_min
        between = torch.logical_and(~below, ~above)
```

### Cell 8 code

```
    J2K_FOLDER.mkdir(exist_ok=True)
        delayed(convert_dicom_to_j2k)(img, save_folder=J2K_FOLDER)
    j2kfiles = list(J2K_FOLDER.glob("*.jp2"))
            # Dali -> Torch
            img_torch = torch.empty(img.shape(), dtype=torch.int16, device="cuda")
            feed_ndarray(img, img_torch, cuda_stream=torch.cuda.current_stream(device=0))
            img = img_torch.float()
            img = torch_voi_lut(img, dicom)
            if dicom.PhotometricInterpretation == "MONOCHROME1":
    shutil.rmtree(J2K_FOLDER)
```

### Cell 11 code

```
def process(f, size=2048, save_folder=""):
    cv2.imwrite(str(save_folder/f"{patient}_{image}.png"), (img * 255).astype(np.uint8))
```

### Cell 12 code

```
    delayed(process)(img, size=SIZE, save_folder=EXPORT_DIR)
```

### Cell 14 code

```
!ln -s ../input/rsna-mammo-2023/dataset/timm/
!ln -s ../input/rsna-mammo-2023/dataset/segmentation_models_pytorch/
from timm.layers import convert_sync_batchnorm
from metrics import Pfbeta, PercentilePfbeta
from torch.cuda import amp
```

### Cell 18 code

```
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import timm
from timm.models.layers import DropPath, trunc_normal_
from timm.models import register_model
from torch import nn
def merge_pre_bn(module, pre_bn_1, pre_bn_2=None):
    """ Merge pre BN to reduce inference runtime.
        zeros = torch.zeros(module.out_channels, device=weight.device).type(weight.type())
    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)
        self.is_bn_merged = False
    def merge_bn(self):
        if not self.is_bn_merged:
            self.mlp.merge_bn(self.norm)
            self.is_bn_merged = True
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
        self.is_bn_merged = False
    def merge_bn(self, pre_bn):
        merge_pre_bn(self.q, pre_bn)
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merged = True
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
        self.is_bn_merged = False
    def merge_bn(self):
        if not self.is_bn_merged:
            self.e_mhsa.merge_bn(self.norm1)
            self.mlp.merge_bn(self.norm2)
            self.is_bn_merged = True
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
        x = torch.cat([x, out], dim=1)
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]  # stochastic depth decay rule
    def merge_bn(self):
                module.merge_bn()
        x = torch.flatten(x, 1)
        # self.register_buffer('mean', torch.FloatTensor([0.5, 0.5, 0.5]
...[truncated]
```

### Cell 20 code

```
!ln -s ../input/rsna-mammo-2023/dataset/timm/
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
        forward_features.add_module("relu", torch.nn.PReLU())
    ckpt = torch.load(weight_path, map_location="cpu")
    def get_roi_crop(self, image, threshold=0.1, buffer=30):
        y_slice = np.where(y_mean > threshold)[0]
        x_slice = np.where(x_mean > threshold)[0]
            image, threshold=th, buffer=buffer
        images = torch.concat([self.apply_transform(image) for image in images])
```

### Cell 22 code

```
train = pd.read_csv('../input/rsna-breast-cancer-detection/train.csv')
test = pd.read_csv('../input/rsna-breast-cancer-detection/test.csv')
```

### Cell 24 code

```
    def get_oof_path(self, cfg):
        return Path('../input/rsna-oof/kuma_oofs_v3/')/f'kuma_{cfg.name}.csv'
            test_loader: pytorch dataloader
            pd.DataFrame([{'name': 'dummy', 'xmin': 0, 'ymin': 0, 'xmax':0, 'ymax':0}]).to_csv(
        folds=None,
            folds: list of folds to include
            if isinstance(input_t, torch.Tensor):
            with torch.no_grad():
        if folds is None:
            folds = list(range(cfg.cv))
        for fold in range(cfg.cv):
            if not fold in folds:
            checkpoint = torch.load(proj_dir/f'fold{fold}.pt', 'cpu')
            ys = np.stack(ys, axis=0) # (cv, bs, 1)
        torch.cuda.empty_cache()
        for plr, pdf in df.groupby(['patient_id', 'laterality']):
        agg_predictions = agg_predictions.groupby('prediction_id').agg({
    def inference_block(self, cfg, test_df, batch_size_ratio=1, fp16=True, extend=0, threshold_params={}):
```

### Cell 25 code

```
    def get_oof_path(self, cfg):
        return list(Path('../input/rsna-oof/ariyasu/').glob(f'{cfg.file_name}*.csv'))[0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = timm.create_model(model_name, pretrained=False, num_classes=cfg.num_classes)
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
            torch_state_dict = {}
                torch_state_dict[k[6:]] = v
            model.load_state_dict(torch_state_dict)
            #     model = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input(min_shape=[1, 3, image_size[1], image_size[0]],opt_shape=[cfg.batch_size, 3, image_size[1], image_size[0]],max_shape=[cfg.batch_size, 3, image_size[1], image_size[0]],dtype=torch.float32)],
            #         enabled_precisions = torch.float32, # Run with FP32
        torch.cuda.empty_cache()
    def inference_block(self, cfg, test, threshold_params={}, fp16=True):
        assert cfg.tta == 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('tta:', cfg.tta)
        with torch.no_grad():
                ys = np.stack(ys) # (cv, bs, 1)
        del models; torch.cuda.empty_cache()
        test[[f'pred_fold_{fold}' for fold in range(4)]] = preds[:,:,0].T
        for fold in range(4):
            df = pd.DataFrame(test.groupby('prediction_id')[f'pred_fold_{fold}'].mean()).reset_index()
            del test[f'pred_fold_{fold}']
            test = test.merge(df, on='prediction_id')
        predictions = df[[f'pred_fold_{fold}' for fold in range(4)]].values.T
        df[cfg.file_name] = df[[f'pred_fold_{fold}' for fold in range(4)]].mean(1)
        for col in [f'pred_fold_{fold}' for fold in range(4)]:
```
