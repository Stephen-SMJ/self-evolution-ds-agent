# Medical / siim-isic-melanoma-classification

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 2311 quality strong score 55

File: `Medical/siim-isic-melanoma-classification/rank2311_70pct/melanoma-using-fastai2.ipynb`
Cells: 28 total, 26 code, 2 markdown

### Cell 0 markdown

```
# This is a beginner kernel to showcase the use of fastai2 to approach this competition
```

### Cell 4 code

```
from fastai2.torch_basics import *
from fastai2.metrics import *
from sklearn.metrics import roc_auc_score
```

### Cell 5 code

```
train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
```

### Cell 16 code

```
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
fold = 0
    fold+= 1
    print('In fold',fold)
    print("TRAIN LENGTH:", len(train_index), "VALIDATION LENGTH:", len(test_index))
    df[f'fold_{fold}_valid']=False
    df.loc[test_index,f'fold_{fold}_valid']=True
```

### Cell 18 code

```
test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
```

### Cell 21 code

```
ss = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
```

### Cell 22 code

```
#roc_auc=skm_to_fastai(roc_auc_score)
metrics = [accuracy,roc_auc]
```

### Cell 23 code

```
def dataloader(fold):
    dls = ImageDataLoaders.from_df(df, fn_col='file_name',label_col='target', valid_col=f'fold_{fold}_valid',path='', folder='/', seed=42,batch_tfms = [*tfm, Normalize.from_stats(*imagenet_stats)],bs=32,num_workers=0)
```

### Cell 24 code

```
fold = 0
for fold in range(3):
    fold+=1
    print('In fold:',fold)
    dls=dataloader(fold)
    learn = cnn_learner(dls,resnet34,metrics=metrics)
    preds, _ = learn.tta(dl=test_dl)
    print('Prediction completed in fold: {}'.format(str(fold)))
```

### Cell 27 code

```
test[['image_name', 'target']].to_csv('Sub.csv', index=False)
```

## 40pct rank 1322 quality strong score 61

File: `Medical/siim-isic-melanoma-classification/rank1322_40pct/siim-predictions-xgboost-and-resnet50.ipynb`
Cells: 63 total, 58 code, 5 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```

### Cell 2 code

```
train=pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")
```

### Cell 14 code

```
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
```

### Cell 26 code

```
import keras
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)
```

### Cell 27 code

```
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.layers import Activation, Convolution2D, Dropout, Conv2D,AveragePooling2D, BatchNormalization,Flatten,GlobalAveragePooling2D
from keras import layers
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
```

### Cell 28 code

```
base_model = ResNet50(weights='imagenet',include_top=False, input_shape=(50,50,3))
```

### Cell 29 code

```
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### Cell 30 code

```
history=model.fit(x_train,y_train,batch_size=128,epochs=20,verbose=1,validation_split=0.33,callbacks=[checkpoint])
```

### Cell 31 code

```
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
```

### Cell 36 code

```
a.to_csv("benign_malignant.csv",index=False)
```

### Cell 38 code

```
from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
```

### Cell 44 code

```
from sklearn.model_selection import train_test_split,KFold,cross_val_score
```

## 20pct rank 755 quality usable score 25

File: `Medical/siim-isic-melanoma-classification/rank755_20pct/melanoma-minmax.ipynb`
Cells: 5 total, 4 code, 1 markdown

### Cell 0 markdown

```
https://www.kaggle.com/solomonk/minmax-ensemble-0-9526-lb?rvi=1
https://www.kaggle.com/truonghoang/stacking-ensemble-on-my-submissions
https://www.kaggle.com/datafan07/analysis-of-melanoma-metadata-and-effnet-ensemble
```

### Cell 1 code

```
def MinMaxBestBaseStacking(input_folder, best_base, output_path):
    sub_base = pd.read_csv(best_base)
    all_files = os.listdir(input_folder)
    outs = [pd.read_csv(os.path.join(input_folder, f), index_col=0) for f in all_files]
    # get the data fields ready for stacking
    # set up cutoff threshold for lower and upper bounds
    concat_sub[['image_name', 'target']].to_csv(output_path,
```

### Cell 2 code

```
MinMaxBestBaseStacking('../input/cs0099/', '../input/cs0099/submission.csv', 'submission_6.csv')
# data1 = pd.read_csv('../input/minmax-ensemble-0-9526-lb/submission.csv')
# data2 = pd.read_csv('../input/stacking-ensemble-on-my-submissions/submission_mean.csv')
```

### Cell 4 code

```
# submission.to_csv('submission.csv', index=False, float_format='%.6f')
```

## 10pct rank 335 quality usable score 37

File: `Medical/siim-isic-melanoma-classification/rank335_10pct/chris-submission-oof-weighted-ensemble-prlb-0-9427.ipynb`
Cells: 9 total, 9 code, 0 markdown

### Cell 0 code

```
## reference: https://www.kaggle.com/ipythonx/optimizing-metrics-out-of-fold-weights-ensemble
```

### Cell 1 code

```
!ls ../input/melanoma-oof-and-sub
```

### Cell 2 code

```
from sklearn.metrics import roc_auc_score
```

### Cell 4 code

```
oof_01  = pd.read_csv('../input/melanoma-oof-and-sub/oof_0.csv')
test_01 = pd.read_csv('../input/melanoma-oof-and-sub/sub_0.csv')
oof_01  = oof_01.sort_values(by=['image_name'],
oof_02  = pd.read_csv('../input/melanoma-oof-and-sub/oof_100.csv')
test_02 = pd.read_csv('../input/melanoma-oof-and-sub/sub_100.csv')
oof_02  = oof_02.sort_values(by=['image_name'],
oof_03  = pd.read_csv('../input/melanoma-oof-and-sub/oof_105.csv')
test_03 = pd.read_csv('../input/melanoma-oof-and-sub/sub_105.csv')
oof_03  = oof_03.sort_values(by=['image_name'],
oof_04  = pd.read_csv('../input/melanoma-oof-and-sub/oof_108.csv')
test_04 = pd.read_csv('../input/melanoma-oof-and-sub/sub_108.csv')
oof_04  = oof_04.sort_values(by=['image_name'],
oof_05  = pd.read_csv('../input/melanoma-oof-and-sub/oof_109.csv')
test_05 = pd.read_csv('../input/melanoma-oof-and-sub/sub_109.csv')
oof_05  = oof_05.sort_values(by=['image_name'],
oof_06  = pd.read_csv('../input/melanoma-oof-and-sub/oof_11.csv')
test_06 = pd.read_csv('../input/melanoma-oof-and-sub/sub_11.csv')
oof_06  = oof_06.sort_values(by=['image_name'],
oof_07  = pd.read_csv('../input/melanoma-oof-and-sub/oof_110.csv')
test_07 = pd.read_csv('../input/melanoma-oof-and-sub/sub_110.csv')
oof_07  = oof_07.sort_values(by=['image_name'],
oof_08  = pd.read_csv('../input/melanoma-oof-and-sub/oof_111.csv')
test_08 = pd.read_csv('../input/melanoma-oof-and-sub/sub_111.csv')
oof_08  = oof_08.sort_values(by=['image_name'],
oof_09  = pd.read_csv('../input/melanoma-oof-and-sub/oof_113.csv')
test_09 = pd.read_csv('../input/melanoma-oof-and-sub/sub_113.csv')
oof_09  = oof_09.sort_values(by=['image_name'],
oof_10  = pd.read_csv('../input/melanoma-oof-and-sub/oof_116.csv')
test_10 = pd.read_csv('../input/melanoma-oof-and-sub/sub_116.csv')
oof_10  = oof_10.sort_values(by=['
...[truncated]
```

### Cell 5 code

```
blend_train = []
blend_test = []
# out of fold prediction
blend_train.append(oof_01.pred)
blend_train.append(oof_02.pred)
blend_train.append(oof_03.pred)
blend_train.append(oof_04.pred)
blend_train.append(oof_05.pred)
blend_train.append(oof_06.pred)
blend_train.append(oof_07.pred)
blend_train.append(oof_08.pred)
blend_train.append(oof_09.pred)
blend_train.append(oof_10.pred)
blend_train.append(oof_11.pred)
blend_train.append(oof_12.pred)
blend_train.append(oof_13.pred)
blend_train.append(oof_14.pred)
blend_train.append(oof_15.pred)
blend_train.append(oof_16.pred)
blend_train.append(oof_17.pred)
blend_train.append(oof_18.pred)
blend_train.append(oof_19.pred)
blend_train.append(oof_20.pred)
blend_train.append(oof_21.pred)
blend_train.append(oof_22.pred)
blend_train.append(oof_23.pred)
blend_train.append(oof_24.pred)
blend_train.append(oof_25.pred)
blend_train.append(oof_26.pred)
blend_train.append(oof_27.pred)
blend_train.append(oof_28.pred)
blend_train.append(oof_29.pred)
blend_train.append(oof_30.pred)
blend_train.append(oof_31.pred)
blend_train.append(oof_32.pred)
blend_train.append(oof_33.pred)
blend_train.append(oof_34.pred)
blend_train.append(oof_35.pred)
blend_train.append(oof_36.pred)
blend_train.append(oof_37.pred)
blend_train.append(oof_38.pred)
blend_train.append(oof_39.pred)
blend_train = np.array(blend_train)
# submission scores
blend_test.append(test_01.target)
blend_test.append(test_02.target)
blend_test.append(test_03.target)
blend_test.append(test_04.target)
blend_test.append(test_05.target)
blend_test.append(test_06.target)
blend_test.append(test_07.target)
blend_test.append(test_08.target)
blend_test.append(test_09.target)
blend_test.append(test_10.target)
blend_test.append(test_11.target)
blend_test.append(test_12.target)
blend_test.append(test_13.targ
...[truncated]
```

### Cell 6 code

```
    for weight, prediction in zip(weights, blend_train):
    return roc_auc_score(np.array(oof_01.target), final_prediction)
print('\n Finding Blending Weights ...')
    #starting_values = np.random.uniform(size=len(blend_train))
    starting_values = np.random.uniform(size=len(blend_train)) * 1/len(blend_train)
    #bounds = [(0, 1)] * len(blend_train)
    #bounds = [(0, 1/len(blend_train))] * len(blend_train)
    bounds = [(0, 1/len(blend_train)) for _ in range(len(blend_train))]
    print('{iter}\tScore: {score}\tWeights: {weights}'.format(
        score=res['fun'],
blend_score = round(bestSC, 6)
```

### Cell 7 code

```
print('\n Ensemble Score: {best_score}'.format(best_score=bestSC))
train_prices = np.zeros(len(blend_train[0]))
test_prices  = np.zeros(len(blend_test[0]))
for k in range(len(blend_test)):
    test_prices += blend_test[k] * weights[k]
for k in range(len(blend_train)):
    train_prices += blend_train[k] * weights[k]
```

### Cell 8 code

```
test_01.to_csv('final_weighted_average_ensemble.csv', index=False)
```

## 1st rank 755 quality usable score 30

File: `Medical/siim-isic-melanoma-classification/rank755_1st/melanoma-minmax.ipynb`
Cells: 5 total, 4 code, 1 markdown

### Cell 0 markdown

```
https://www.kaggle.com/solomonk/minmax-ensemble-0-9526-lb?rvi=1
https://www.kaggle.com/truonghoang/stacking-ensemble-on-my-submissions
https://www.kaggle.com/datafan07/analysis-of-melanoma-metadata-and-effnet-ensemble
```

### Cell 1 code

```
def MinMaxBestBaseStacking(input_folder, best_base, output_path):
    sub_base = pd.read_csv(best_base)
    all_files = os.listdir(input_folder)
    outs = [pd.read_csv(os.path.join(input_folder, f), index_col=0) for f in all_files]
    # get the data fields ready for stacking
    # set up cutoff threshold for lower and upper bounds
    concat_sub[['image_name', 'target']].to_csv(output_path,
```

### Cell 2 code

```
MinMaxBestBaseStacking('../input/cs0099/', '../input/cs0099/submission.csv', 'submission_6.csv')
# data1 = pd.read_csv('../input/minmax-ensemble-0-9526-lb/submission.csv')
# data2 = pd.read_csv('../input/stacking-ensemble-on-my-submissions/submission_mean.csv')
```

### Cell 4 code

```
# submission.to_csv('submission.csv', index=False, float_format='%.6f')
```
