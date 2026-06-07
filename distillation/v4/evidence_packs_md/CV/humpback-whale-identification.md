# CV / humpback-whale-identification

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1473 quality strong score 59

File: `CV/humpback-whale-identification/rank1473_70pct/fishing-whales.ipynb`
Cells: 78 total, 78 code, 0 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras import models,callbacks,layers,utils,optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet50'''
from fastai.metrics import error_rate
```

### Cell 1 code

```
df=pd.read_csv('../input/train.csv')
```

### Cell 3 code

```
df['total_count'] = df.groupby('Id')['Id'].transform('count')
```

### Cell 6 code

```
df_grouped = df.groupby('Id').apply(lambda x : x.sample(frac=0.2,random_state=22))
```

### Cell 9 code

```
df_merged = pd.merge(left=df,right=df_grouped,how='left',on='Image',suffixes=('','_y'))
```

### Cell 10 code

```
df_merged['is_valid'] = df_merged.Id_y.isnull() != True
df_merged.tail(10)
```

### Cell 11 code

```
df_merged.drop(columns=['Id_y','total_count_y'],inplace=True)
df_merged.tail(10)
```

### Cell 13 code

```
data_src=(ImageItemList.from_df(df=df_merged,path='../input/',folder='train')
       .split_from_df(col='is_valid')
       #How to split in train/valid?
```

### Cell 14 code

```
       .add_test_folder(test_folder='../input/test/')
       #Data augmentation?
```

### Cell 17 code

```
learn = create_cnn(data=data,arch=models.resnet50,metrics=error_rate,model_dir='.',path='../working/tmp/',pretrained=True)
#learn = create_cnn(data=data,arch=models.resnet18,metrics=error_rate,model_dir='.',callback_fns=ShowGraph)
```

### Cell 26 code

```
#        .add_test_folder(test_folder='../input/test/')
#        #Data augmentation?
```

### Cell 33 code

```
len(data.valid_ds)==len(losses)==len(idxs)
```

## 40pct rank 774 quality strong score 67

File: `CV/humpback-whale-identification/rank774_40pct/triplet-loss-from-pretrained-siamese-net-0-76.ipynb`
Cells: 15 total, 8 code, 7 markdown

### Cell 0 markdown

```
- [Whale Recognition Model with score 0.78563](https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563)
- [tensorflow-triplet-loss](https://github.com/omoindrot/tensorflow-triplet-loss)
```

### Cell 2 code

```
import tensorflow
tensorflow.__version__
```

### Cell 4 code

```
import tensorflow as tf
from tensorflow.python.keras import Input, regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Activation, Add, BatchNormalization, Conv2D, GlobalMaxPooling2D, \
from tensorflow.python.keras.models import Model
from tensorflow.python.ops import control_flow_ops
    show_embedding, keras_convert_model_to_estimator_ckpt, InitFromPretrainedCheckpointHook, \
    return control_flow_ops.merge([
    Distorting images provides a useful technique for augmenting the data
        # todo 2.shift
    Distorting images provides a useful technique for augmenting the data
```

### Cell 6 code

```
                "E:/frkhit/Download/AI/pre-trained-model/resnet_v2_50.ckpt"
        return os.path.join(self.working_path, "resnet_v2_50.ckpt")
    def download_resnet_v2_50(self):
            tar_file = "resnet_v2_50_2017_04_14.tar.gz"
            download_big_file("http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz",
        whales = pd.read_csv(raw_path_manager.train_csv)
        whales = pd.read_csv(self.path_manager.train_csv)
        boxes = pd.read_csv(self.path_manager.bounding_boxes_csv)
```

### Cell 8 code

```
class WhaleRankingUtils(object):
    def simple_rank(self, result_list: list, distance_cutoff: float = None, only_distance_fit: bool = False) -> list:
            rank_list: list, like, [class_id_1, class_id_2, ... ]
    def get_similar_cutoff(self, validating_labels: list, validating_feature_list: list) -> float:
        for i, class_id in enumerate(validating_labels):
            for (_distance, _class_id, _file_name) in validating_feature_list[i]:
    def analyze_validating_result(self, feature_file: str):
        class_id_result_list = [self.simple_rank(feature)[:self.top_k] for feature in feature_list]
        self.logger.info("validating data result is: map@5 is {},  map@1 is {}!".format(total_map_5, total_map_1))
            "validating data result with cutoff is: map@5 is {},  map@1 is {}!".format(total_map_5, total_map_1))
        submission = pd.read_csv(csv_file)
            _new_result[file_id] = self.simple_rank(
            _new_result[file_id] = self.simple_rank(
    def _calc_weight_score(feature_list, file_id_list, cutoff) -> dict:
        # todo simple score
        cutoff_score = 5.0
                        _new_result[file_id][class_id] += cutoff_score
    def _get_result_by_weight_score(score_dict: dict, top_k: int = 5) -> dict:
        for image_id, score in score_dict.items():
            result_dict[image_id] = _sort_by_value(score)[:top_k]
    def combine_submission_file(self, validating_file_list: list, submission_file_list: list, output_file: str):
        def _update_weight_score(all_dict: dict, one_dict: dict, model_weight: float):
            for _image_id, _score_dict in one_dict.items():
                for _class_id, _score in _score_dict.items():
                    all_dict[_image_id][_class_id] += _score * model_weight
        assert 
...[truncated]
```

### Cell 10 code

```
class TripletLossModelCNN(AbstractEstimator):
        super(TripletLossModelCNN, self).__init__(
            feature = np.vstack(feature_list).reshape((len(feature_list), self.params.dimension))
        file_id_list, feature_list = WhaleRankingUtils.get_feature_list(feature_file=tmp_feature_list[0])
        for score in result_list:
            feature_file_list.append(byte_to_string(score[self._features_filename_key]))
            feature_list.append(score[self._features_embedding_key].reshape(1, self.params.dimension))
    def map_search(self, only_validate: bool = False, clear_cache_after_exists: bool = True,
                map_5, map_1 = search.validate(data_percent=1.0)
                if only_validate:
                    self.logger.info("success to validate with {}, time cost {} seconds!".format(
        self.logger.info("success to merge all submission file into sub_ens.csv")
            self.map_search(only_validate=False, clear_cache_after_exists=True)
class TripletLossModelSiamese(TripletLossModelCNN):
                 pretrained_ckpt_file: str = None, keras_model: str = None):
            pretrained_ckpt_file=None if keras_model else pretrained_ckpt_file,
        self.keras_model_file = keras_model
        if not self._checkpoint_exist and keras_model:
            ckpt_file = keras_convert_model_to_estimator_ckpt(
                keras_model_path=keras_model, log_dir=self.TRAIN_CKPT_DIR, logger=self.logger)
                self.logger.warning("fail to convert keras model to estimator!")
```

### Cell 11 markdown

```
# 6.training model
```

### Cell 12 code

```
!mkdir -p ./keras
!cp ../input/piotte/mpiotte-standard.model ./keras/
#         keras_model="./keras/mpiotte-standard.model",
        keras_model=None,
estimator.map_search(only_validate=False, clear_cache_after_exists=True)
```

### Cell 14 code

```
!rm ./keras/ -R
```

## 20pct rank 461 quality usable score 33

File: `CV/humpback-whale-identification/rank461_20pct/esemble-lb-0-842.ipynb`
Cells: 5 total, 4 code, 1 markdown

### Cell 0 markdown

```
2. Ref: https://www.kaggle.com/ateplyuk/ensemble-lb-0-833
3. Ref: https://www.kaggle.com/axel81/siamese-ensemble-of-ensemble-lb-0-824
6. Ref: https://www.kaggle.com/monuwio/ensemble-of-many-good-submissions
```

### Cell 3 code

```
                 "../input/ensemble-0842/sub_simi_800.csv",
                 "../input/ensemble-0842/sub_simi_805.csv",
                 "../input/ensemble-0842/sub_ens_833.csv",
                 "../input/ensemble-0842/sub_ens_824.csv",
                 "../input/ensemble-0842/sub_tri_760.csv",
                 "../input/ensemble-0842/sub_siam_822.csv",
```

## 10pct rank 316 quality weak score 17

File: `CV/humpback-whale-identification/rank316_10pct/whale-ensemble-lb-0-866.ipynb`
Cells: 4 total, 4 code, 0 markdown

### Cell 2 code

```
                "../input/ensemble-whale/whale1.csv",
                "../input/ensemble-whale/whale2.csv",
               # "../input/ensemble-whale/whale3.csv",
                #"../input/ensemble-whale/whale4.csv",
               # "../input/ensemble-whale/whale5.csv",
                "../input/ensemble-whale/whale10.csv",
```

### Cell 3 code

```
out = open("whale_ensemble_3.csv", "w", newline='')
```

## 1st rank 4 quality strong score 58

File: `CV/humpback-whale-identification/rank4_1st/visualisation-of-siamese-net.ipynb`
Cells: 10 total, 5 code, 5 markdown

### Cell 2 markdown

```
**Create model and load weights**
```

### Cell 3 code

```
    from keras.applications.densenet import preprocess_input
    from keras.applications.densenet import DenseNet121
    from keras import backend as K
    from keras.optimizers import Adam
    from keras.engine.topology import Input
    from keras.layers import Concatenate, Conv2D, Dense, Flatten, Lambda, Reshape
    from keras.models import Model, load_model
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid', name='hm_conv_2d_1')(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid', name='hm_conv_2d_2')(x)
    # Weighted sum implemented as a Dense layer.
    x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
```

### Cell 6 markdown

```
**Create activation maps**

It's main functions. We got tensor from last BatchNormalization layer (also we can use last activation layer). It has shape like (10, 16, 16, 1024) or (16, 16, 1024) for single image in batch. On the next GlobalMaxPooling layer it converted to vector with 1024 elements in it. To generate N-th element of vector maximum value is taken from (16 x 16) matrix. So pixel located in position with maximum value define the vector value. This vector next goes to head model, which compare whale parameters. So we need to find places on (16x16) matrix where maximum value appear often. Also it's interesting to find places where values change often from one feature map to next feature map. We can use value of standard deviation to mark such places.

So colors on generated images:
* Pink - places where value of max appears often
* Green - places where Std is large
* White - Pink + Green
```

### Cell 7 code

```
        ch = np.stack((ch0, ch1, ch2), axis=2)
    from keras.models import Model
    image_ids = pd.read_csv('../input/humpback-whale-identification/sample_submission.csv')['Image'].values[start:end]
```
