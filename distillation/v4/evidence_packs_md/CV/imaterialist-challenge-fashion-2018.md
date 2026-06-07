# CV / imaterialist-challenge-fashion-2018

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 171 quality reject score -25

File: `CV/imaterialist-challenge-fashion-2018/rank171_70pct/download-datasets-to-your-google-drive.ipynb`
Cells: 15 total, 9 code, 6 markdown

### Cell 1 markdown

```

###Features:
1.   Smaller version of images can be found by replacing "-large" with "-small" at the end of url. In this code, the small picture (around 5 kb on average) are downloaded.
2.   This script has been built for Colab users. As the instance gets destroyed every 12 hours, one can't download datasets everytime.
3.   Also, if you have a locally available GPU and want to download images to your drive (and then to your computer on one click), you can use this script.
4.   This notebook can be used to download data to drive for any competition that gives urls in JSON files (ofcourse with little modifications).

```

### Cell 10 code

```
!mkdir drive/kaggle/train drive/kaggle/validation drive/kaggle/test
```

### Cell 12 code

```
#To download validation data on to your drive...
val_data = json.load(open('drive/kaggle/validation.json'))
!echo ImageURL, ImgName, ImgId, LabelId >> drive/kaggle/validation/validation.txt
  !curl $img_url_small > drive/kaggle/validation/$img_name_small
  !echo $img_name_actual,$img_id,$label_id >> drive/kaggle/validation/validation.txt
```

### Cell 14 markdown

```
###Note

1.   This is it for now. Later, I shall include EDA and hopefully the actual CV architecture part! But boy, those data sets are too hot to handle :P

2.   And yes, too large too! Anyway, it seems really fun to play with this dataset! Good luck to everyone!!

**Upvote this kernel if you find it useful so that others can find it easily.**
```

## 40pct rank 84 quality reject score -20

File: `CV/imaterialist-challenge-fashion-2018/rank84_40pct/simplest-multithreading-downloader.py`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
exitFlag = 0
        if exitFlag:
```

## 20pct rank 45 quality reject score -8

File: `CV/imaterialist-challenge-fashion-2018/rank45_20pct/download-image-progress-resume-multiprocessing.py`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
  if 'train' in data_file or 'validation' in data_file:
    print('Syntax: %s <train|validation|test.json> <output_dir/>' % sys.argv[0])
```

## 10pct rank 3 quality reject score 4

File: `CV/imaterialist-challenge-fashion-2018/rank3_10pct/download-images-using-r.R`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
  df <- merge(df_1, df_2, by = 'imageId', all = TRUE)
```

## 1st rank 3 quality reject score 9

File: `CV/imaterialist-challenge-fashion-2018/rank3_1st/download-images-using-r.R`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
  df <- merge(df_1, df_2, by = 'imageId', all = TRUE)
```
