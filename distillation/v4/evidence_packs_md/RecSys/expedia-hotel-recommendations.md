# RecSys / expedia-hotel-recommendations

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 1378 quality usable score 36

File: `RecSys/expedia-hotel-recommendations/rank1378_70pct/ehr-1-with-comments.py`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
    # Increase score of a hotel cluster, giving the key (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
    # Increase score of a hotel cluster, giving the key (user_id, srch_destination_id, hotel_country, hotel_market)
    # Increase score of a hotel cluster, giving the key (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
    # Increase score of a hotel cluster, giving the key (user_location_city, orig_destination_distance)
    # Increase score of a hotel cluster, giving the key (srch_destination_id,hotel_country,hotel_market)
    # Increase score of a hotel cluster, giving the key (hotel_country)
    # Increase the overall score of a hotel cluster
    This function will produce a guess from a dictionary of scores, filling 'filled' if the guess is not present on it
    :param d: dictionary where keys are hotel_cluster and values are scores
```

## 40pct rank 762 quality usable score 27

File: `RecSys/expedia-hotel-recommendations/rank762_40pct/map-k-demo.ipynb`
Cells: 6 total, 4 code, 2 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ml_metrics as metrics
```

### Cell 2 code

```
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
```

### Cell 3 code

```
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
```

### Cell 4 markdown

```
As you see from the above example, the "earlier" you predict the correct answer 1, the higher your score.
```

### Cell 5 code

```
metrics.mapk([[1],[1],[1],[1],[1]],[[1,2,3,4,5],[2,1,3,4,5],[3,2,1,4,5],[4,2,3,1,5],[4,2,3,5,1]], 5)
```

## 20pct rank 391 quality weak score 24

File: `RecSys/expedia-hotel-recommendations/rank391_20pct/valium-1.py`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
# coding: utf-8
__author__ = 'Ravi: https://kaggle.com/company'

import datetime
from heapq import nlargest
from operator import itemgetter
import math

def prepare_arrays_match():
    f = open("../input/train.csv", "r")
    f.readline()

    best_hotels_od_ulc = dict()
    best_hotels_uid_miss = dict()
    best_hotels_search_dest = dict()
    best_hotels_country = dict()
    popular_hotel_cluster = dict()
    best_s00 = dict()
    best_s01 = dict()
    total = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 2000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")

        if arr[11] != '':
            book_year = int(arr[11][:4])
            book_month = int(arr[11][5:7])
        else:
            book_year = int(arr[0][:4])
            book_month = int(arr[0][5:7])

        if book_month<1 or book_month>12 or book_year<2012 or book_year>2015:
            #print(book_month)
            #print(book_year)
            #print(line)
            continue

        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        user_id = arr[7]
        is_package = arr[9]
        srch_destination_id = arr[16]
        hotel_country = arr[21]
        hotel_market = arr[22]
        is_booking = float(arr[18])
        hotel_cluster = arr[23]

        append_0 = ((book_year - 2012)*12 + (book_month - 12))
        if not (append_0>0 and append_0<=36):
            #print(book_year)
            #print(book_month)
            print(line)
            #print(append_0)
            continue

        append_1 = pow(math.log(append_0), 1.35) * (-0.1+0.95*pow(append_0, 1.46)) * (3.5*((book_year - 2012)/2) + 17.55*is_booking)
        appe
...[truncated]
```

## 10pct rank 762 quality usable score 27

File: `RecSys/expedia-hotel-recommendations/rank762_10pct/map-k-demo.ipynb`
Cells: 6 total, 4 code, 2 markdown

### Cell 0 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ml_metrics as metrics
```

### Cell 2 code

```
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
```

### Cell 3 code

```
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
print('AP@5 =',metrics.apk(actual,predicted,5) )
```

### Cell 4 markdown

```
As you see from the above example, the "earlier" you predict the correct answer 1, the higher your score.
```

### Cell 5 code

```
metrics.mapk([[1],[1],[1],[1],[1]],[[1,2,3,4,5],[2,1,3,4,5],[3,2,1,4,5],[4,2,3,1,5],[4,2,3,5,1]], 5)
```

## 1st rank 2 quality reject score 9

File: `RecSys/expedia-hotel-recommendations/rank2_1st/last-week-xkcd.py`
Cells: 1 total, 1 code, 0 markdown

### Cell 0 code

```
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [12, 8]
plt.xkcd()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar(range(4), [4, 6, 40, 50], 0.6)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(4) + 0.3)
ax.set_ylim([0, 60])
ax.set_xticklabels(['RESTARTING TRAINING\nJOBS ON EC2',
                    'TRYING TO ADD\nNEW FEATURES',
                    'THINKING ABOUT HOW\nIS IT EVEN POSSIBLE\nTO REACH 0.6 MAP@5??',
                    'REFRESHING THE\nLEADERBOARD'])
plt.ylabel('TIME')
plt.yticks([])
plt.title("MY LAST WEEK IN THE COMPETITION")
plt.show()
plt.savefig('last_week_xkcd.png')
```
