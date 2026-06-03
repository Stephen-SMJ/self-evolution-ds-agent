# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import os
import time
import cv2
import datetime
import numpy as np
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils


# Input data files are available in the "../input/" directory.
# load input data files
def get_im_cv2(path, img_rows, img_cols):
    img = cv2.imread(path, 0)
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

def load_train(img_rows, img_cols):
    X_train = []
    Y_train = []
    X_train_id = []
    start_time = time.time()
    
    print('Read train images')
    
    for parent,dirnames,filenames in os.walk('../input/train/'):   
        for filename in filenames:
            filepath = os.path.join(parent, filename)
            img = get_im_cv2(filepath, img_rows, img_cols)
            X_train.append(img)
            X_train_id.append(filename[:-4])
            if filename.find('dog') > -1:
                Y_train.append(1.0)
            else:
                Y_train.append(0.0)
    print('Read train data time: {} seconds'.format(
        round(time.time() - start_time, 2)))
    return X_train, Y_train, X_train_id
    
def load_test(img_rows, img_cols):
    X_test = []
    X_test_id = []
    start_time = time.time()
    
    print('Read test images')
    
    for parent,dirnames,filenames in os.walk('../input/test/'):   
        for filename in filenames:
            filepath = os.path.join(parent, filename)
            img = get_im_cv2(filepath, img_rows, img_cols)
            X_test.append(img)
            X_test_id.append(filename[:-4])
    print('Read test data time: {} seconds'.format(
        round(time.time() - start_time, 2)))
    return X_test, X_test_id

def read_and_normalize_train_data(img_rows, img_cols):
    train_data, train_target, train_id = load_train(img_rows, img_cols)
    train_data = np.array(train_data, dtype=np.uint8)
    
    train_target = np.array(train_target, dtype=np.uint8)
    train_target = np_utils.to_categorical(train_target, 2)
    
    train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
    train_data = train_data.astype('float32')
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id

def read_and_normalize_test_data(img_rows, img_cols):
    test_data, test_id = load_test(img_rows, img_cols)
    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
    test_data = test_data.astype('float32')
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id

# Create CNN model
def create_model(img_rows, img_cols):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',
                            input_shape=(img_rows, img_cols, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv
    
def getPredScorePercent(train_target, predictions_valid):
    perc = 0
    for i in range(len(train_target)):
        pred = 1
        if predictions_valid[i][0] > 0.5:
            pred = 0
        real = 1
        if train_target[i][0] > 0.5:
            real = 0
        if real == pred:
            perc += 1
    perc /= len(train_target)
    return perc

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def create_submission(predictions, test_id):
    sub_file = os.path.join('submission_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    subm = open(sub_file, "w")
    subm.write("id,label\n")
    for i in range(len(test_id)):
        subm.write(str(test_id[i]) + ',')
        subm.write(str(predictions[i][1]) + '\n')
    subm.close()

def run():
    img_rows, img_cols = 32, 32
    batch_size = 32
    nb_epoch = 10
    random_state = 51

    X_train, Y_train, train_id = read_and_normalize_train_data(img_rows, img_cols)
    X_test, test_id = read_and_normalize_test_data(img_rows, img_cols)
    
    X_train = X_train
    Y_train = Y_train
    train_id = train_id
    
    model = create_model(img_rows, img_cols)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ]
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=2, validation_split=0.1, callbacks=callbacks)
    test_prediction = model.predict(X_test, batch_size=batch_size, verbose=2)
    create_submission(test_prediction, test_id)

if __name__ == '__main__':
    run()
