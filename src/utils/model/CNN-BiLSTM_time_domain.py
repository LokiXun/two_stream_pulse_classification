"""CNN-BiLSTM in time_domain, created by 陈磊学长"""
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import cycle
from scipy import interp
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers import *
from sklearn.metrics import accuracy_score
from keras.models import *
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import anotherdg2 as dg
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss


def predictFromModel(model, testX):
    testX = testX.astype(float)
    predictions = model.predict(testX)
    # print predictions
    predictions = np.argmax(predictions, axis=1)

    return predictions


def CM(y_true, y_pred):
    F1 = f1_score(y_true, y_pred, average='macro')
    Precision = precision_score(y_true, y_pred, average='macro')
    Recall = recall_score(y_true, y_pred, average='macro')
    ham_distance = hamming_loss(y_true, y_pred)
    return Recall, Precision, F1, ham_distance


def calcAccuracy(testY, p):
    # print the accuracy
    accuracy = sum(testY == p) / float(len(testY))
    return accuracy


def Normalize(group):
    minVals = group.min(0)  # 为0时：求每列的最小值[0 3 1]   .shape=(3,)
    maxVals = group.max(0)  # 为0时：求每列的最大值[2 7 8]   .shape=(3,)
    ranges = maxVals - minVals

    m = group.shape[0]
    normDataSet = np.zeros(np.shape(group))  # np.shape(group) 返回一个和group一样大小的数组，但元素都为0
    diffnormData = group - np.tile(minVals, (m, 1))  # (oldValue-min)  减去最小值
    normDataSet1 = diffnormData / np.tile(ranges, (m, 1))
    return normDataSet1


maxlen = 270
INPUT_DIM = 1
lstm_units = 32
path1 = 'C:/Users/HaveNiceDay/Desktop/608_3/608pulse/pulse_h'
label, data = dg.data_root(path1)

t_all, p_all, data, label = dg.data_test_train(data, label)
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)
cvscores_acc = []
cvscores_rec = []
cvscores_ham = []
cvscores_pre = []
cvscores_F1 = []
cvscores_auc = []
X = np.array(data)
Y = np.array(label)
k = 1
for train, test in kfold.split(X, Y):
    X_train = X[train]
    Y_train = Y[train]
    X_test = X[test]
    Y_test = Y[test]

    encoder = LabelEncoder()
    encoder.fit(Y_train)
    encoded_Y = encoder.transform(Y_train)
    dataY = np_utils.to_categorical(encoded_Y)
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='post')
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='post')
    dataX = np.array(X_train)
    testX = np.array(X_test)
    shape0 = np.shape(dataX)[0]
    shape1 = 36917 - shape0
    dataX = dataX.reshape(shape0, maxlen)
    testX = testX.reshape(shape1, maxlen)
    # dataX=Normalize(dataX)
    # testX=Normalize(testX)
    dataX = dataX.reshape(shape0, maxlen, 1)
    testX = testX.reshape(shape1, maxlen, 1)
    shuffle_vector = np.random.randint(low=0, high=dataX.shape[0], size=dataX.shape[0])
    dataX = dataX[shuffle_vector]
    dataY = dataY[shuffle_vector]

    # 模型训练
    inputs = Input(shape=(maxlen, INPUT_DIM))
    x = Conv1D(32, 3, strides=1, name='conv1', use_bias=False)(inputs)
    x = MaxPooling1D(3, padding="same")(x)
    x = Conv1D(64, 3, strides=1, name='conv2', use_bias=False)(x)
    x = MaxPooling1D(3, padding="same")(x)
    x = Conv1D(128, 3, strides=1, name='conv3', use_bias=False)(x)
    x = MaxPooling1D(3, padding="same")(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(0.1)(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=output)
    opt = Adam(lr=0.005)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # 模型测试
    model.fit(dataX, dataY, epochs=100, batch_size=128, validation_split=0.3)
    test_data = testX
    y_pred = predictFromModel(model, test_data)
    acc = accuracy_score(Y_test, y_pred)
    Recall, Precision, F1, ham = CM(Y_test, y_pred)
    cvscores_rec.append(Recall * 100)
    cvscores_pre.append(Precision * 100)
    cvscores_F1.append(F1 * 100)
    cvscores_ham.append(ham * 100)
    cvscores_acc.append(acc * 100)
    model.save('C:/Users/HaveNiceDay/Desktop/608_3/pModel/CNN-BiLSTM_NoN' + str(k) + '.h5')
    print(confusion_matrix(Y_test, y_pred))
    print("------------------------------")
    k = k + 1
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_acc), np.std(cvscores_acc)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_rec), np.std(cvscores_rec)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_pre), np.std(cvscores_pre)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_ham), np.std(cvscores_ham)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_F1), np.std(cvscores_F1)))
