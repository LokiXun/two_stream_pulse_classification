"""CNN-BiLSTM structure, created by 陈磊学长"""
import numpy as np
import pandas as pd
import h5py
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense,Masking,LSTM,Bidirectional,GRU
from keras.utils import np_utils
import random
import anotherdg2 as dg
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from itertools import cycle
from sklearn.preprocessing import LabelEncoder
from scipy import interp

from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers import *
from keras.models import *


maxlen = 153
INPUT_DIM = 1
lstm_units = 16
def CM(y_true,y_pred):
    F1 = f1_score( y_true, y_pred, average='macro' )
    Precision = precision_score(y_true, y_pred, average='macro')
    Recall = recall_score(y_true, y_pred, average='macro')
    return Recall,Precision,F1,
def predictFromModel(model, testX):
    testX = testX.astype(float)
    predictions = model.predict(testX)
    #print predictions
    predictions = np.argmax(predictions,axis=1)
    

    return predictions


def calcAccuracy(testY, p):
    # print the accuracy
    accuracy = sum(testY == p)/float(len(testY))
    return accuracy

def Normalize(group):
	minVals = group.min(0)  # 为0时：求每列的最小值[0 3 1]   .shape=(3,)
	maxVals = group.max(0)  # 为0时：求每列的最大值[2 7 8]   .shape=(3,)
	ranges = maxVals - minVals
 
	m = group.shape[0]
	normDataSet = np.zeros(np.shape(group))       #  np.shape(group) 返回一个和group一样大小的数组，但元素都为0
	diffnormData =group - np.tile(minVals,(m,1))  #  (oldValue-min)  减去最小值
	normDataSet1 =diffnormData / np.tile(ranges,(m,1))
	return normDataSet1
maxlen=154
path1 = 'C:/transformer/image'

label, data = dg.data_root(path1)

t_all, p_all, data, label = dg.data_test_train(data,label)
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
cvscores_LA_ac=[]
cvscores_LA_re=[]
cvscores_LA_pr=[]
cvscores_LA_f1=[]
cvscores_LA_ha=[]
X=np.array(data)
Y=np.array(label)
for train, test in kfold.split(X, Y):

	X_train=X[train]
	Y_train=Y[train]
	X_test=X[test]
	Y_test=Y[test]
	'''
	X_train1=[]
	X_test1=[]
	for arr in X_train:
		while len(arr)<154:
			arr.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
		if len(arr)>154:
			arr=arr[:154]
		X_train1.append(arr)
	for arr in X_test:
		while len(arr)<154:
			arr.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
		if len(arr)>154:
			arr=arr[:154]
		X_test1.append(arr)	
	X_train=np.array(X_train1)
	X_test=np.array(X_test1)
	'''
	X_train=sequence.pad_sequences(X_train, maxlen=maxlen, padding='post')
	X_test=sequence.pad_sequences(X_test, maxlen=maxlen, padding='post')
	X_train=np.array(X_train)
	X_test=np.array(X_test)
	batch_size = 5
	max_features=30000

	encoder = LabelEncoder()
	encoder.fit(Y_train)
	encoded_Y = encoder.transform(Y_train)
	y_traint = np_utils.to_categorical(encoded_Y)
	encoded_Y = encoder.transform(Y_test)
	y_testt = np_utils.to_categorical(encoded_Y)
	shuffle_vector = np.random.randint(low=0, high=X_train.shape[0], size=X_train.shape[0])
	X_train=X_train[shuffle_vector]
	y_traint=y_traint[shuffle_vector]
	#模型训练
	inputs = Input(shape=(maxlen, 36))

	
	con_out1=Conv1D(32,3)(inputs)

	
	pool1=MaxPooling1D(2)(con_out1)

	lstm_out1 = Bidirectional(LSTM(lstm_units, return_sequences=True))(pool1)
	drop1=Dropout(0.1)(lstm_out1)
	lstm_out2 = Bidirectional(LSTM(lstm_units, return_sequences=True))(drop1)
	drop2=Dropout(0.5)(lstm_out2)
	attention_flatten = Flatten()(drop2)
	#out=Dense(30)(attention_flatten)
	#out2=Dense(10)(out)
	output = Dense(2, activation='sigmoid')(attention_flatten)
	'''
	lstm_out1 = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
	drop2=Dropout(0.1)(lstm_out1)

	attention_mul = attention_3d_block(drop2)
	attention_flatten = Flatten()(attention_mul)
	output = Dense(2, activation='sigmoid')(attention_flatten)
	'''
	model = Model(inputs=inputs, outputs=output)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#模型测试
	model.fit(X_train, y_traint, epochs=100, batch_size=32,validation_split=0.3)
	predictions = predictFromModel(model, X_test)
	acc_LA=calcAccuracy(Y_test, predictions)
	re_LA,pr_LA,f1_LA = CM(Y_test,predictions)
	cvscores_LA_ac.append(acc_LA * 100)
	cvscores_LA_re.append(re_LA * 100)
	cvscores_LA_pr.append(pr_LA * 100)
	cvscores_LA_f1.append(f1_LA * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_LA_ac), np.std(cvscores_LA_ac)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_LA_re), np.std(cvscores_LA_re)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_LA_pr), np.std(cvscores_LA_pr)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_LA_f1), np.std(cvscores_LA_f1)))



