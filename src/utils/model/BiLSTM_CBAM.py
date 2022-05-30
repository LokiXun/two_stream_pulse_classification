# encoding: utf-8
"""
Function:
@author: LokiXun
@contact: 2682414501@qq.com
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout, GRU, \
    Conv1D, MaxPooling1D, Flatten, Reshape
from tensorflow.keras import Input, Model

from utils.CBAM_module import cbam_module

# 搭建 RNN 结构
# BiLSTM-2layer
BiLSTM_model = tf.keras.Sequential([
    Bidirectional(LSTM(128, return_sequences=True)),  # 2个循环计算层，因此中间层的循环核需要输出每个时间步的状态信息 ht
    Bidirectional(LSTM(256)),
    Dense(128, activation='selu'),
    Dropout(0.1),
    Dense(7, activation='softmax')
])

# BiGRU
BiGRU_model = tf.keras.Sequential([
    Bidirectional(GRU(128, return_sequences=True)),  # 2个循环计算层，因此中间层的循环核需要输出每个时间步的状态信息 ht
    # Dropout(0.1),
    Bidirectional(GRU(256)),
    # Dropout(0.1),
    Dense(128, activation='selu'),
    Dropout(0.1),
    Dense(7, activation='softmax')
])

# CNN-BiLSTM: 输入层加了2层卷积
CNN_BiLSTM_model = tf.keras.Sequential([
    Conv1D(32, 3, strides=1, name='conv1', use_bias=False),
    MaxPooling1D(3, padding="same"),
    Conv1D(64, 3, strides=1, name='conv2', use_bias=False),
    MaxPooling1D(3, padding="same"),
    Conv1D(128, 3, strides=1, name='conv3', use_bias=False),
    MaxPooling1D(3, padding="same"),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.1),
    Bidirectional(LSTM(64)),
    Dropout(0.1),
    Flatten(),
    Dense(7, activation='softmax')
])


def BiLSTM_CBAM():
    """BiLSTM 输入增加 CBAM： 认为有的时间步重要，有的不重要"""
    x = Input(shape=(40, 60), name="BiLSTM_CBAM")

    # CBAM Attention
    x = Reshape((40, 60, 1))(x)
    x = tf.transpose(x, [0, 3, 1, 2])
    x = cbam_module(x)
    x = tf.transpose(x, [0, 2, 3, 1])
    x = Reshape((40, 60))(x)

    lstm_model = tf.keras.Sequential([
        Bidirectional(LSTM(128, return_sequences=True)),  # 2个循环计算层，因此中间层的循环核需要输出每个时间步的状态信息 ht
        Bidirectional(LSTM(256)),
        Dense(128, activation='selu'),
        Dropout(0.1),
        Dense(7, activation='softmax')
    ])
    y = lstm_model(x)

    _model = Model(inputs=[x], outputs=y, name="CBAM_BiLSTM")
    return _model


if __name__ == '__main__':
    model = BiLSTM_CBAM()
