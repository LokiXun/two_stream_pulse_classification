# encoding: utf-8
"""
Function: CBAM attention mechanism, 两种实现方式：
    1. 使用 keras.backend 实现（参考博客的，不方便嵌入 Model）
    2. 使用 keras.layers 实现，相应 layer 都有的，完全 ok
@author: LokiXun
@contact: 2682414501@qq.com
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, \
    Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense, Bidirectional, Attention, \
    GlobalAvgPool2D, GlobalMaxPooling2D, Reshape, Dense, Add, Activation, Multiply, Concatenate
from tensorflow.keras import Model, layers, Sequential
import keras.backend as K
import keras.layers as KL

# 判断输入数据格式，是channels_first还是channels_last
channel_axis = 1 if K.image_data_format() == "channels_first" else 3


# CAM
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal',
                         use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True,
                         bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])


# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid',
                     kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


def cbam_module(input_xs, reduction_ratio=0.25):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    print(channel_refined_feature.shape)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])


class CbamModule(Model):
    """继承 Model 类，便于嵌入模型， ok的"""

    def __init__(self, channel, reduction_ratio=16):
        super(CbamModule, self).__init__()
        self.reduction_ratio = reduction_ratio
        # CAM
        self.cam_maxpool = GlobalMaxPooling2D(name="cbam_maxPool")
        self.cam_avgpool = GlobalAvgPool2D(name="cbam_avgPool")
        self.Dense_One = Dense(units=int(channel // reduction_ratio), activation='relu', kernel_initializer='he_normal',
                               use_bias=True, bias_initializer='zeros', name="cbam_dense1")
        self.Dense_Two = Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True,
                               bias_initializer='zeros', name="cbam_dense2")
        self.add = Add()
        self.sigmoid_activation = Activation('sigmoid')
        self.multiply = Multiply()
        # SAM
        self.sam_concate = Concatenate(axis=3)
        self.sam_conv2d = Conv2D(filters=1, kernel_size=(7, 7), strides=1, padding='same', name="same_Conv2D_7x7")

    def call(self, input_xs):
        # 1. CAM
        maxpool_channel = self.cam_maxpool(input_xs)
        avgpool_channel = self.cam_avgpool(input_xs)
        # max path
        mlp_1_max = self.Dense_One(maxpool_channel)
        mlp_2_max = self.Dense_Two(mlp_1_max)
        # avg path
        mlp_1_avg = self.Dense_One(avgpool_channel)
        mlp_2_avg = self.Dense_Two(mlp_1_avg)
        channel_attention_feature = self.add([mlp_2_max, mlp_2_avg])
        channel_attention_feature = self.sigmoid_activation(channel_attention_feature)
        channel_refined_feature = self.multiply([channel_attention_feature, input_xs])

        # 2. SAM
        maxpool_spatial = tf.reduce_mean(channel_refined_feature, axis=3, keepdims=True, name="SAM_avgPool")
        avgpool_spatial = tf.reduce_max(channel_refined_feature, axis=3, keepdims=True, name="SAM_avgPool")
        max_avg_pool_spatial = self.sam_concate([maxpool_spatial, avgpool_spatial])
        spatial_attention_feature = self.sam_conv2d(max_avg_pool_spatial)
        spatial_attention_feature = self.sigmoid_activation(spatial_attention_feature)

        # CBAM output
        cbam_output = self.multiply([channel_refined_feature, spatial_attention_feature])
        return self.add([cbam_output, input_xs])


if __name__ == '__main__':
    # 使用numpy模拟一个真实图片的尺寸
    input_xs = np.ones([2, 40, 60], dtype='float32') * 0.5
    # numpy转Tensor
    input_xs = tf.convert_to_tensor(input_xs)
    print(input_xs.shape)  # output： (2, 256, 256, 3)
    # outputs = cbam_module(input_xs)
    # outputs = spatial_attention(input_xs)
    # print(outputs.shape) # output： (2, 256, 256, 3)

    x = Reshape((40, 60, 1))(input_xs)
    print(x.shape)
    # x = cbam_module(x)
    cbam_layer = CbamModule(1)
    x = cbam_layer(x)
    x = Reshape((40, 60))(x)
    print(x.shape)
