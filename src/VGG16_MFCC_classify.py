# encoding: utf-8
"""
Function: 频域分类 MFCC、Fbank + 图像分类
@author: LokiXun
@contact: 2682414501@qq.com
"""
import os
from pathlib import Path
import random
from typing import Tuple, Union, Dict, List
import math
import time

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback

from utils.logging_utils import get_logger
from data_preprocessing import PulsePreprocessing, ShowResult

np.set_printoptions(threshold=np.inf)

logger = get_logger()
base_path = Path(__file__).resolve().parent
pulse_dataset_dir = base_path.joinpath("seven_pulse2018")
train_dataset_dir_path = pulse_dataset_dir.joinpath("train")
test_dataset_dir_path = pulse_dataset_dir.joinpath("test")

pulse_all_data_path = pulse_dataset_dir.joinpath("all")
feature_result_dir = base_path.joinpath("feature_result")
assert os.path.exists(pulse_dataset_dir), f"pulse_dataset_dir={pulse_dataset_dir} loading failed!"
# model_save_dir = base_path.joinpath("model")
# model_save_dir.mkdir(exist_ok=True, parents=True)

# # ------------------------- 分片 --------------------------------------------------------------
# # 读取数据信息
# import math
#
# pulse_preprocess = PulsePreprocessing()
# downsampled_npz_filename = "downsampled_train_test_dict.npz"
sample_rate = 666.0  # 统一的采样率
# npz_save_path_str = pulse_dataset_dir.joinpath(downsampled_npz_filename).as_posix()
# assert Path(npz_save_path_str).is_file(), f"文件不存在, filepath={npz_save_path_str}"
# pulse_all_class_data_dict = pulse_preprocess.load_npy_saved_pulse_data(npz_save_path=npz_save_path_str)
#
# # 切片
# start_time = time.time()
heart_beat_average_seconds = 1.2  # 按平均一次心跳的秒数分片
frame_length = math.ceil(heart_beat_average_seconds * sample_rate) + 40  # 840
frame_step = 280  # int(frame_length/3)
# # x_train, y_train, x_test, y_test = PulsePreprocessing.get_train_test_data_slice(
# #     pulse_all_class_data_dict, frame_length=frame_length, frame_step=frame_step)
# # print(f"get data slices costs={time.time() - start_time}s")  # 9s

# --------------------------读取 MFCC 本地文件----------------------------------------------------
mfcc_pulse_data_npz_filename = "mfcc_pulse_data_600x350.npz"
data_save_path = pulse_dataset_dir.joinpath(mfcc_pulse_data_npz_filename).as_posix()
mfcc_pulse_data_dict = PulsePreprocessing.load_npy_saved_pulse_data(data_save_path)
x_train, y_train, x_test, y_test = mfcc_pulse_data_dict["x_train"], mfcc_pulse_data_dict["y_train"], \
                                   mfcc_pulse_data_dict["x_test"], mfcc_pulse_data_dict["y_test"]
del mfcc_pulse_data_dict
x_train = np.reshape(x_train, (*x_train.shape, 1))  # 增加一维
x_test = np.reshape(x_test, (*x_test.shape, 1))

# 打乱数据集
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
# --------------------------train test split-----------------------------------------------------
# 分割 train、validation set
assert len(x_train.shape) == 3, "x_train 没有增加一维！"
print(f"origin x_train.shape={x_train.shape}, y_train.shape={y_train.shape}")

train_num = int(x_train.shape[0] * 0.9)
x_validate = x_train[train_num:, :]
y_validate = y_train[train_num:, :]
x_train = x_train[0:train_num, :]
y_train = y_train[0:train_num, :]

print(y_train[:10])  # 检查是否已经打乱
print(f"x_train.shape={x_train.shape}, y_train.shape={y_train.shape},"
      f"x_validate.shape={x_validate.shape}, y_validate.shape={y_validate.shape}")


# ------------------------
# DataLoader 方式读取: 无法一次装入内存时使用
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_path_array, y_data_array, batch_size=64, shuffle=True, feature_save_dir_path=""):
        self.x_path_array = x_path_array  # npy 文件路径 array
        self.y_data_array = y_data_array
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index_array = np.arange(len(self.x_path_array))
        self.feature_save_dir = Path(feature_save_dir_path)
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x_path_array) / self.batch_size)

    def __getitem__(self, idx):
        # Generate indexes of the batch
        indexes = self.index_array[idx * self.batch_size: (idx + 1) * self.batch_size]
        # Find list of IDs
        y_data_return = self.y_data_array[indexes, :]
        x_path_list_temp = [self.x_path_array[k, :] for k in indexes]  # 取出一部分数据
        x_data_return = self.__load_batch_x_data_from_file(x_path_list_temp)
        if len(x_data_return.shape) != 4:
            x_data_return = np.reshape(x_data_return, (*x_data_return.shape, 1))
        print(f"x_data_return shape={x_data_return.shape}")
        assert False
        return x_data_return, y_data_return

    def on_epoch_end(self):
        self.index_array = np.arange(len(self.x_path_array))
        if self.shuffle:
            np.random.shuffle(self.index_array)

    def __load_batch_x_data_from_file(self, x_path_list) -> np.ndarray:
        """读取本地文件，生成 batch data（ndarray）"""
        x_return_array = []
        for idx in range(len(x_path_list)):
            npy_name = Path(x_path_list[idx][0]).name  # avoid saved as absolute path
            x_data_save_path = self.feature_save_dir.joinpath(f"{npy_name}.npy").as_posix()
            x_data = np.load(x_data_save_path)
            x_return_array.append(x_data)

        return np.array(x_return_array)


# ---------------------------model definition------------------------------------------------------------
class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()
        self.c1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')  # 卷积层1
        self.b1 = BatchNormalization()  # BN层1
        self.a1 = Activation('selu')  # 激活层1
        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', )
        self.b2 = BatchNormalization()  # BN层1
        self.a2 = Activation('selu')  # 激活层1
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = Dropout(0.2)  # dropout层

        self.c3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b3 = BatchNormalization()  # BN层1
        self.a3 = Activation('selu')  # 激活层1
        self.c4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b4 = BatchNormalization()  # BN层1
        self.a4 = Activation('selu')  # 激活层1
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d2 = Dropout(0.2)  # dropout层

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b5 = BatchNormalization()  # BN层1
        self.a5 = Activation('selu')  # 激活层1
        self.c6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b6 = BatchNormalization()  # BN层1
        self.a6 = Activation('selu')  # 激活层1
        self.c7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b7 = BatchNormalization()
        self.a7 = Activation('selu')
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d3 = Dropout(0.2)

        self.c8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b8 = BatchNormalization()  # BN层1
        self.a8 = Activation('selu')  # 激活层1
        self.c9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b9 = BatchNormalization()  # BN层1
        self.a9 = Activation('selu')  # 激活层1
        self.c10 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b10 = BatchNormalization()
        self.a10 = Activation('selu')
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d4 = Dropout(0.2)

        self.c11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b11 = BatchNormalization()  # BN层1
        self.a11 = Activation('selu')  # 激活层1
        self.c12 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b12 = BatchNormalization()  # BN层1
        self.a12 = Activation('selu')  # 激活层1
        self.c13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b13 = BatchNormalization()
        self.a13 = Activation('selu')
        self.p5 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d5 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='selu')
        self.d6 = Dropout(0.2)
        self.f2 = Dense(512, activation='selu')
        self.d7 = Dropout(0.2)
        self.f3 = Dense(7, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.b7(x)
        x = self.a7(x)
        x = self.p3(x)
        x = self.d3(x)

        x = self.c8(x)
        x = self.b8(x)
        x = self.a8(x)
        x = self.c9(x)
        x = self.b9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.b10(x)
        x = self.a10(x)
        x = self.p4(x)
        x = self.d4(x)

        x = self.c11(x)
        x = self.b11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.b12(x)
        x = self.a12(x)
        x = self.c13(x)
        x = self.b13(x)
        x = self.a13(x)
        x = self.p5(x)
        x = self.d5(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d6(x)
        x = self.f2(x)
        x = self.d7(x)
        y = self.f3(x)
        return y


class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        """
        ResNet块，由 2 层卷积组成，其中第 1 层卷积层的 strides 最后输出尺寸是否改变
        :param filters:
        :param strides:
        :param residual_path:
        """
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        # 矩阵对应元素相加
        return out


class ResNet18(Model):
    """ResNet 采用两种残差跳连结构: ResNet18/34, ResNet50/101/152"""

    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍

        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.blocks(x)

        x = self.p1(x)
        y = self.f1(x)
        return y


# model = ResNet18([2, 2, 2, 2])
# model = ResNet18([3, 4, 6, 3])  # ResNet34

model = VGG16()
from tensorflow.keras.layers import GlobalAvgPool2D, GlobalMaxPooling2D, Reshape, Dense, Add, Activation, Multiply, \
    Concatenate


# CBAM module
class CbamModule(Model):
    def __init__(self, channel, reduction_ratio=16):
        super(CbamModule, self).__init__()
        self.reduction_ratio = reduction_ratio
        # CAM
        self.cam_maxpool = GlobalMaxPooling2D(name="cbam_maxPool")
        self.cam_avgpool = GlobalAvgPool2D(name="cbam_avgPool")
        self.Dense_One = Dense(units=int(channel//reduction_ratio), activation='relu', kernel_initializer='he_normal',
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


# # --------------------------------TRAINING-----------------------------------------------------------------------
# 训练
import time

epoch_num = 120
batch_size = 80
# Fbank
# train_generator = DataGenerator(x_train, y_train, batch_size=batch_size,
#                                 feature_save_dir_path=fbank_save_dir.as_posix())
# validate_generator = DataGenerator(x_validate, y_validate, batch_size=batch_size,
#                                    feature_save_dir_path=fbank_save_dir.as_posix())

feature_data_name = "Fbank"  # "Fbank"  if x_train.shape[2]>100 else "MFCC"
data_process_type_str = "detrend_balance"  # "raw"
# assert False, "warning 检查修改 feature + model名字"

model_name_str = f"VGG16_{data_process_type_str}_{feature_data_name}_{epoch_num}_{sample_rate}_{frame_length}x{frame_step}"
learning_rate = 1e-4
model.compile(optimizer='adam',  # tf.keras.optimizers.Adam(learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy', 'accuracy'])

# 0.0 断点续训
model_save_dir = base_path.joinpath("model")
model_save_dir.mkdir(exist_ok=True, parents=True)
checkpoint_save_path = model_save_dir.joinpath(f"checkpoint_{model_name_str}.ckpt").as_posix()
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)


# 0.1 每个 epoch 保存模型
class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch >= 10:
            model_save_path = model_save_dir.joinpath(f"{model_name_str}_epoch{epoch + 120}_bt{batch_size}").as_posix()
            model.save(model_save_path)


# my_call_back=MyCallback()


start_time = time.time()
# 直接使用 x_train,y_train 方式
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_num, validation_data=(x_validate, y_validate),
                    validation_freq=1,
                    callbacks=[cp_callback])
# # Fbank 使用 Datagenerator 分批从本地读
# history = model.fit(train_generator, batch_size=batch_size, epochs=epoch_num, validation_data=validate_generator,
#                     validation_freq=1,
#                     callbacks=[cp_callback])
train_cost_time = int(time.time() - start_time) / 3600
print(f"train_cost_time={train_cost_time}hours")
# # ------------------------------------- 保存模型 -------------------------------------------------------------------------
# model.summary()
# model.save(model_save_dir.joinpath(f"{model_name_str}_100epoch.h5").as_posix())

# save best model recorded by checkpoint
checkpoint_save_path = model_save_dir.joinpath(f"checkpoint_{model_name_str}.ckpt").as_posix()
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
model.save(model_save_dir.joinpath(f"{model_name_str}_best_checkpoint").as_posix())

# # ------------------------------------- 显示训练集和验证集的acc和loss曲线 -------------------------------------------
ShowResult.plot_model_trained_result_acc_loss_curve(history=history, model_name_str=model_name_str)
