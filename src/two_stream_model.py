# encoding: utf-8
"""
Function:
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
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, \
    Bidirectional, LSTM, GRU, \
    concatenate, Add, Average, Maximum, Reshape

from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from keras.utils.vis_utils import plot_model

from utils.logging_utils import get_logger
from data_preprocessing import PulsePreprocessing, ShowResult, PulseFeature
from utils.CBAM_module import CbamModule
from utils.model.Transformer_Encoder import TransformerEncoder

logger = get_logger()
base_path = Path(__file__).resolve().parent
pulse_dataset_dir = base_path.joinpath("seven_pulse2018")
train_dataset_dir_path = pulse_dataset_dir.joinpath("train")
test_dataset_dir_path = pulse_dataset_dir.joinpath("test")

pulse_all_data_path = pulse_dataset_dir.joinpath("all")
feature_result_dir = base_path.joinpath("feature_result")
assert os.path.exists(pulse_dataset_dir), f"pulse_dataset_dir={pulse_dataset_dir} loading failed!"
model_save_dir = base_path.joinpath("model")
model_save_dir.mkdir(exist_ok=True, parents=True)


# --------------融合的模型
class VGG16Combine(Model):
    def __init__(self):
        super(VGG16Combine, self).__init__()
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


time_model = tf.keras.Sequential([
    Bidirectional(LSTM(100, return_sequences=True)),  # 2个循环计算层，因此中间层的循环核需要输出每个时间步的状态信息 ht
    Dropout(0.1),
    Bidirectional(LSTM(100)),
    Dropout(0.1),
    Dense(64, activation='selu'),
    Dropout(0.1),
    Dense(7, activation='softmax')  # 用前60天开盘价，预测第61天的open价格（因此只有1个值）
])

freq_model = VGG16Combine()


# multi_attention_head_num = 10
# multi_head_layers = 12
# feature_num_each_step = 60
# assert feature_num_each_step % multi_attention_head_num == 0, f"MultiHead-attention num ERROR!"
# time_model = TransformerEncoder(num_layers=multi_head_layers, d_model=feature_num_each_step,
#                                 num_heads=multi_attention_head_num,
#                                 dff=240, )


# -----------------------


def two_stream_classifier_late_fusion(time_model, freq_model,freq_input_shape=(119,200,1)):
    time_model_input = Input(shape=(40, 60), name="time_model_input")  # 1x2400 -> 40x60
    freq_model_input = Input(shape=freq_input_shape, name="freq_model_input")  # MFCC 119x60x1
    time_result = time_model(time_model_input)
    freq_result = freq_model(freq_model_input)

    # two_stream_result = Average()([time_result, freq_result])  # two-stream late fusion
    two_stream_result = freq_result  # frequency only
    # two_stream_result = time_result

    network = Model(inputs=[time_model_input, freq_model_input], outputs=two_stream_result, name="two_stream_PulseNet")
    return network


def two_stream_classifier_early_fusion(time_model, freq_model, freq_input_shape=(119,200,1)):
    time_model_input = Input(shape=(40, 60), name="time_model_input")  # 1x2400 -> 40x60
    freq_model_input = Input(shape=freq_input_shape, name="freq_model_input")  # MFCC 119x60x1

    time_result = time_model(time_model_input, training=False)
    freq_result = freq_model(freq_model_input, training=False)
    # shape == (None,7)

    # # 0. 按特征维度扩展
    # combined_result = concatenate([time_result, freq_result])
    # 1. 按通道维度扩展
    time_result = tf.expand_dims(time_result, -1)
    time_result = tf.expand_dims(time_result, -1)
    freq_result = tf.expand_dims(freq_result, -1)
    freq_result = tf.expand_dims(freq_result, -1)
    combined_result = concatenate([time_result, freq_result])
    # print(combined_result.shape)
    # # combined_result = cbam_module(combined_result)
    combined_result = CbamModule(channel=2)(combined_result)
    combined_result = Flatten()(combined_result)
    two_stream_result = Dense(7, activation="softmax")(combined_result)

    network = Model(inputs=[time_model_input, freq_model_input], outputs=two_stream_result, name="two_stream_PulseNet")
    return network


class TwoStreamPredict:
    def __init__(self, pulse_feature_instance: PulseFeature, freq_model: Model, time_model: Model,
                 feature_sample_rate=666, freq_frame_info=(840, 280), time_frame_info=(2400, 350, 40),
                 uniform_sample_num=10, freq_feature_choice="MFCC"):
        self.pulse_feature = pulse_feature_instance
        self.sample_rate = feature_sample_rate
        self.uniform_sample_num = uniform_sample_num
        assert freq_feature_choice in ["MFCC", "Fbank"], f"freq_feature_choice={freq_feature_choice} ERROR!"
        self.freq_feature_choice = True if freq_feature_choice == "MFCC" else False  # True->MFCC, False->Fbank

        # 分片信息
        self.freq_frame_length = freq_frame_info[0]  # 频域模型使用的分片大小
        self.freq_frame_step = freq_frame_info[1]
        self.time_frame_length = time_frame_info[0]
        self.time_frame_step = time_frame_info[1]
        self.RNN_time_step = time_frame_info[2]
        assert self.time_frame_length % self.RNN_time_step == 0, f"RNN input shape error!"
        self.RNN_feature_num_each_input = int(self.time_frame_length / self.RNN_time_step)

        self.freq_model = freq_model
        self.time_model = time_model
        self.two_stream_model = two_stream_classifier_late_fusion(time_model=self.time_model,
                                                                  freq_model=self.freq_model)

    @staticmethod
    def get_data_slice(wave_data_array, frame_length, frame_step):
        assert isinstance(wave_data_array, np.ndarray) and len(wave_data_array.shape) == 1, f"wave data shape error!"

        signal_length = len(wave_data_array)
        num_frames = int(
            np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1
        pad_signal_length = (num_frames - 1) * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        # 分帧后最后一帧点数不足，则补零
        # 获取帧：frames 是二维数组，每一行是一帧，列数是每帧的采样点数，之后的短时 fft 直接在每一列上操作
        pad_signal = np.append(wave_data_array, z)
        indices = np.arange(0, frame_length).reshape(
            1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1, 1)
        frames = pad_signal[indices]
        return frames

    def batch_sample_predict(self, x_test_list: List[np.ndarray], numcep_mfcc=20) -> np.ndarray:
        """
        对每个原始样本预测
        :param x_test_list: 原始波形样本，长度可能不一致！
        :param numcep_mfcc:
        :return: y_sample_predict_score_result
        """
        assert len(x_test_list[0].shape) == 1, f"x_test_list element_array shape error!"
        y_sample_predict_result = []
        for index in range(len(x_test_list)):
            x_data = x_test_list[index]
            print(f"x_data shape={x_data.shape}")
            # 0. 时域分片 1x2400
            x_data_frames = self.get_data_slice(wave_data_array=x_data,
                                                frame_length=self.time_frame_length, frame_step=self.time_frame_step)

            # 均匀取 n 帧 进行预测
            select_index_array = np.linspace(0, x_data_frames.shape[0], self.uniform_sample_num,
                                             endpoint=False, dtype=int)
            select_slice_array = x_data_frames[select_index_array, :]
            # print(f"select_index_array={select_index_array}, select_slice_array shape={select_slice_array.shape}")

            # 1. 频域时域输入特征转化
            # 时域
            time_data_array = np.reshape(select_slice_array, (select_slice_array.shape[0],
                                                              self.RNN_time_step, self.RNN_feature_num_each_input))
            # 频域
            freq_data_array = []
            for wave_data in select_slice_array:
                wave_data = np.reshape(wave_data[:self.freq_frame_length], (1, self.freq_frame_length))
                if self.freq_feature_choice:
                    freq_feature_matrix = self.pulse_feature.get_dynamic_mfcc_matrix(wave_data=wave_data,
                                                                                     sample_rate=self.sample_rate,
                                                                                     numcep=numcep_mfcc)
                else:
                    freq_feature_matrix = self.pulse_feature.get_fbank_matrix(wave_data=wave_data,
                                                                              sample_rate=sample_rate)
                freq_data_array.append(freq_feature_matrix)
            freq_data_array = np.array(freq_data_array)
            freq_data_array = np.reshape(freq_data_array, (*freq_data_array.shape, 1))
            print(f"freq_data_array.shape={freq_data_array.shape}")

            # 2. 对每个分片预测
            # # 0. 单独调用方式
            # freq_predict_scores = self.freq_model.predict(freq_data_array)
            # time_predict_scores = self.time_model.predict(time_data_array)
            # y_slices_predict_scores = (freq_predict_scores + time_predict_scores) / 2

            # 1. 模型融合
            y_slices_predict_scores = self.two_stream_model.predict([time_data_array, freq_data_array])

            y_sample_predict_scores = np.average(y_slices_predict_scores, axis=0)
            y_sample_predict_result.append(y_sample_predict_scores)
            print(
                f"index={index} y_sample_predict_scores={y_sample_predict_scores} shape={y_sample_predict_scores.shape}")
            # break

        return np.array(y_sample_predict_result)

    def predict_time_slice_2_stream(self, x_test_array: np.ndarray, numcep_mfcc=20) -> np.ndarray:
        """
        直接取分片好的 1x2400 大小分片数据，进行预测
        :param x_test_array: Nx2400
        :param numcep_mfcc:
        :return: y_sample_predict_scores -> Nx7
        """
        assert isinstance(x_test_array, np.ndarray) and len(x_test_array.shape) == 2, "x_test shape not match!"

        # 1. 频域时域输入特征转化
        # 时域
        time_data_array = np.reshape(x_test_array, (x_test_array.shape[0],
                                                    self.RNN_time_step, self.RNN_feature_num_each_input))
        # 频域
        freq_data_array = []
        for index in range(x_test_array.shape[0]):
            wave_data = x_test_array[index, :self.freq_frame_length]
            wave_data = np.reshape(wave_data, (1, -1))
            if self.freq_feature_choice:
                freq_feature_matrix = self.pulse_feature.get_dynamic_mfcc_matrix(wave_data=wave_data,
                                                                                 sample_rate=self.sample_rate,
                                                                                 numcep=numcep_mfcc)
            else:
                freq_feature_matrix = self.pulse_feature.get_fbank_matrix(wave_data=wave_data,
                                                                          sample_rate=sample_rate)
            freq_data_array.append(freq_feature_matrix)
        freq_data_array = np.array(freq_data_array)
        freq_data_array = np.reshape(freq_data_array, (*freq_data_array.shape, 1))
        print(f"freq_data_array.shape={freq_data_array.shape}")

        # two-stream model predict
        y_sample_predict_scores = self.two_stream_model.predict([time_data_array, freq_data_array])
        print(f"y_sample_predict_scores={y_sample_predict_scores} shape={y_sample_predict_scores.shape}")

        return y_sample_predict_scores


class TwoStreamEarlyFusionPredict:

    def __init__(self, pulse_feature_instance: PulseFeature, two_stream_model: Model,
                 feature_sample_rate=666, freq_frame_info=(840, 280), time_frame_info=(2400, 350, 40),
                 uniform_sample_num=10, freq_feature_choice="MFCC"):
        self.pulse_feature = pulse_feature_instance
        self.sample_rate = feature_sample_rate
        self.uniform_sample_num = uniform_sample_num
        assert freq_feature_choice in ["MFCC", "Fbank"], f"freq_feature_choice={freq_feature_choice} ERROR!"
        self.freq_feature_choice = True if freq_feature_choice == "MFCC" else False  # True->MFCC, False->Fbank

        # 分片信息
        self.freq_frame_length = freq_frame_info[0]  # 频域模型使用的分片大小
        self.freq_frame_step = freq_frame_info[1]
        self.time_frame_length = time_frame_info[0]
        self.time_frame_step = time_frame_info[1]
        self.RNN_time_step = time_frame_info[2]
        assert self.time_frame_length % self.RNN_time_step == 0, f"RNN input shape error!"
        self.RNN_feature_num_each_input = int(self.time_frame_length / self.RNN_time_step)

        self.two_stream_model = two_stream_model

    @staticmethod
    def get_data_slice(wave_data_array, frame_length, frame_step):
        assert isinstance(wave_data_array, np.ndarray) and len(wave_data_array.shape) == 1, f"wave data shape error!"

        signal_length = len(wave_data_array)
        num_frames = int(
            np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1
        pad_signal_length = (num_frames - 1) * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        # 分帧后最后一帧点数不足，则补零
        # 获取帧：frames 是二维数组，每一行是一帧，列数是每帧的采样点数，之后的短时 fft 直接在每一列上操作
        pad_signal = np.append(wave_data_array, z)
        indices = np.arange(0, frame_length).reshape(
            1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1, 1)
        frames = pad_signal[indices]
        return frames

    def batch_sample_predict(self, x_test_list: List[np.ndarray], numcep_mfcc=20) -> np.ndarray:
        """
        对每个原始样本预测
        :param x_test_list: 原始波形样本，长度可能不一致！
        :param numcep_mfcc:
        :return: y_sample_predict_score_result
        """
        assert len(x_test_list[0].shape) == 1, f"x_test_list element_array shape error!"
        y_sample_predict_result = []
        for index in range(len(x_test_list)):
            x_data = x_test_list[index]
            print(f"x_data shape={x_data.shape}")
            # 0. 时域分片 1x2400
            x_data_frames = self.get_data_slice(wave_data_array=x_data,
                                                frame_length=self.time_frame_length, frame_step=self.time_frame_step)

            # 均匀取 n 帧 进行预测
            select_index_array = np.linspace(0, x_data_frames.shape[0], self.uniform_sample_num,
                                             endpoint=False, dtype=int)
            select_slice_array = x_data_frames[select_index_array, :]
            # print(f"select_index_array={select_index_array}, select_slice_array shape={select_slice_array.shape}")

            # 1. 频域时域输入特征转化
            # 时域
            time_data_array = np.reshape(select_slice_array, (select_slice_array.shape[0],
                                                              self.RNN_time_step, self.RNN_feature_num_each_input))
            # 频域
            freq_data_array = []
            for wave_data in select_slice_array:
                wave_data = np.reshape(wave_data[:self.freq_frame_length], (1, self.freq_frame_length))
                if self.freq_feature_choice:
                    freq_feature_matrix = self.pulse_feature.get_dynamic_mfcc_matrix(wave_data=wave_data,
                                                                                     sample_rate=self.sample_rate,
                                                                                     numcep=numcep_mfcc)
                else:
                    freq_feature_matrix = self.pulse_feature.get_fbank_matrix(wave_data=wave_data,
                                                                              sample_rate=sample_rate)
                freq_data_array.append(freq_feature_matrix)
            freq_data_array = np.array(freq_data_array)
            freq_data_array = np.reshape(freq_data_array, (*freq_data_array.shape, 1))
            print(f"freq_data_array.shape={freq_data_array.shape}")

            # 2. 对每个分片预测
            # # 0. 单独调用方式
            # freq_predict_scores = self.freq_model.predict(freq_data_array)
            # time_predict_scores = self.time_model.predict(time_data_array)
            # y_slices_predict_scores = (freq_predict_scores + time_predict_scores) / 2

            # 1. 模型融合
            y_slices_predict_scores = self.two_stream_model.predict([time_data_array, freq_data_array])

            y_sample_predict_scores = np.average(y_slices_predict_scores, axis=0)
            y_sample_predict_result.append(y_sample_predict_scores)
            print(
                f"index={index} y_sample_predict_scores={y_sample_predict_scores} shape={y_sample_predict_scores.shape}")
            # break

        return np.array(y_sample_predict_result)

    def predict_time_slice_2_stream(self, x_test_array: np.ndarray, numcep_mfcc=20) -> np.ndarray:
        """
        直接取分片好的 1x2400 大小分片数据，进行预测
        :param x_test_array: Nx2400
        :param numcep_mfcc:
        :return: y_sample_predict_scores -> Nx7
        """
        assert isinstance(x_test_array, np.ndarray) and len(x_test_array.shape) == 2, "x_test shape not match!"

        # 1. 频域时域输入特征转化
        # 时域
        time_data_array = np.reshape(x_test_array, (x_test_array.shape[0],
                                                    self.RNN_time_step, self.RNN_feature_num_each_input))
        # 频域
        freq_data_array = []
        for index in range(x_test_array.shape[0]):
            wave_data = x_test_array[index, :self.freq_frame_length]
            wave_data = np.reshape(wave_data, (1, -1))
            if self.freq_feature_choice:
                freq_feature_matrix = self.pulse_feature.get_dynamic_mfcc_matrix(wave_data=wave_data,
                                                                                 sample_rate=self.sample_rate,
                                                                                 numcep=numcep_mfcc)
            else:
                freq_feature_matrix = self.pulse_feature.get_fbank_matrix(wave_data=wave_data,
                                                                          sample_rate=sample_rate)
            freq_data_array.append(freq_feature_matrix)
        freq_data_array = np.array(freq_data_array)
        freq_data_array = np.reshape(freq_data_array, (*freq_data_array.shape, 1))
        print(f"freq_data_array.shape={freq_data_array.shape}")

        # two-stream model predict
        y_sample_predict_scores = self.two_stream_model.predict([time_data_array, freq_data_array])
        print(f"y_sample_predict_scores={y_sample_predict_scores} shape={y_sample_predict_scores.shape}")

        return y_sample_predict_scores


if __name__ == '__main__':
    pulse_preprocess = PulsePreprocessing()
    pulse_feature = PulseFeature()

    # 降采样数据
    downsampled_npz_filename = "downsampled_train_test_dict.npz"
    sample_rate = 666.0  # 统一的采样率
    npz_save_path_str = pulse_dataset_dir.joinpath(downsampled_npz_filename).as_posix()
    assert Path(npz_save_path_str).is_file(), f"文件不存在, filepath={npz_save_path_str}"
    pulse_all_class_data_dict = pulse_preprocess.load_npy_saved_pulse_data(npz_save_path=npz_save_path_str)

    # 去基线漂移+去噪
    start_time = time.time()
    pulse_all_class_data_dict = pulse_preprocess.detrend_wave_data(pulse_all_class_data_dict)
    pulse_all_class_data_dict = pulse_preprocess.detrend_wave_data(pulse_all_class_data_dict)
    print(f"去噪+去基线漂移完成！costs={time.time() - start_time}s")

    # 测试数据: 取出完整样本预测
    x_test = []
    y_test = []
    for class_no in pulse_all_class_data_dict:
        data_type = "test"
        for data_info_dict in pulse_all_class_data_dict[class_no][data_type]:
            x_test.append(data_info_dict["data"][0, :])
            y_test.append(class_no)
    x_test_list = x_test  # 长度可能不一致！
    y_test = np.array([y_test]).T

    # 对每个样本预测
    # 0. 加载模型
    time_model_name = "BiLSTM-2layer_detrend_balance_TimeDomain_100_666.0_2400x350_best_checkpoint"
    freq_model_name = "VGG16_detrend_MFCC_200_666.0_840x280_200epoch_best_checkpoint"
    time_model_path = model_save_dir.joinpath(time_model_name)
    freq_model_path = model_save_dir.joinpath(freq_model_name)
    assert time_model_path.exists() and freq_model_path.exists(), f"model file not exists"
    time_domain_model = load_model(time_model_path)
    freq_domain_model = load_model(freq_model_path)

    # 1. 分片信息
    _time_frame_info = (2400, 350, 40)  # frame_length, frame_step, RNN time step
    _freq_frame_info = (840, 280)
    numcep_mfcc = 20
    freq_feature_choice_str = "Fbank"
    assert _time_frame_info[0] % _time_frame_info[2] == 0, f"RNN input shape error!"
    two_stream_predict = TwoStreamPredict(pulse_feature_instance=pulse_feature,
                                          freq_model=freq_domain_model, time_model=time_domain_model,
                                          feature_sample_rate=666, freq_frame_info=_freq_frame_info,
                                          time_frame_info=_time_frame_info,
                                          uniform_sample_num=10, freq_feature_choice=freq_feature_choice_str)
    y_pred_score_result = two_stream_predict.batch_sample_predict(x_test_list=x_test_list,
                                                                  numcep_mfcc=numcep_mfcc)
