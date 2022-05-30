"""波形数据预处理"""
import os
import re
from pathlib import Path
import random
from typing import Tuple, Union, Dict, List
import math
import time

import shutil
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import dct
from scipy.signal import spectrogram
from scipy.signal import resample_poly, butter, filtfilt, medfilt
from python_speech_features import mfcc, delta, fbank
from biosppy.signals import ecg

from utils.const import pulse_class_name_list
from utils.logging_utils import get_logger

logger = get_logger()
base_path = Path(__file__).resolve().parent
pulse_dataset_dir = base_path.joinpath("seven_pulse2018")
train_dataset_dir_path = pulse_dataset_dir.joinpath("train")
test_dataset_dir_path = pulse_dataset_dir.joinpath("test")

pulse_all_data_path = pulse_dataset_dir.joinpath("all")
feature_result_dir = base_path.joinpath("feature_result")
assert os.path.exists(pulse_all_data_path), "origin pulse data loading failed!"
train_dataset_dir_path.mkdir(exist_ok=True, parents=True)
test_dataset_dir_path.mkdir(exist_ok=True, parents=True)
feature_result_dir.mkdir(exist_ok=True, parents=True)


class ShowResult:

    @staticmethod
    def print_data_info(pulse_all_data_dict):
        # count data info: time, pressure
        time_info_dict, pressure_info_dict, sample_rate_dict = {}, {}, {}
        for class_no in pulse_all_data_dict:
            for data_info_dict in pulse_all_data_dict[class_no]["train"]:
                time_info_dict[data_info_dict['t']] = time_info_dict.get(data_info_dict['t'], 0) + 1
                pressure_info_dict[data_info_dict['p']] = pressure_info_dict.get(data_info_dict['p'], 0) + 1
                sample_rate = f"{data_info_dict['data'].shape[1] / eval(data_info_dict['t']):.2f}Hz"
                sample_rate_dict[sample_rate] = sample_rate_dict.get(sample_rate, 0) + 1
            for data_info_dict in pulse_all_data_dict[class_no]["test"]:
                time_info_dict[data_info_dict['t']] = time_info_dict.get(data_info_dict['t'], 0) + 1
                pressure_info_dict[data_info_dict['p']] = pressure_info_dict.get(data_info_dict['p'], 0) + 1
                sample_rate = f"{data_info_dict['data'].shape[1] / eval(data_info_dict['t']):.2f}Hz"
                sample_rate_dict[sample_rate] = sample_rate_dict.get(sample_rate, 0) + 1

        print(f"time_info_dict={time_info_dict}")
        print(f"pressure_info_dict={sorted(pressure_info_dict.items(), key=lambda item: item[1], reverse=True)[:10]}")
        print(f"sample_rate_dict={sorted(sample_rate_dict.items(), key=lambda item: item[0], reverse=False)}")
        each_class_sample_num_list = [
            (class_no, len(pulse_all_data_dict[class_no]['train']), len(pulse_all_data_dict[class_no]['test'])) for
            class_no
            in pulse_all_data_dict]
        print(f"sample num: {each_class_sample_num_list}")

    @staticmethod
    def compare_downsample_effect(origin_pulse_all_class_data_dict, target_sample_rate=666):
        class_no, sample_index = 0, 0
        wave_data_array = origin_pulse_all_class_data_dict[class_no]["train"][sample_index]["data"]
        time_len = eval(origin_pulse_all_class_data_dict[class_no]["train"][sample_index]["t"])
        time_index_list = np.linspace(start=0, stop=wave_data_array.shape[1],
                                      num=math.ceil(time_len * target_sample_rate), endpoint=False, dtype=int)
        # 0. equally select points
        equally_downsampled_data = wave_data_array[:, time_index_list]

        # 1. numpy interp
        downsampled_data = np.interp(time_index_list, np.arange(0, wave_data_array.shape[1]), wave_data_array[0])
        downsampled_data = downsampled_data.reshape((1, downsampled_data.shape[0]))

        # 2. scipy signal
        current_sample_rate = int(wave_data_array.shape[1] / time_len)
        scipy_downsampled_data = resample_poly(wave_data_array[0, :], up=target_sample_rate, down=current_sample_rate)
        scipy_downsampled_data = scipy_downsampled_data.reshape((1, scipy_downsampled_data.shape[0]))
        print(f"wave_data_list={wave_data_array}, shape={wave_data_array.shape}")
        print(f"downsampled_data={downsampled_data}, shape={downsampled_data.shape}")
        print(f"shape={scipy_downsampled_data.shape}, type={type(scipy_downsampled_data)}")

        plt.figure()
        # 1.1 compare 不同降维方法： interp 比较好
        # plt.plot(wave_data_array[0, :600],'k-', label="origin data")
        # plt.plot(scipy_downsampled_data[0, :600], 'b-', label="scipy fft")
        # plt.plot(downsampled_data[0, :600], 'r-', label="numpy interp")
        # plt.legend(loc="best")

        # 1.2 compare 降维效果，与原始信号对比
        plt.subplot(2, 1, 1)
        show_time_seconds = 0.6
        origin_data_point = int(current_sample_rate * show_time_seconds)
        downsampled_data_point = int(target_sample_rate * show_time_seconds)
        plt.plot(wave_data_array[0, :origin_data_point], 'k-')
        plt.title(f"origin: {current_sample_rate}Hz for {show_time_seconds} seconds")
        plt.xlabel('time')
        plt.grid(which="major", axis='both', linestyle='-.')

        plt.subplot(2, 1, 2)
        plt.plot(scipy_downsampled_data[0, :downsampled_data_point], '-', label="scipy fft")
        plt.plot(downsampled_data[0, :downsampled_data_point], '-', label="numpy interp")
        plt.plot(equally_downsampled_data[0, :downsampled_data_point], '-', label="equally downsample")

        plt.title(f"downsampled: {target_sample_rate}Hz for {show_time_seconds} seconds")
        plt.xlabel("time")
        plt.legend(loc="best")
        plt.grid(which="major", axis='both', linestyle='-.')
        plt.show()

    @staticmethod
    def show_slice_effect(x_array, y_array):
        """检查分片长度是否合适"""
        columns_num = 5
        ros_num = int(x_array.shape[0] / columns_num) + 1
        plt.figure(figsize=(15, 5 * ros_num))
        for i in range(0, x_array.shape[0]):
            plt.subplot(ros_num, columns_num, i + 1)
            plt.plot(x_array[i, :])
            plt.title(f"class={y_array[i,]}")
        plt.show()

    @staticmethod
    def plot_model_trained_result_acc_loss_curve(history, model_name_str):
        # 显示训练集和验证集的acc和loss曲线
        acc = history.history['sparse_categorical_accuracy']
        val_acc = history.history['val_sparse_categorical_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(1, len(history.epoch) + 1)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Train Set')
        plt.plot(epochs_range, val_acc, label='Validation Set')
        plt.legend(loc="best")
        plt.xlabel('Epochs')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Train Set')
        plt.plot(epochs_range, val_loss, label='Validation Set')
        plt.legend(loc="best")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.suptitle(f"{model_name_str}")
        plt.show()

    @staticmethod
    def compare_detrend_wave_data(wave_data_array, sample_rate=666):
        """去除基线漂移: 基线漂移是一种低频噪声成分, 频率通常小于1Hz, 使用巴特沃斯高通过滤"""
        assert len(wave_data_array.shape) == 2
        start_time = time.time()

        #  1. 0.5-50Hz 四阶巴特沃斯带通过滤器
        LOW_CUT = 0.5
        HIGH_CUT = 100
        fn = 0.5 * sample_rate  # 最大频率
        f_low = LOW_CUT / fn
        f_high = HIGH_CUT / fn

        method_name = f"{LOW_CUT}Hz-{HIGH_CUT}Hz bandpass"
        freq_window = [f_low, f_high]
        b, a = butter(N=4, Wn=freq_window, btype="bandpass")
        filtered_signal = filtfilt(b, a, wave_data_array)
        print(f"filtered_signal.shape={filtered_signal.shape}， filtered_signal={filtered_signal}")

        # 2. 1Hz 巴特沃斯高通
        # method_name = f"{LOW_CUT}Hz HighPass"
        # freq_window = f_low
        # b, a = butter(N=4, Wn=freq_window, btype="highpass")
        #
        # filtered_signal = filtfilt(b, a, wave_data_array)
        # print(f"filtered_signal.shape={filtered_signal.shape}， filtered_signal={filtered_signal}")

        # # 3. 去除 50Hz工频噪声 + 中值滤波去曲线漂移
        # method_name=f"50Hz LowPass + Medfilt"
        # fn = 0.5 * sample_rate
        # f_low = HIGH_CUT / fn
        # b, a = butter(N=4, Wn=f_low, btype='lowpass')
        # filtered_signal = filtfilt(b, a, wave_data_array)
        # # 去除基线漂移
        # ks = int(sample_rate * 0.5)
        # if ks % 2 == 0:
        #     ks -= 1
        # baseline = medfilt(filtered_signal[0,:], kernel_size=ks)
        # filtered_signal = filtered_signal - baseline
        # print(f"filtered_signal.shape={filtered_signal.shape},costs={time.time() - start_time}s")

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(wave_data_array[0, :], 'r-', label=f"origin data")
        plt.subplot(2, 1, 2)
        plt.plot(filtered_signal[0, :], 'b-', label=f"detrend data")
        plt.suptitle(method_name)

        plt.show()


class PulsePreprocessing:
    def __init__(self):
        self._test_num_each_class = 5

    @staticmethod
    def detrend_sigle_wave_data(wave_data_array, current_sample_rate=666) -> np.ndarray:
        """
        去除基线漂移: 基线漂移是一种低频噪声成分, 频率通常小于1Hz, 使用巴特沃斯高通过滤
        :param wave_data_array:
        :param current_sample_rate:
        :return:
        """
        assert len(wave_data_array.shape) == 2, f"wave_data_array shape={wave_data_array.shape} not match!"
        start_time = time.time()

        #  1. 0.5-100Hz 四阶巴特沃斯带通过滤器
        LOW_CUT = 0.5
        HIGH_CUT = 100
        fn = 0.5 * current_sample_rate  # 最大频率
        f_low = LOW_CUT / fn
        f_high = HIGH_CUT / fn

        method_name = f"{LOW_CUT}Hz-{HIGH_CUT}Hz bandpass"
        freq_window = [f_low, f_high]
        b, a = butter(N=4, Wn=freq_window, btype="bandpass")
        filtered_signal = filtfilt(b, a, wave_data_array)
        # print(f"filtered_signal.shape={filtered_signal.shape}， filtered_signal={filtered_signal}")

        # # 2. 1Hz 巴特沃斯高通
        # method_name = f"{LOW_CUT}Hz HighPass"
        # freq_window = f_low
        # b, a = butter(N=4, Wn=freq_window, btype="highpass")
        # filtered_signal = filtfilt(b, a, wave_data_array)
        # # print(f"filtered_signal.shape={filtered_signal.shape}， filtered_signal={filtered_signal}")

        return filtered_signal

    @staticmethod
    def get_wave_peak_bottom_index(wave_data_array, current_sample_rate=666):
        # TODO: pip install biosppy
        assert len(wave_data_array.shape) == 2, f"shape not match"
        peak_result = ecg.christov_segmenter(wave_data_array[0,], sampling_rate=current_sample_rate)
        rpeaks_result = peak_result['rpeaks']
        # get bottom
        peak_result = ecg.christov_segmenter(wave_data_array[0,] * (-1), sampling_rate=current_sample_rate)
        bottom_result = peak_result['rpeaks']
        return rpeaks_result, bottom_result

    @staticmethod
    def copy_file_to_another_dir(source_path, save_path) -> Tuple[bool, str]:
        try:
            if not Path(source_path).is_file():
                raise Exception(f"原始文件不存在！source_path={source_path}")

            shutil.copy(source_path, save_path)
            return True, save_path

        except Exception as e:
            state_message = f"转移文件失败！message={str(e)}"
            logger.exception(state_message)
            return False, state_message

    @staticmethod
    def plot_wave_data(origin_pulse_array, downsampled_pulse_array, time_len: int):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(origin_pulse_array, 'r-')
        plt.title(f"origin freq={origin_pulse_array.shape[0] / time_len} Hz")
        plt.subplot(2, 1, 2)
        plt.plot(downsampled_pulse_array, "b-")
        plt.title(f"downsample freq={downsampled_pulse_array.shape[0] / time_len} Hz")
        plt.grid(which="minor", linestyle='-.', axis='both')
        plt.show()

    @staticmethod
    def save_npy_file(pulse_data_dict, save_filename="train_test_data.npz"):
        save_path = pulse_dataset_dir.joinpath(save_filename).as_posix()
        np.savez(save_path, pulse_data=pulse_data_dict)
        print(f"保存npz 文件成功！save_path={save_path}")

    @staticmethod
    def load_npy_saved_pulse_data(npz_save_path: str) -> Dict[int, Dict[str, List[dict]]]:
        """
        load from npz file
            Pulse data description: In total 988 pulse samples, pulse's time info={'60.0s': 899, '6.0s': 89}: pulse data
           have 2 kinds of times info. And pulse data have sparse pressure info.
        :param npz_save_path:
        :return: pulse data dict-> Dict[ key=class->int, value=Dict[key in ('test','train'), value= signle_data_dict] ]
            single_data_dict=Dict['t':int, 'p':float, 'data':numpy array]
        """
        npz_result = np.load(npz_save_path, allow_pickle=True)
        pulse_all_class_data_dict = npz_result['pulse_data'].item()
        return pulse_all_class_data_dict

    @staticmethod
    def load_one_txt_data(data_filepath: str) -> Tuple[bool, Union[str, dict]]:
        """
        读取 txt 波形数据到 pandas.DataFrame
        :param data_filepath: 读取 txt 文件中数据
        :return: Dict[ 't': 秒数->float, 'p': 压力值->float, "data":时序波形数据->ndarray, 'filepath': str]
        """
        try:
            pulse_wave_data_list = []
            with open(data_filepath, "r+") as fp:
                for line_no, data_str in enumerate(fp.readlines()):
                    data_str = data_str.replace("\n", "")
                    if line_no == 0:
                        t_pulse, p_pulse = re.findall(r"\d+.\d*", data_str)[:2]
                    else:
                        pulse_wave_data_list.append(eval(data_str))

            return True, {"t": t_pulse, "p": p_pulse, "data": np.array([pulse_wave_data_list]),
                          'filepath': Path(data_filepath).name}
        except Exception as e:
            logger.exception(f"load_one_txt_data Failed! data_filepath={data_filepath}")
            return False, str(e)

    def split_origin_train_test_data(self) -> Tuple[bool, Union[dict, str]]:
        """
        分割 train、test 数据，存入字典，保存 npy 文件
        :return: data dict-> Dict[class_label, Dict[test:[], train:[]] ]
        """
        pulse_data_dict = {}
        for class_name in os.listdir(pulse_all_data_path.as_posix()):
            class_label = pulse_class_name_list.index(class_name)
            print(f"class={class_name}, class_label={class_label}")
            if class_name not in pulse_class_name_list:
                raise Exception(f"unsupported class type!class_name={class_name}")

            class_dir_path = pulse_all_data_path.joinpath(class_name)
            class_data_path_list = [class_dir_path.joinpath(item_filename).as_posix()
                                    for item_filename in os.listdir(class_dir_path.as_posix())]
            if not class_data_path_list:
                raise Exception(f"class={class_name} with empty data!")
            # 0. shuffle sequence
            random.seed(1)  # 设置随机种子，使得每次运行结果可复现
            random.shuffle(class_data_path_list)
            # 1. split train test
            test_filename_list = class_data_path_list[:self._test_num_each_class]
            train_filename_list = class_data_path_list[self._test_num_each_class:]

            # 2. extract data
            pulse_data_dict[class_label] = {"train": [], "test": []}
            for data_filepath in test_filename_list:
                success_flag, data_info_result = \
                    self.load_one_txt_data(data_filepath=data_filepath)
                if not success_flag:
                    raise Exception(data_info_result)
                pulse_data_dict[class_label]["test"].append(data_info_result)

            for data_filepath in train_filename_list:
                success_flag, data_info_result = \
                    self.load_one_txt_data(data_filepath=data_filepath)
                if not success_flag:
                    raise Exception(data_info_result)
                pulse_data_dict[class_label]["train"].append(data_info_result)

            logger.info(f"class={class_name}, label={class_label}: train_test_split success!"
                        f"train_num={len(train_filename_list)}, test_num={len(test_filename_list)}")
        # save npy file
        self.save_npy_file(pulse_data_dict)
        print(f"train_test_split finish!")
        return True, pulse_data_dict

    @staticmethod
    def downsample_pulse_data(wave_data_array, origin_time_length, target_sample_rate):
        """对数据进行重采样"""
        assert isinstance(wave_data_array, np.ndarray) and len(wave_data_array.shape) == 2
        # 1. 均匀采样
        time_index_list = np.linspace(start=0, stop=wave_data_array.shape[1],
                                      num=math.ceil(origin_time_length * target_sample_rate), endpoint=False, dtype=int)
        # downsampled_data = wave_data_array[:, time_index_list]
        # print(f"shape={wave_data_array.shape},wave_data_list={wave_data_array}")
        # print(f"shape={wave_data_array.shape}, time_index_list={time_index_list}")
        # print(f"shape={downsampled_data.shape}, downsampled_data={downsampled_data}, type={type(downsampled_data)}")

        # 1.2 np.interp
        downsampled_data = np.interp(time_index_list, np.arange(0, wave_data_array.shape[1]), wave_data_array[0])
        downsampled_data = downsampled_data.reshape((1, downsampled_data.shape[0]))
        print(f"wave_data_list={wave_data_array}, shape={wave_data_array.shape}")
        print(f"downsampled_data={downsampled_data}, shape={downsampled_data.shape}")

        # # 2. scipy signal
        # current_sample_rate = int(wave_data_array.shape[1] / origin_time_length)
        # downsampled_data = resample_poly(wave_data_array[0, :], up=target_sample_rate, down=current_sample_rate)
        # downsampled_data = downsampled_data.reshape((1, downsampled_data.shape[0]))
        # print(f"shape={wave_data_array.shape},wave_data_list={wave_data_array}")
        # print(f"shape={downsampled_data.shape}, downsampled_data={downsampled_data}, type={type(downsampled_data)}")

        return downsampled_data

    def unify_sample_rate(self, pulse_all_class_data_dict, target_sample_rate=666):
        """
        统一sample rate, 全部降采样到 666 Hz （采样率的最小值）
        :param pulse_all_class_data_dict:
        :param target_sample_rate:
        :return:
        """
        for class_no in pulse_all_class_data_dict:
            for type_str in ["train", "test"]:
                for data_index in range(len(pulse_all_class_data_dict[class_no][type_str])):
                    data_info_dict = pulse_all_class_data_dict[class_no][type_str][data_index]
                    wave_data_array = data_info_dict['data']
                    down_sample_data = self.downsample_pulse_data(wave_data_array=data_info_dict["data"],
                                                                  origin_time_length=eval(data_info_dict['t']),
                                                                  target_sample_rate=target_sample_rate)
                    pulse_all_class_data_dict[class_no][type_str][data_index]["data"] = down_sample_data
                    # self.plot_wave_data(wave_data_array[0,], down_sample_data[0,], time_len=eval(data_info_dict['t']))

            print(f"class_no={class_no} 降采样完成！")
        return pulse_all_class_data_dict

    def get_wave_peak_clip_data(self, wave_data_array, current_sample_rate=666, clip_length=600,
                                center_delta=20, clip_step=1):
        """获取波峰片段"""
        assert len(wave_data_array.shape) == 2, f"wave_data_array {wave_data_array.shape} shape not match!"
        assert clip_length > 0

        peak_clip_array = []
        clip_left_size = int(clip_length / 2)
        peak_index_list, _ = self.get_wave_peak_bottom_index(wave_data_array, current_sample_rate=current_sample_rate)
        for peak_index in peak_index_list:
            left_pos_list = [index for index in range(peak_index - center_delta - clip_left_size,
                                                      peak_index + center_delta - clip_left_size + 1)
                             if index >= 0 and index + clip_length <= wave_data_array.shape[1]]
            # 边缘不取
            if not left_pos_list:
                continue

            # 存在满足的片，按 clip_step 取片
            left_pos_set = set(left_pos_list[0::clip_step])
            for left_pos in left_pos_set:
                right_pos = left_pos + clip_length
                assert left_pos >= 0 and right_pos <= wave_data_array.shape[1], f"clip left、right pos 越界！"

                peak_clip_array.append([wave_data_array[0, left_pos:right_pos]])

        return np.array(peak_clip_array)

    def get_train_test_data_slice(self, pulse_all_data_dict, frame_length=600, frame_step=350):
        """
        对原始时序数据按 步长、帧移 分片，获取模型输入 x_train, y_train
        :param pulse_all_data_dict: Dict[class_no->int, key=Dict[key in ('train','test'), value=List[data_info_dict]] ]
            data_info_dict -> Dict['t'->str, 'p'->str, 'data'->np.ndarray(shape=(1,n))]
        :param frame_length:
        :param frame_step:
        :return: x_train->np.ndarray , shape= (n, frame_length), y_train->np.ndarray, shape=(n,1)
        """
        train_data_dict = {"x_train": [], "y_train": [], "x_test": [], "y_test": []}
        # class_frame_step_dict = {6: int(frame_step / 0.36), 5: int(frame_step / 0.81), 4: int(frame_step / 1.2),
        #                          3: int(frame_step / 5), 2: int(frame_step / 3), 1: int(frame_step / 4),
        #                          0: int(frame_step / 1.7)}
        class_frame_step_dict = {6: min(frame_length, frame_length - 100), 5: frame_step, 4: frame_step,
                                 3: int(frame_step / 5), 2: int(frame_step / 3), 1: frame_step, 0: frame_step}
        minority_class_no_set = {1, 2, 3, 4, 5}
        origin_frame_step = frame_step
        for class_no in pulse_all_data_dict:
            for data_type_str in ["train", "test"]:
                # 少数类,增加分片数, 多数类减少分片数
                frame_step = class_frame_step_dict[class_no] if data_type_str == "train" else origin_frame_step

                for data_no, data_info_dict in enumerate(pulse_all_data_dict[class_no][data_type_str]):
                    # 分片
                    # 0. 帧移均匀选片
                    wave_data_array = data_info_dict["data"][0,]  # 原始 shape=(1,n)
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

                    # # 1. add precise wave peak clip: time costly!
                    # current_sample_rate = int(data_info_dict["data"].shape[1] / eval(data_info_dict["t"]))
                    # precise_peak_clip_array = self.get_wave_peak_clip_data(data_info_dict["data"],
                    #                                       current_sample_rate=current_sample_rate,
                    #                                       clip_length=frame_length,
                    #                                       center_delta=0, clip_step=1)
                    # print(f"class_no={class_no},data_type_str={data_type_str}:"
                    #       f"data_no={data_no}/{len(pulse_all_data_dict[class_no][data_type_str])}, frames shape={frames.shape}")
                    # # frames = np.concatenate((frames, precise_peak_clip_array), axis=0)

                    # add to x_train, y_train
                    train_data_dict[f"x_{data_type_str}"].extend(frames.tolist())
                    train_data_dict[f"y_{data_type_str}"].extend([int(class_no)] * frames.shape[0])

                print(f"class_no={class_no}, data_type_str={data_type_str} 分片完成！")

        x_train = np.array(train_data_dict["x_train"])
        y_train = np.array(train_data_dict["y_train"]).reshape((-1, 1))
        x_test = np.array(train_data_dict["x_test"])
        y_test = np.array(train_data_dict["y_test"]).reshape((-1, 1))

        print(f"x_train shape={x_train.shape}")
        print(f"y_train shape={y_train.shape}")
        print(f"x_test shape={x_test.shape}")
        print(f"y_test shape={y_test.shape}")

        return x_train, y_train, x_test, y_test

    def detrend_wave_data(self, pulse_all_data_dict):
        for class_no in pulse_all_data_dict:
            for type_str in ["train", "test"]:
                for data_index in range(len(pulse_all_data_dict[class_no][type_str])):
                    data_info_dict = pulse_all_data_dict[class_no][type_str][data_index]
                    current_sample_rate = int(data_info_dict["data"].shape[1] / eval(data_info_dict['t']))
                    detrend_wave_data = self.detrend_sigle_wave_data(data_info_dict["data"],
                                                                     current_sample_rate=current_sample_rate)
                    pulse_all_data_dict[class_no][type_str][data_index]["data"] = detrend_wave_data

            print(f"class_no={class_no} 去基线漂移完成！")
        return pulse_all_data_dict


class PulseFeature:
    def __init__(self):
        pass

    @staticmethod
    def standardize_wave_data(wave_data):
        wave_mean = np.mean(wave_data, axis=1)
        wave_std = np.std(wave_data, axis=1)
        return (wave_data - wave_mean) / wave_std

    @staticmethod
    def plot_spectrogram(spec, save_path):
        fig = plt.figure(figsize=(20, 5))
        heatmap = plt.pcolor(spec)
        fig.colorbar(mappable=heatmap)
        plt.xlabel('Frames')
        # tight_layout 会自动调整子图参数，使之填充整个图像区域
        # plt.tight_layout()
        plt.title(save_path.stem)
        plt.savefig(save_path)
        plt.show()

    @staticmethod
    def get_scipy_spectrogram(wave_data, sample_rate, frame_size=0.025, frame_stride=0.01, filename=""):
        freq_array, time_segment, Spectrogram_x = spectrogram(x=wave_data[0, :], fs=sample_rate)
        plt.figure(1)
        plt.pcolormesh(time_segment, freq_array, Spectrogram_x, shading="gouraud")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    @staticmethod
    def get_frames(wave_data, sample_rate, frame_size=0.025, frame_stride=0.01):
        """
        预加重, 分帧, 幅值归一化
        :param wave_data: ndarray, shape=(1,n)
        :param sample_rate: 采样率
        :param frame_size: 帧长 默认 25 ms
        :param frame_stride: 帧移 默认 10 ms
        :return:
        """
        # 1. 预加重: 补偿高频部分振幅
        pre_emphasis = 0.97
        emphasized_signal = np.append(
            wave_data[0, 0], wave_data[:, 1:] - pre_emphasis * wave_data[:, -1])

        # 2. 分帧
        frame_length, frame_step = int(
            round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
        signal_length = len(emphasized_signal)
        num_frames = int(
            np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1
        pad_signal_length = (num_frames - 1) * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        # 分帧后最后一帧点数不足，则补零
        # 获取帧：frames 是二维数组，每一行是一帧，列数是每帧的采样点数，之后的短时 fft 直接在每一列上操作
        pad_signal = np.append(emphasized_signal, z)
        indices = np.arange(0, frame_length).reshape(
            1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1, 1)
        frames = pad_signal[indices]
        print("shape of frames:", frames.shape)

        # 3. 加 hamming 窗
        frames *= np.hamming(frame_length)
        return frames

    def save_fbank_mfcc_feature(self, wave_data, sample_rate, frame_size=0.025, frame_stride=0.01, save_name=""):
        """
        提取 Fbank、MFCC 特征: 自己编写的
        :param wave_data:
        :param sample_rate:
        :param frame_size:
        :param frame_stride:
        :param save_name:
        :return:
        """
        # 分帧, 加窗
        frames = self.get_frames(wave_data=wave_data, sample_rate=sample_rate,
                                 frame_size=frame_size, frame_stride=frame_stride)
        print(f"frames shape={frames.shape}, {frames}")
        # FFT
        NFFT = 512  # 频谱系数的点数
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # 频谱幅值
        pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # 能量频谱
        print(f"FFT result: pow_frames shape={pow_frames.shape}, pow_frames={pow_frames}")

        # Fbank
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
        # mel 滤波器中心点：为了方便后面计算 mel 滤波器组，左右两边各补一个中心点
        nfilt = 200  # Mel 滤波器个数
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)

        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))  # 各个 mel 滤波器在能量谱对应点的取值
        bin = (hz_points / (sample_rate / 2)) * \
              (NFFT / 2)  # 各个 mel 滤波器中心点对应 FFT 的区域编码，找到有值的位置
        # 计算 mel 滤波器函数: 矩阵运算更快
        for i in range(1, nfilt + 1):
            left = int(bin[i - 1])
            center = int(bin[i])
            right = int(bin[i + 1])
            for j in range(left, center):
                fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
            for j in range(center, right):
                fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])
                # print(fbank)
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)  # dB
        # print(filter_banks)
        print("shape of FBank", filter_banks.shape)
        print(f"filter_banks={filter_banks}")
        self.plot_spectrogram(filter_banks.T, feature_result_dir.joinpath(f"Fbank_{save_name}"))

        # MFCC
        num_ceps = 12  # 保留的倒谱系数的个数
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')  # shape of MFCC=(570 帧数, 200 滤波器个数)
        mfcc_feature_result = mfcc[:, 1:(num_ceps + 1)]  # 选取 2-13 维度
        print(f"shape of MFCC={mfcc.shape}, mfcc={mfcc}")
        self.plot_spectrogram(mfcc_feature_result.T, feature_result_dir.joinpath(f"MFCC_{save_name}"))

        # Normalization
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        self.plot_spectrogram(filter_banks.T, feature_result_dir.joinpath(f"Fbank_normalized_{save_name}"))
        mfcc_feature_result -= (np.mean(mfcc_feature_result, axis=0) + 1e-8)
        self.plot_spectrogram(mfcc_feature_result.T, feature_result_dir.joinpath(f"MFCC_normalized_{save_name}"))

    @staticmethod
    def get_dynamic_mfcc_matrix(wave_data, sample_rate, mfcc_normalization=False,
                                winlen=0.025, winstep=0.01, numcep=13, nfilt=200, winfunc=np.hamming):
        """
        第三方开源模块 python_speech_features, 提取 动态 mfcc 特征： 静态+一、二阶差分
        :param wave_data:
        :param sample_rate:
        :param mfcc_normalization:
        :param winlen:
        :param winstep:
        :param numcep:
        :param nfilt:
        :param winfunc:
        :return:
        """
        assert isinstance(wave_data, np.ndarray) and len(wave_data.shape) == 2, f"wave_data 格式错误！"

        mfcc_feature_result = mfcc(signal=wave_data.T, samplerate=sample_rate, winlen=winlen, winstep=winstep,
                                   numcep=numcep, nfilt=nfilt, winfunc=winfunc)
        # print(f"mfcc: static feature result={mfcc_feature_result.shape}, {mfcc_feature_result}")

        # 动态特征：计算一阶、二阶差分, 对每个特征向量进行延长（1*13-> 1*39)
        d_mfcc_feat = delta(mfcc_feature_result, 1)  # 每行为一个特征向量
        d_mfcc_feat2 = delta(mfcc_feature_result, 2)
        mfcc_final_result = np.hstack((mfcc_feature_result, d_mfcc_feat, d_mfcc_feat2))

        # MFCC, Fbank features: normalization
        if mfcc_normalization:
            mfcc_final_result -= (np.mean(mfcc_final_result, axis=0) + 1e-8)

        return mfcc_final_result

    @staticmethod
    def get_fbank_matrix(wave_data, sample_rate, winlen=0.025, winstep=0.01, nfilt=200, winfunc=np.hamming):
        """
        第三方开源模块 python_speech_features, 提取 fbank 特征
        :param wave_data:
        :param sample_rate:
        :param winlen:
        :param winstep:
        :param nfilt:
        :param winfunc:
        :return:
        """
        assert isinstance(wave_data, np.ndarray) and len(wave_data.shape) == 2, f"wave_data 格式错误！"

        fbank_result, energy_each_frame = fbank(signal=wave_data.T, samplerate=sample_rate, winlen=winlen,
                                                winstep=winstep,
                                                nfilt=nfilt, winfunc=winfunc)  # 未取对数
        return np.log(fbank_result)

    def get_train_test_data_mfcc_feature(self, x_data, sample_rate=666.0, numcep=20):
        """
        每片序列数据转换为 mfcc 或 fbank 矩阵
        :param x_data:
        :param get_dynamic_mfcc_matrix_func:
        :param sample_rate:
        :param numcep:
        :return:
        """
        assert isinstance(x_data, np.ndarray) and len(x_data.shape) == 2

        mfcc_feature_data = []
        for data_index in range(x_data.shape[0]):
            wave_data_array = np.reshape(x_data[data_index, :], (1, -1))
            freq_feature_result = self.get_dynamic_mfcc_matrix(wave_data=wave_data_array, sample_rate=sample_rate,
                                                               numcep=numcep)
            mfcc_feature_data.append(freq_feature_result)

        mfcc_feature_data = np.array(mfcc_feature_data, dtype=np.float32)
        return mfcc_feature_data

    def get_train_test_data_fbank_feature(self, x_data, sample_rate=666.0):
        """
        每片序列数据转换为 mfcc 或 fbank 矩阵
        :param x_data:
        :param sample_rate:
        :return:
        """
        assert isinstance(x_data, np.ndarray) and len(x_data.shape) == 2

        fbank_feature_data = []
        for data_index in range(x_data.shape[0]):
            wave_data_array = np.reshape(x_data[data_index, :], (1, -1))
            freq_feature_result = self.get_fbank_matrix(wave_data=wave_data_array, sample_rate=sample_rate)
            fbank_feature_data.append(freq_feature_result)
            if int(data_index % 20) == 0:
                print(f"fbank 特征转换: No.{data_index}/{x_data.shape[0]}")

        fbank_feature_data = np.array(fbank_feature_data, dtype=np.float16)
        return fbank_feature_data

    def convert_save_fbank_feature(self, x_data, y_data, fbank_save_dir_path: str,
                                   sample_rate=666.0, filename_prefix="x_train") \
            -> Tuple[bool, list]:
        """
        每片序列数据转换为 fbank 矩阵 , 单独存储为 npy 文件
        :param x_data:
        :param y_data
        :param fbank_save_dir_path
        :param sample_rate:
        :param filename_prefix:
        :return: 返回文件路径列表
        """
        assert isinstance(x_data, np.ndarray) and len(x_data.shape) == 2
        assert x_data.shape[0] == y_data.shape[0]
        fbank_save_dir_path = Path(fbank_save_dir_path)
        assert fbank_save_dir_path.exists(), f"Fbank 保存路径不存在！{fbank_save_dir_path}"

        start_time = time.time()
        fbank_data_path_list = []
        for data_index in range(x_data.shape[0]):
            wave_data_array = np.reshape(x_data[data_index, :], (1, -1))
            freq_feature_result = self.get_fbank_matrix(wave_data=wave_data_array, sample_rate=sample_rate)
            # save npy file
            fbank_save_name = f"{filename_prefix}_{data_index}_{int(y_data[data_index, 0])}"
            fbank_save_path = fbank_save_dir_path.joinpath(fbank_save_name).as_posix()
            np.save(fbank_save_path, freq_feature_result)
            fbank_data_path_list.append(fbank_save_path)

            if data_index % 1500 == 0:
                print(
                    f"{filename_prefix} 存储 Fbank 特征：No.{data_index}/{x_data.shape[0]} costs={int(time.time() - start_time) / 60}min")

        return True, fbank_data_path_list


def zip_directory(source_dir_path, target_zip_path):
    import zipfile
    import os
    from pathlib import Path

    assert os.path.exists(source_dir_path), f"source={source_dir_path} not exists!"
    with zipfile.ZipFile(target_zip_path, "a") as zipfp:
        for parent_dir, _, filename_list in os.walk(source_dir_path):
            for origin_filename in filename_list:
                origin_filepath_tmp = Path(parent_dir).joinpath(origin_filename)
                zipfp.write(origin_filepath_tmp.as_posix(), origin_filepath_tmp.name)
    print(f"source_dir_path={source_dir_path}, target_zip_path={target_zip_path} 压缩完成！")


if __name__ == '__main__':
    pulse_preprocess = PulsePreprocessing()
    # 0. LOAD PULSE DATA
    # _success_flag, pulse_all_class_data_dict = pulse_preprocess.split_origin_train_test_data()
    # 原始数据
    origin_npz_filename = "train_test_data.npz"
    npz_save_path_str = pulse_dataset_dir.joinpath(origin_npz_filename).as_posix()
    assert Path(npz_save_path_str).is_file(), f"文件不存在, filepath={npz_save_path_str}"
    origin_pulse_all_class_data_dict = pulse_preprocess.load_npy_saved_pulse_data(npz_save_path=npz_save_path_str)

    # 降采样数据
    downsampled_npz_filename = "downsampled_train_test_dict.npz"
    sample_rate = 666.0  # 统一的采样率
    npz_save_path_str = pulse_dataset_dir.joinpath(downsampled_npz_filename).as_posix()
    assert Path(npz_save_path_str).is_file(), f"文件不存在, filepath={npz_save_path_str}"
    pulse_all_class_data_dict = pulse_preprocess.load_npy_saved_pulse_data(npz_save_path=npz_save_path_str)

    # ------------------------------------------原始 vs 降采样------------------------------------------------------------
    ShowResult.print_data_info(pulse_all_class_data_dict)
    ShowResult.compare_downsample_effect(origin_pulse_all_class_data_dict)
    del origin_pulse_all_class_data_dict

    # -----------------------------------------数据预处理----------------------------------------
    # # unify sample rate
    # pulse_all_class_data_dict = pulse_preprocess.unify_sample_rate(pulse_all_class_data_dict,
    #                                                                target_sample_rate=int(sample_rate))
    # pulse_preprocess.save_npy_file(pulse_all_class_data_dict, save_filename="downsampled_train_test_dict.npz")

    # # 去基线漂移+去噪
    # start_time = time.time()
    # pulse_all_class_data_dict = pulse_preprocess.detrend_wave_data(pulse_all_class_data_dict)
    # print(f"去噪+去基线漂移完成！costs={time.time() - start_time}s")
    #
    # # 分片
    # start_time = time.time()
    # heart_beat_average_seconds = 1.2  # 按平均一次心跳的秒数分片
    # frame_length = math.ceil(heart_beat_average_seconds * sample_rate) + 40  # 840
    # frame_step = 280  # int(frame_length/3)
    # x_train, y_train, x_test, y_test = pulse_preprocess.get_train_test_data_slice(
    #     pulse_all_class_data_dict, frame_length=frame_length, frame_step=frame_step)
    # print(f"get data slices costs={time.time() - start_time}s")  # 9s

    # ----------------------------------------查看每类分片信息: tips: watch out the data imbalance problem
    # # 检查每一类分片数
    # # 原始脉诊样本
    # class_data_info_dict = {
    #     class_no: (len(pulse_all_class_data_dict[class_no]["train"]), len(pulse_all_class_data_dict[class_no]["test"]))
    #     for class_no in pulse_all_class_data_dict}
    # print(class_data_info_dict)
    #
    # # train、test、validate 中的各类分布
    # train_distribution_dict = {class_no: np.sum(y_train == class_no) for class_no in class_data_info_dict}
    # print(f"x_train distribution:{train_distribution_dict}")

    # # 检查分片效果
    # ShowResult.show_slice_effect(x_train[:10, ], y_train[:10, ])

    # ------------------获取频域特征---------------------------------------------------------------------
    # # 1. 获取 MFCC 特征
    pulse_feature = PulseFeature()
    # start_time = time.time()
    # mfcc_feature_x_train = pulse_feature.get_train_test_data_mfcc_feature(x_data=x_train,
    #                                                                       sample_rate=sample_rate, numcep=20)
    # print(f"mfcc_feature_x_train: mfcc 特征转换成功！costs={time.time() - start_time}s")
    # mfcc_feature_x_test = pulse_feature.get_train_test_data_mfcc_feature(x_data=x_test,
    #                                                                      sample_rate=sample_rate, numcep=20)
    # print(f"mfcc_feature_x_train shape={mfcc_feature_x_train.shape}, "
    #       f"mfcc_feature_x_test shape={mfcc_feature_x_test.shape}, costs={time.time() - start_time}s")
    # mfcc_pulse_data_dict = {"x_train": mfcc_feature_x_train, "y_train": y_train,
    #                         "x_test": mfcc_feature_x_test, "y_test": y_test}
    # mfcc_pulse_data_npz_filename = f"mfcc_pulse_data_{frame_length}x{frame_step}.npz"
    # # 保存 npz 文件
    # pulse_preprocess.save_npy_file(pulse_data_dict=mfcc_pulse_data_dict, save_filename=mfcc_pulse_data_npz_filename)
    # del mfcc_pulse_data_dict, mfcc_feature_x_train, mfcc_feature_x_test
    #

    # 2. 获取 Fbank 特征
    # start_time = time.time()
    # # 2.1 load into memory directly
    # fbank_feature_x_train = pulse_feature.get_train_test_data_fbank_feature(x_data=x_train, sample_rate=sample_rate)
    # print(f"fbank_feature_x_train: fbank 特征转换成功！costs={time.time() - start_time}s")
    # fbank_feature_x_test = pulse_feature.get_train_test_data_fbank_feature(x_data=x_test, sample_rate=sample_rate)
    # print(f"fbank_feature_x_train shape={fbank_feature_x_train.shape}, "
    #       f"fbank_feature_x_test shape={fbank_feature_x_test.shape}, costs={time.time() - start_time}s")
    # fbank_pulse_data_dict = {"x_train": fbank_feature_x_train, "y_train": y_train,
    #                          "x_test": fbank_feature_x_test, "y_test": y_test}
    # fbank_pulse_data_npz_filename = f"fbank_pulse_data_{frame_length}x{frame_step}.npz"
    # # 保存 npz 文件
    # pulse_preprocess.save_npy_file(pulse_data_dict=fbank_pulse_data_dict, save_filename=fbank_pulse_data_npz_filename)
    # del fbank_pulse_data_dict, fbank_feature_x_train, fbank_feature_x_test

    # # 2.2 每个分片转换为 Fbank 后单独保存 npy 文件, 使用 Sequence 方式，在训练时分批读取数据
    # start_time = time.time()
    # fbank_data_name = f"Fbank_{int(sample_rate)}_{frame_length}x{frame_step}"
    # fbank_save_dir = pulse_dataset_dir.joinpath(fbank_data_name)
    # fbank_save_dir.mkdir(exist_ok=True, parents=True)
    #
    # _, x_train_path_list = pulse_feature.convert_save_fbank_feature(x_train[:10], y_train[:10],
    #                                                                 fbank_save_dir_path=fbank_save_dir,
    #                                                                 sample_rate=sample_rate,
    #                                                                 filename_prefix="train")
    # print(f"fbank_feature_x_train: fbank 特征转换成功！costs={time.time() - start_time}s")
    # _, x_test_path_list = pulse_feature.convert_save_fbank_feature(x_test[:10], y_test[:10],
    #                                                                fbank_save_dir_path=fbank_save_dir,
    #                                                                sample_rate=sample_rate,
    #                                                                filename_prefix="test")
    # print(f"fbank_feature_x_test: fbank 特征转换成功！total costs={time.time() - start_time}s")
    # fbank_pulse_data_dict = {"x_train_path_list": x_train_path_list, "y_train": y_train,
    #                          "x_test_path_list": x_test_path_list, "y_test": y_test}
    # pulse_preprocess.save_npy_file(pulse_data_dict=fbank_pulse_data_dict, save_filename=fbank_data_name)
    # del fbank_pulse_data_dict

    # ----------------- 测试声学特征 ------------------------------------------------------------------------
    # test_class_no, test_sample_index = 0, 1
    # test_pulse_data_dict = pulse_all_class_data_dict[test_class_no]["train"][test_sample_index]
    # print(f"test_pulse_data_dict={test_pulse_data_dict}, data shape={test_pulse_data_dict['data'].shape}")
    # wave_data = test_pulse_data_dict["data"][:, :]
    # sample_time = float(test_pulse_data_dict['t'])
    # sample_rate = test_pulse_data_dict["data"].shape[1] / sample_time
    # print(f"sample_rate={sample_rate}, data shape={test_pulse_data_dict['data'].shape}")
    # wave_data = (wave_data - np.min(wave_data)) / (np.max(wave_data) - np.min(wave_data))
    # print(f"wave_data={wave_data},wave_data.shape={wave_data.shape}, ")
    # pulse_preprocess.plot_wave_data(wave_data[0, :], wave_data[0, :], eval(test_pulse_data_dict['t']))

    # # 1.1 MFCC, Fbank
    # pulse_feature = PulseFeature()
    # # # self design
    # # pulse_feature.save_fbank_mfcc_feature(wave_data=wave_data, sample_rate=sample_rate,
    # #                                       save_name=f"class{test_class_no}_index{test_sample_index}")
    # # third party module
    # start_time = time.time()
    # mfcc_final_result = pulse_feature.get_dynamic_mfcc_matrix(wave_data=wave_data, sample_rate=sample_rate)
    # print(f"dynamic MFCC shape={mfcc_final_result.shape}, costs={time.time() - start_time}s")  # 0.01s
    # pulse_feature.plot_spectrogram(
    #     mfcc_final_result.T, feature_result_dir.joinpath(f"MFCC_dynamic_class={test_class_no}-{test_sample_index}"))
    #
    # start_time = time.time()
    # fbank_result = pulse_feature.get_fbank_matrix(wave_data=wave_data, sample_rate=sample_rate)
    # print(f"fbank_resul shape={fbank_result.shape}, costs={time.time() - start_time}s")  # 0.005s
    # pulse_feature.plot_spectrogram(
    #     fbank_result.T, feature_result_dir.joinpath(f"fbank_result_class={test_class_no}-{test_sample_index}"))
