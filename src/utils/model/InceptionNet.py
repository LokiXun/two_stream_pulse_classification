# encoding: utf-8
"""
Function: InceptionNet, 目前仅取了 1 个Inception Block 组成的 Inception10 模型
@author: LokiXun
@contact: 2682414501@qq.com
"""
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, GlobalAveragePooling2D, \
    Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, Bidirectional, Attention

from utils.CBAM_module import CbamModule, cbam_module


# 搭建 Inception 结构
class ConvBNRelu(Model):
    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        """
        整合 CBA 卷积、批标准化、激活层操作
        :param ch:卷积核的个数（目标输出的深度）
        :param kernelsz:
        :param strides:
        :param padding: 默认为”全0填充“ 使得 输出特征图大小和原来一致
        """
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(ch, kernelsz, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        x = self.model(x, training=False)
        # training 决定 BN 操作进行归一化的数据范围
        #   training=False时，BN 通过整个训练集计算均值、方差去做批归一化
        #   training=True时，通过当前batch的均值、方差去做批归一化。
        # 推理时 training=False效果好
        return x


class InceptionBlk(Model):
    # Inception 基本单元中，每一个 Conv，均为 CBA
    # 每一块均用 padding 全 0 填充，使得输入输出大小一致
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides, padding='same')

        self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides, padding='same')
        self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1, padding='same')

        self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides, padding='same')
        self.c3_2 = ConvBNRelu(ch, kernelsz=5, strides=1, padding='same')

        self.p4_1 = MaxPool2D(3, strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernelsz=1, strides=strides, padding='same')

    def call(self, x):
        x1 = self.c1(x)

        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1)

        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)

        x4_1 = self.p4_1(x)
        x4_2 = self.c4_2(x4_1)
        # concat along axis=channel
        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)
        return x


class Inception10(Model):
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
        """

        :param num_blocks: 1个block 由 2个Inception基本单元组成
        :param num_classes:
        :param init_ch: 第一层 Conv 输出特征图深度
        :param kwargs: key=value 形式传入
        """
        super(Inception10, self).__init__(**kwargs)
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_blocks = num_blocks
        self.init_ch = init_ch
        self.c1 = ConvBNRelu(init_ch)

        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                # 每个 block 2个 Inception 块
                # out_channel == Inception 基本单元模块的卷积核个数
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2)
                    # strides 步长=2，使得输出特征图大小为原来一半
                else:
                    block = InceptionBlk(self.out_channels, strides=1)
                self.blocks.add(block)
            # enlarger out_channels per block
            self.out_channels *= 2
            # 图像大小减半（信息减少了）要增加之后输出图的深度，保持信息承载能力

        self.p1 = GlobalAveragePooling2D()
        self.f1 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        # CBAM
        x = cbam_module(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


if __name__ == '__main__':
    # 对于 globalAvgPool 之前加入了 CBAM
    model = Inception10(num_blocks=2, num_classes=7)
