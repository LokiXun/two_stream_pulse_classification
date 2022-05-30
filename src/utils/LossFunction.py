# encoding: utf-8
"""
Function: 自定义损失函数： Focal Loss, PolyLoss
@author: LokiXun
@contact: 2682414501@qq.com
"""
import tensorflow as tf


def multi_category_focal_loss1_fixed(y_true, y_pred, gamma=2, class_num=7):
    epsilon = 1.e-7
    y_true = tf.one_hot(y_true, depth=class_num)
    alpha = tf.constant([[1], [1], [1], [1], [1], [1], [1]], dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
    ce = -tf.math.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.matmul(tf.multiply(weight, ce), alpha)
    loss = tf.reduce_mean(fl)
    return loss


def poly1_cross_entropy(y_true, y_pred, epsilon=1.0, class_num=7):
    """y_pred already is softmax result"""
    # pt, CE, and Poly1 have shape [batch].
    y_true_onehot = tf.one_hot(y_true, depth=class_num)
    pt = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)
    CE = tf.nn.softmax_cross_entropy_with_logits(y_true_onehot, y_pred)
    Poly1 = CE + epsilon * (1 - pt)
    return Poly1
