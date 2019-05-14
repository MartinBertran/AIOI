"""
Adapted from https://github.com/kkweon/UNet-in-Tensorflow/blob/master/train.py

"""

import time
import os
import pandas as pd
import tensorflow as tf

from ops import *
from utils import *

def conv_conv_pool(input_,
                   n_filters,
                   is_training,
                   #flags,
                   name,
                   pool=True,
                   activation=tf.nn.leaky_relu,
                  use_bn=True):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activation functions
        use_bn: True/False use batch_norm or instance_norm
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F, (3, 3),
                activation=None,
                padding='same',
                #kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
                name="conv_{}".format(i + 1))
            if use_bn:
                net = tf.layers.batch_normalization(
                    net, training=is_training, name="bn_{}".format(i + 1))
            else:
                net = instance_norm(net, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool

def upconv_concat_MB(inputA, input_B, n_filter, name):
    """Upsample `inputA` and concat with `input_B`
    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = upconv_2D_MB(inputA, n_filter, name)

    return tf.concat(
        [up_conv, input_B], axis=-1, name="concat_{}".format(name))

def upconv_2D_MB(tensor, n_filter, name):
    """Up SAMPLING `tensor` by 2 times
    Args:
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations
    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    return tf.keras.layers.UpSampling2D(size=(2,2),
        name="upsample_{}".format(name))(tensor)
    
    
def upconv_concat(inputA, input_B, n_filter, name):
    """Upsample `inputA` and concat with `input_B`
    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = upconv_2D(inputA, n_filter, name)

    return tf.concat(
        [up_conv, input_B], axis=-1, name="concat_{}".format(name))


def upconv_2D(tensor, n_filter, name):
    """Up Convolution `tensor` by 2 times
    Args:
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations
    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    return tf.layers.conv2d_transpose(
        tensor,
        filters=n_filter,
        kernel_size=2,
        strides=2,
        #kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
        name="upsample_{}".format(name))


def make_unet(X, is_training):
    """Build a U-Net architecture
    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers
    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor
    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """

    net = X / 127.5 - 1
    conv1, pool1 = conv_conv_pool(net, [8, 8], is_training,  name=1)
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], is_training,  name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], is_training, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], is_training,  name=4)
    conv5 = conv_conv_pool(
        pool4, [128, 128], is_training,  name=5, pool=False)
    
    #stochastic noise layer
    conv5 =  conv5 + tf.random_normal(tf.shape(conv5), 0, 1, dtype=tf.float32)

    up6 = upconv_concat(conv5, conv4, 64, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], is_training,  name=6, pool=False)

    up7 = upconv_concat(conv6, conv3, 32,  name=7)
    conv7 = conv_conv_pool(up7, [32, 32], is_training,  name=7, pool=False)

    up8 = upconv_concat(conv7, conv2, 16,  name=8)
    conv8 = conv_conv_pool(up8, [16, 16], is_training,  name=8, pool=False)

    up9 = upconv_concat(conv8, conv1, 8,  name=9)
    conv9 = conv_conv_pool(up9, [8, 8], is_training,  name=9, pool=False)

    return tf.layers.conv2d(
        conv9,
        1, (1, 1),
        name='final',
        activation=tf.nn.sigmoid,
        padding='same')



def make_stochastic_unet(X, is_training,nout=1, use_bn=False):
    """Build a U-Net architecture
    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers
    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor
    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """

    net = X
    conv1, pool1 = conv_conv_pool(net, [16,16], is_training,  name=1, use_bn=use_bn)
    conv2, pool2 = conv_conv_pool(pool1, [32,32], is_training,  name=2, use_bn=use_bn)
    conv3, pool3 = conv_conv_pool(pool2, [64,64], is_training, name=3, use_bn=use_bn)
    conv4, pool4 = conv_conv_pool(pool3, [128,128], is_training,  name=4, use_bn=use_bn)
    conv5 = conv_conv_pool(pool4, [256,256], is_training,  name=5, pool=False, use_bn=use_bn)
    
    #stochastic noise layer
#     conv5 =  conv5 + tf.random_normal(tf.shape(conv5), 0, 1, dtype=tf.float32)
    conv5 = tf.concat([conv5, tf.random_normal(tf.shape(conv5), 0, 1, dtype=tf.float32)],axis = -1)
    

    up6 = upconv_concat_MB(conv5, conv4, 128, name=6)
    conv6 = conv_conv_pool(up6, [128,128], is_training,  name=6, pool=False, use_bn=use_bn)

    up7 = upconv_concat_MB(conv6, conv3, 64,  name=7)
    conv7 = conv_conv_pool(up7, [64,64], is_training,  name=7, pool=False, use_bn=use_bn)

    up8 = upconv_concat_MB(conv7, conv2, 32,  name=8)
    conv8 = conv_conv_pool(up8, [32,32], is_training,  name=8, pool=False, use_bn=use_bn)

    up9 = upconv_concat_MB(conv8, conv1, 16, name=9)
    conv9 = conv_conv_pool(up9, [16,16], is_training,  name=9, pool=False, use_bn=use_bn)

    return tf.layers.conv2d(
        conv9,
        nout, (1, 1),
        name='final',
        activation=tf.nn.sigmoid,
        padding='same')


