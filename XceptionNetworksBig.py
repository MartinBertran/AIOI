import tensorflow as tf
import numpy as np
from ops import *

USE_FUSED_BN = True
BN_EPSILON = 0.001
BN_MOMENTUM = 0.99

'''snippet from inception V3 in slim
'''
def reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [
            min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])
        ]
    return kernel_size_out

def batch_norm_(x, name, training=True, reuse=None, axis=-1,use_bn=True):
    if use_bn:
        return batch_norm(x, name, training=True, reuse=None, axis=-1)
    else:
        return instance_norm(x,name)

def relu_separable_bn_block(inputs, filters, name_prefix, is_training, data_format, use_bn = True):
    bn_axis = -1 if data_format == 'channels_last' else 1

    inputs = tf.nn.relu(inputs, name=name_prefix + '_act')
    inputs = tf.layers.separable_conv2d(inputs, filters, (3, 3),
                        strides=(1, 1), padding='same',
                        data_format=data_format,
                        activation=None, use_bias=False,
                        depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                        pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name=name_prefix, reuse=None)
    
    inputs =  batch_norm_(inputs, name=name_prefix + '_bn', axis=bn_axis,training=is_training,reuse=None, use_bn=use_bn)

#     inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name=name_prefix + '_bn', axis=bn_axis,
#                             epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    return inputs

"""Contains the definition for Xception V1 classification network."""
def XceptionModel(input_image, num_classes, is_training = False, data_format='channels_last', name_prefix='', use_bn=True):
    bn_axis = -1 if data_format == 'channels_last' else 1

    
        
    
    # Entry Flow
    inputs = tf.layers.conv2d(input_image, 32, (3, 3), use_bias=False, name=name_prefix+'block1_conv1', strides=(2, 2),
                padding='valid', data_format=data_format, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer())
    
    
    inputs =  batch_norm_(inputs, name=name_prefix+'block1_conv1_bn', axis=bn_axis,training=is_training,reuse=None, use_bn=use_bn)
    
#     inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name=name_prefix+'block1_conv1_bn', axis=bn_axis,
#                             epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name=name_prefix+'block1_conv1_act')

    inputs = tf.layers.conv2d(inputs, 64, (3, 3), use_bias=False, name=name_prefix+'block1_conv2', strides=(1, 1),
                padding='valid', data_format=data_format, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer())
    
    inputs =  batch_norm_(inputs, name=name_prefix+'block1_conv2_bn', axis=bn_axis,training=is_training,reuse=None, use_bn=use_bn)
#     inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name=name_prefix+'block1_conv2_bn', axis=bn_axis,
#                             epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name=name_prefix+'block1_conv2_act')

    residual = tf.layers.conv2d(inputs, 128, (1, 1), use_bias=False, name=name_prefix+'conv2d_1', strides=(2, 2),
                padding='same', data_format=data_format, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer())
    
    residual =  batch_norm_(residual, name=name_prefix+'batch_normalization_1', axis=bn_axis,training=is_training,reuse=None, use_bn=use_bn)
#     residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name=name_prefix+'batch_normalization_1', axis=bn_axis,
#                             epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = tf.layers.separable_conv2d(inputs, 128, (3, 3),
                        strides=(1, 1), padding='same',
                        data_format=data_format,
                        activation=None, use_bias=False,
                        depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                        pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name=name_prefix+'block2_sepconv1', reuse=None)
    inputs =  batch_norm_(inputs, name=name_prefix+'block1_sepconv1_bn', axis=bn_axis,training=is_training,reuse=None, use_bn=use_bn)
#     inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name=name_prefix+'block2_sepconv1_bn', axis=bn_axis,
#                             epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 128, name_prefix+'block2_sepconv2', is_training, data_format, use_bn=use_bn)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                    padding='same', data_format=data_format,
                                    name=name_prefix+'block2_pool')

    inputs = tf.add(inputs, residual, name=name_prefix+'residual_add_0')
    residual = tf.layers.conv2d(inputs, 128, (1, 1), use_bias=False, name=name_prefix+'conv2d_2', strides=(2, 2),
                padding='same', data_format=data_format, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer())
    
    residual =  batch_norm_(residual, name=name_prefix+'batch_normalization_2', axis=bn_axis,training=is_training,reuse=None, use_bn=use_bn)
#     residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name=name_prefix+'batch_normalization_2', axis=bn_axis,
#                             epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 128, name_prefix+'block3_sepconv1', is_training, data_format, use_bn=use_bn)
    inputs = relu_separable_bn_block(inputs, 128, name_prefix+'block3_sepconv2', is_training, data_format, use_bn=use_bn)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                    padding='same', data_format=data_format,
                                    name=name_prefix+'block3_pool')
    inputs = tf.add(inputs, residual, name=name_prefix+'residual_add_1')

    residual = tf.layers.conv2d(inputs, 256, (1, 1), use_bias=False, name=name_prefix+'conv2d_3', strides=(2, 2),
                padding='same', data_format=data_format, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer())
    residual =  batch_norm_(residual, name=name_prefix+'batch_normalization_3', axis=bn_axis,training=is_training,reuse=None, use_bn=use_bn)
#     residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name=name_prefix+'batch_normalization_3', axis=bn_axis,
#                             epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 256, name_prefix+'block4_sepconv1', is_training, data_format, use_bn=use_bn)
    inputs = relu_separable_bn_block(inputs, 256, name_prefix+'block4_sepconv2', is_training, data_format, use_bn=use_bn)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                    padding='same', data_format=data_format,
                                    name=name_prefix+'block4_pool')
    inputs = tf.add(inputs, residual, name=name_prefix+'residual_add_2')
    # Middle Flow
    for index in range(8):
        residual = inputs
        prefix = name_prefix+'block' + str(index + 5)

        inputs = relu_separable_bn_block(inputs, 256, prefix + '_sepconv1', is_training, data_format, use_bn=use_bn)
        inputs = relu_separable_bn_block(inputs, 256, prefix + '_sepconv2', is_training, data_format, use_bn=use_bn)
        inputs = relu_separable_bn_block(inputs, 256, prefix + '_sepconv3', is_training, data_format, use_bn=use_bn)
        inputs = tf.add(inputs, residual, name=prefix + '_residual_add')
    # Exit Flow
    residual = tf.layers.conv2d(inputs, 512, (1, 1), use_bias=False, name=name_prefix+'conv2d_4', strides=(2, 2),
                padding='same', data_format=data_format, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer())
    residual =  batch_norm_(residual, name=name_prefix+'batch_normalization_4', axis=bn_axis,training=is_training,reuse=None, use_bn=use_bn)
#     residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name=name_prefix+'batch_normalization_4', axis=bn_axis,
#                             epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 512, name_prefix+'block13_sepconv1', is_training, data_format, use_bn=use_bn)
    inputs = relu_separable_bn_block(inputs, 512, name_prefix+'block13_sepconv2', is_training, data_format, use_bn=use_bn)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                    padding='same', data_format=data_format,
                                    name=name_prefix+'block13_pool')
    inputs = tf.add(inputs, residual, name=name_prefix+'residual_add_3')

    inputs = tf.layers.separable_conv2d(inputs, 728, (3, 3),
                        strides=(1, 1), padding='same',
                        data_format=data_format,
                        activation=None, use_bias=False,
                        depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                        pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name=name_prefix+'block14_sepconv1', reuse=None)
    inputs =  batch_norm_(inputs, name=name_prefix+'block14_sepconv1_bn', axis=bn_axis,training=is_training,reuse=None, use_bn=use_bn)
#     inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name=name_prefix+'block14_sepconv1_bn', axis=bn_axis,
#                             epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name=name_prefix+'block14_sepconv1_act')

    inputs = tf.layers.separable_conv2d(inputs, 728, (3, 3),
                        strides=(1, 1), padding='same',
                        data_format=data_format,
                        activation=None, use_bias=False,
                        depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                        pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name=name_prefix+'block14_sepconv2', reuse=None)
    inputs =  batch_norm_(inputs, name=name_prefix+'block14_sepconv2_bn', axis=bn_axis,training=is_training,reuse=None, use_bn=use_bn)

#     inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name=name_prefix+'block14_sepconv2_bn', axis=bn_axis,
#                             epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name=name_prefix+'block14_sepconv2_act')

    if data_format == 'channels_first':
        channels_last_inputs = tf.transpose(inputs, [0, 2, 3, 1])
    else:
        channels_last_inputs = inputs

    inputs = tf.layers.average_pooling2d(inputs, pool_size = reduced_kernel_size_for_small_input(channels_last_inputs, [10, 10]), strides = 1, padding='valid', data_format=data_format, name=name_prefix+'avg_pool')

    if data_format == 'channels_first':
        inputs = tf.squeeze(inputs, axis=[2, 3])
    else:
        inputs = tf.squeeze(inputs, axis=[1, 2])

    outputs = tf.layers.dense(inputs, num_classes,
                            activation=tf.nn.softmax, use_bias=True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer(),
                            name=name_prefix+'dense', reuse=None)

    return outputs