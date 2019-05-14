from __future__ import division
import tensorflow as tf
from ops import *
from utils import *
from XceptionNetworksBig import XceptionModel
from UnetNetworks3 import make_stochastic_unet

def utility_model(input_image, reuse=False, is_training=False,nu=None,use_bn=True):
    with tf.variable_scope("utility") as scope:
        if reuse:
            scope.reuse_variables()
        return XceptionModel(input_image, nu, is_training = is_training, data_format='channels_last',
                             name_prefix='u_',  use_bn=use_bn)
    
def utility_raw_model(input_image, reuse=False, is_training=False,nu=None,use_bn=True):
    with tf.variable_scope("utility_raw") as scope:
        if reuse:
            scope.reuse_variables()
        return XceptionModel(input_image, nu, is_training = is_training, data_format='channels_last',
                             name_prefix='u_',  use_bn=use_bn)
    
def secret_model(input_image, reuse=False, is_training=False,ns=None,use_bn=True):
    with tf.variable_scope("secret") as scope:
        if reuse:
            scope.reuse_variables()
        return XceptionModel(input_image, ns, is_training = is_training, data_format='channels_last',
                             name_prefix='s_',  use_bn=use_bn)
    
def filter_model(input_image, reuse=False, is_training=False,use_bn=True, nout=1):
    with tf.variable_scope("filter") as scope:
        if reuse:
            scope.reuse_variables()
        return make_stochastic_unet(input_image, is_training, nout=input_image.shape[-1], use_bn=use_bn)
    
def combined_model(input_image, is_training=False,nu=None,ns=None, use_bn_f=True, use_bn_c = True):
    
    filt_img = filter_model(input_image, True, is_training,use_bn=use_bn_f)
    u_pred = utility_model(filt_img, True, False,use_bn=use_bn_c,nu=nu)
    s_pred = secret_model(filt_img, True, False,use_bn=use_bn_c,ns=ns)
    
    return [u_pred, s_pred]
