"""Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import copy
import tensorflow as tf

try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread

'''
def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False):
    img = imread(image_path)
    if not is_testing:
        img = scipy.misc.imresize(img, [load_size, load_size])
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img = img[h1:h1+fine_size, w1:w1+fine_size]

        if np.random.random() > 0.5:
            img = np.fliplr(img)
    else:
        img = scipy.misc.imresize(img, [fine_size, fine_size])

    img = img/127.5 - 1.

    return img

def load_test_data(image_path, fine_size=256):
    img = imread(image_path)
    img = scipy.misc.imresize(img, [fine_size, fine_size])
    img = img/127.5 - 1
    return img

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return _imread(path, flatten=True).astype(np.float)
    else:
        return _imread(path, mode='RGB').astype(np.float)
'''

def count_parameters(var_list):
    total_parameters = 0
    for variable in var_list:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def get_categorical_performance(subject_E,subject_GT,ntop = 5):
    cont = 0
    for i in np.arange(subject_E.shape[0]):
        vector_E = np.zeros(subject_E[i,:].shape)
        vector_E[np.argsort(subject_E[i,:])[-ntop:]] = 1
        cont += np.sum(vector_E*subject_GT[i,:])
    return cont/subject_GT.shape[0]

def get_data_marginals(df_train, nu,ns):
    #Compute data marginals
    Pus_m = np.zeros([nu,ns])
    for u in np.arange(nu):
        for s in np.arange(ns):
            Pus_m[u,s] = df_train.loc[(df_train['utility'] == u) & (df_train['secret'] == s)].shape[0]

    Pus_m /= Pus_m.sum()
    Pus = Pus_m.flatten()

    Pu = Pus_m.sum(1)
    Ps = Pus_m.sum(0)
    return Pus, Pu,Ps

def computeDKL(p,q):
    p = np.clip(np.array(p),1e-7,1)
    q = np.clip(np.array(q),1e-7,1)
    dKL = np.sum(p*np.log(p/q), axis=1)
    return dKL

def computeMI(p,q):
    dKL = computeDKL(p,q)
    dKL = np.mean(dKL)
    return dKL

#LOSSES
def KL_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-5, 1)
    y_truea = tf.clip_by_value(y_true, 1e-5, 1)
    dKL =tf.reduce_sum(y_true * tf.log(y_truea / y_pred), axis=-1)
    I = tf.reduce_mean(dKL)
    return I

def penalty_loss(y_true, y_pred, budget):
    y_pred = tf.clip_by_value(y_pred, 1e-5, 1)
    y_truea = tf.clip_by_value(y_true, 1e-5, 1)
    dKL =tf.reduce_sum(y_true * tf.log(y_truea / y_pred), axis=-1)
    I = tf.reduce_mean(dKL)
    L = tf.square(tf.nn.relu(I-budget))
    return L

def combined_loss_fn(utililty,secret_prior, combined_output, budget_ph, lambda_ph):
    utility_loss_term = KL_loss(utililty, combined_output[0])
    penalty_secret_loss = penalty_loss(combined_output[1], secret_prior,  budget_ph)
    L = utility_loss_term + lambda_ph*penalty_secret_loss
    return L , utility_loss_term, penalty_secret_loss