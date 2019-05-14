"""Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import copy
import tensorflow as tf
import matplotlib.pyplot as plt

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


def save_everything(ganClass, savepath, dg, nu, ns, n_val=None, load_checkpoint=True):
    def computeDKL(p, q):
        p = np.clip(np.array(p), 1e-7, 1)
        q = np.clip(np.array(q), 1e-7, 1)
        dKL = np.sum(p * np.log(p / q), axis=1)
        return dKL

    def computeMI(p, q):
        dKL = computeDKL(p, q)
        dKL = np.mean(dKL)
        return dKL

    ## Save Losses and MIs
    tr_loss = ganClass.stored_filtered_train_losses[0:ganClass.nplot_val, :]
    val_loss = ganClass.stored_filtered_val_losses[0:ganClass.nplot_val, :]
    u_prior = ganClass.utility_prior
    s_prior = ganClass.secret_prior

    ### Load filter and classifiers ###
    if load_checkpoint:
        ganClass.load(ganClass.checkpoint_dir, ganClass.pretrain_checkpoint_dir,
                      load_type='train', budget_val=ganClass.budget_val)

    ### TEST ###
    if n_val is None:
        n_val = len(dg)

    bs = dg.batch_size
    pSY = np.zeros([n_val * bs, ns])
    pUY = np.zeros([n_val * bs, nu])
    pUX = np.zeros([n_val * bs, nu])
    gtU = np.zeros([n_val * bs, nu])
    gtS = np.zeros([n_val * bs, ns])

    count = 0
    for idx in np.arange(n_val):
        # Get data batch
        data_batch = dg.__getitem__(idx)
        imgs = data_batch[0][0]
        utility_gt_idx = data_batch[1][0]
        secret_gt_idx = data_batch[1][1]

        # Evaluate raw image
        pUX_aux = ganClass.sess.run(ganClass.utility_raw_output,
                           feed_dict={ganClass.raw_input_image_ph: imgs})

        # Get filtered images
        gen_imgs = ganClass.sess.run(ganClass.filter_output, feed_dict={ganClass.input_image_ph: imgs})
        pUY_aux, pSY_aux = ganClass.sess.run([ganClass.utility_output, ganClass.secret_output],
                                    feed_dict={ganClass.raw_input_image_ph: gen_imgs})

        # Predictions
        pSY[count:int(count + pUY_aux.shape[0]), :] = pSY_aux
        pUY[count:int(count + pUY_aux.shape[0]), :] = pUY_aux
        pUX[count:int(count + pUY_aux.shape[0]), :] = pUX_aux
        gtU[count:int(count + pUY_aux.shape[0]), :] = utility_gt_idx
        gtS[count:int(count + pUY_aux.shape[0]), :] = secret_gt_idx
        count += pUY_aux.shape[0]

    pSY = pSY[:count, ...]
    pUY = pUY[:count, ...]
    pUX = pUX[:count, ...]
    gtU = gtU[:count, ...]
    gtS = gtS[:count, ...]
    u_prior = u_prior[0, :]
    s_prior = s_prior[0, :]

    save_data = {}
    save_data['tr_loss'] = tr_loss
    save_data['val_loss'] = val_loss
    save_data['pSY'] = pSY
    save_data['pUY'] = pUY
    save_data['pUX'] = pUX
    save_data['gtU'] = gtU
    save_data['gtS'] = gtS
    save_data['u_prior'] = u_prior
    save_data['s_prior'] = s_prior

    print('top 5 acc filtered : ', get_categorical_performance(pUY, gtU, ntop=5))
    print('top 5 acc original : ', get_categorical_performance(pUX, gtU, ntop=5))
    print('acc gender : ', get_categorical_performance(pSY, gtS, ntop=1))

    import pickle
    with open(savepath, 'wb') as f:
        pickle.dump(save_data, f)
    return

    plt.figure(figsize=(10, 5))
    plt.plot(tr_loss[:, 0])
    plt.plot(val_loss[:, 0])
    plt.show()


def pred_network(output, input_ph, pred_dim, dg, n_val=None, flag=0,sess):
    #     dg.shuffle()
    if n_val is None:
        n_val = len(dg)
    bs = dg.batch_size

    prediction = np.zeros([n_val * bs, pred_dim])
    gt_labels = np.zeros([n_val * bs, pred_dim])
    count = 0
    for idx in np.arange(n_val):
        # get data batch
        data_batch = dg.__getitem__(idx)
        imgs = data_batch[0][0]
        gt = data_batch[1][flag]  # utility flag = 0 // secret flag = 1

        # Get filtered images
        feed_dict = {input_ph: imgs}
        pred = sess.run(output, feed_dict=feed_dict)

        # storing and book-keeping
        # classifier losses [u,ur,s]
        prediction[count:int(count + pred.shape[0]), :] = pred
        gt_labels[count:int(count + pred.shape[0]), :] = gt
        count += pred.shape[0]
    prediction = prediction[:count, ...]
    gt_labels = gt_labels[:count, ...]

    return prediction, gt_labels


def pred_ganclass_network(ganClass, pred_dim, dg, filtro=True, n_val=None, flag=0):
    # dg.shuffle()
    if n_val is None:
        n_val = len(dg)
    bs = dg.batch_size

    prediction = np.zeros([n_val * bs, pred_dim])
    gt_labels = np.zeros([n_val * bs, pred_dim])
    count = 0
    for idx in np.arange(n_val):
        # get data batch
        data_batch = dg.__getitem__(idx)
        imgs = data_batch[0][0]
        gt = data_batch[1][flag]  # utility flag = 0 // secret flag = 1

        if filtro:
            gen_imgs = ganClass.sess.run(ganClass.filter_output, feed_dict={ganClass.input_image_ph: imgs})
        else:
            gen_imgs = imgs

        if flag == 0:
            pred = ganClass.sess.run(ganClass.utility_output,
                            feed_dict={ganClass.raw_input_image_ph: gen_imgs})
        else:
            pred = ganClass.sess.run(ganClass.secret_output,
                            feed_dict={ganClass.raw_input_image_ph: gen_imgs})

        # storing and book-keeping
        # classifier losses [u,ur,s]
        prediction[count:int(count + pred.shape[0]), :] = pred
        gt_labels[count:int(count + pred.shape[0]), :] = gt
        count += pred.shape[0]
    prediction = prediction[:count, ...]
    gt_labels = gt_labels[:count, ...]

    return prediction, gt_labels