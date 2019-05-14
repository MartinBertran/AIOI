import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output
#from keras.utils import to_categorical
import pickle

import warnings
warnings.filterwarnings('ignore')

def computeDKL(p,q):
    p = np.clip(np.array(p),1e-7,1)
    q = np.clip(np.array(q),1e-7,1)
    dKL = np.sum(p*np.log(p/q), axis=1)
    return dKL

def computeMI(p,q):
    dKL = computeDKL(p,q)
    dKL = np.mean(dKL)
    return dKL


class AIOIGAN():
    def __init__(
        self, data_generator,
        utility_output, utility_loss, utility_acc, utility_solver,
        utility_raw_output, utility_raw_loss, utility_raw_acc, utility_raw_solver,
        secret_output, secret_loss, secret_acc, secret_solver,
        filter_output, filter_loss, filter_solver,
        combined_loss, utility_combined_loss, secret_combined_loss, combined_solver,
        budget_ph, lambda_ph, lr_u_ph, lr_u_raw_ph, lr_s_ph, lr_f_ph, lr_c_ph,
        input_image, raw_input_image, utility_gt_ph, secret_gt_ph, secret_prior_ph,
        utility_saver, utility_raw_saver, secret_saver, filter_saver,
        sess,lambda_val,budget_val,
        n_f=1,n_s=1,n_u=1, lr_u=1e-2,lr_s=1e-2,lr_f=1e-2,lr_c=1e-2, lr_u_r = 1e-4,
        dg_val=None,Sprior = None,Uprior = None,
#         checkpoint_f=None, checkpoint_u=None, checkpoint_s=None,
        model_name='anon', checkpoint_dir=None, pretrain_checkpoint_dir=None,
    scalar_summary=None, image_summary=None,
    use_batchnorm_filter=False, use_batchnorm_classifiers=False):
        
        
        self.n_f=n_f
        self.n_s=n_s
        self.n_u=n_u
        self.lr_u = lr_u
        self.lr_s = lr_s
        self.lr_f = lr_f
        self.lr_c = lr_c
        self.lr_u_raw_ph =lr_u_r
        self.lambda_val=lambda_val
        self.budget_val = budget_val
        self.model_name =model_name
        self.use_batchnorm_filter = use_batchnorm_filter
        self.use_batchnorm_classifiers=use_batchnorm_classifiers

        self.data_generator = data_generator
        self.data_len = len(data_generator)
        self.scalar_summary = scalar_summary
        self.image_summary = image_summary
        self.logdir = './logs/{:s}_BN_filter_{:}_BN_classifier_{:}/'.format(
            self.model_name, self.use_batchnorm_filter, self.use_batchnorm_classifiers)

#         self.logdir='./logs/{:s}/'.format(self.model_name)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        
        ######################################
        #
        # all tensorflow objects I care about
        #
        ######################################
        
        self.sess = sess
        
        self.utility_output = utility_output
        self.utility_loss = utility_loss
        self.utility_acc = utility_acc
        self.utility_solver = utility_solver
        
        
        self.utility_raw_output = utility_raw_output
        self.utility_raw_loss = utility_raw_loss
        self.utility_raw_acc = utility_raw_acc
        self.utility_raw_solver = utility_raw_solver
        
        
        self.secret_output = secret_output
        self.secret_loss = secret_loss
        self.secret_acc = secret_acc
        self.secret_solver = secret_solver
        
        self.filter_output = filter_output
        self.filter_loss = filter_loss
        self.filter_solver = filter_solver      
        
        
        self.combined_loss = combined_loss
        self.utility_combined_loss = utility_combined_loss
        self.secret_combined_loss = secret_combined_loss
        self.combined_solver = combined_solver
        
        
        ######################################
        #
        # All placeholders we need
        #
        ######################################
        
        self.budget_ph = budget_ph
        self.lambda_ph =  lambda_ph
        self.lr_u_ph = lr_u_ph
        self.lr_u_raw_ph =lr_u_raw_ph
        self.lr_s_ph = lr_s_ph
        self.lr_f_ph = lr_f_ph
        self.lr_c_ph = lr_c_ph
        
        self.input_image_ph = input_image
        self.raw_input_image_ph = raw_input_image
        self.utility_gt_ph = utility_gt_ph
        self.secret_gt_ph = secret_gt_ph
        self.secret_prior_ph =  secret_prior_ph
        
        
        ######################################
        #
        # All savers
        #
        ######################################
        
        self.utility_saver = utility_saver
        self.utility_raw_saver = utility_raw_saver
        self.secret_saver = secret_saver
        self.filter_saver = filter_saver
        
        #validation set stuff
        self.dg_val =dg_val
        self.best_eval = np.inf
        
        self.model_name=model_name
        self.checkpoint_dir = checkpoint_dir
        self.pretrain_checkpoint_dir = pretrain_checkpoint_dir
        
        

        self.secret_prior = np.tile(Sprior,(self.dg_val.batch_size,1))
        self.utility_prior = np.tile(Uprior,(self.dg_val.batch_size,1))


    def train(self,epochs, sample_interval=50, eval_interval=1000,save_image = False):
        n_batches = self.data_len

        self.stored_secret_losses = np.zeros([epochs*n_batches, 4])
        self.stored_utility_losses = np.zeros([epochs*n_batches, 4])
        self.stored_filtered_losses = np.zeros([epochs*n_batches, 4])

        self.stored_secret_val_losses = np.zeros([epochs*n_batches, 4])
        self.stored_utility_val_losses = np.zeros([epochs*n_batches, 4])
        self.stored_filtered_val_losses = np.zeros([epochs*n_batches, 4])

        self.stored_secret_train_losses = np.zeros([epochs*n_batches, 4])
        self.stored_utility_train_losses = np.zeros([epochs*n_batches, 4])
        self.stored_filtered_train_losses = np.zeros([epochs*n_batches, 4])

        self.stored_infos = np.zeros([epochs*n_batches,4])

        self.nplot_val = 0
        
        self.writer = tf.summary.FileWriter(self.logdir+"train_{:.3f}".format(self.budget_val))
        self.val_writer = tf.summary.FileWriter(self.logdir+"val_train_{:.3f}".format(self.budget_val))

        counter=-1
        for epoch in range(epochs):
            self.data_generator.shuffle()
            for idx in np.arange(self.data_len):
                counter+=1
                # ---------------------
                # Get data batch
                # ---------------------
                data_batch = self.data_generator.__getitem__(idx)
                imgs = data_batch[0][0]
                utility_gt = data_batch[1][0]  # GT!
                secret_gt = data_batch[1][1]  # GT!



                # ---------------------
                #  Train filter
                # ---------------------
                feed_dict = {self.raw_input_image_ph: imgs}
                pred_u_raw = self.sess.run(self.utility_raw_output, feed_dict=feed_dict)


                feed_dict = {self.input_image_ph: imgs, self.lr_c_ph: self.lr_c,
                             self.utility_gt_ph: pred_u_raw, self.secret_prior_ph: self.secret_prior[:imgs.shape[0],...],
                             self.budget_ph : self.budget_val, self.lambda_ph:self.lambda_val}
                for _ in np.arange(self.n_f):
                    trash= self.sess.run(self.combined_solver, feed_dict=feed_dict)

                # ---------------------
                #  Train Utility and Secret
                # ---------------------
                # Generate a batch of new images

                feed_dict = {self.input_image_ph: imgs}
                gen_imgs = self.sess.run(self.filter_output, feed_dict=feed_dict)


                feed_dict = {self.raw_input_image_ph: gen_imgs, self.secret_gt_ph : secret_gt, self.lr_s_ph : self.lr_s }
                # Train the secret

                for _ in np.arange(self.n_s):
                    trash, s_loss, s_acc = self.sess.run([self.secret_solver,self.secret_loss,self.secret_acc],
                                                             feed_dict=feed_dict)
                self.stored_secret_losses[counter, 2:] = [s_loss, s_acc]


                # Train the utility
                feed_dict = {self.raw_input_image_ph: gen_imgs, self.utility_gt_ph : utility_gt, self.lr_u_ph : self.lr_u }
                if self.n_u ==0:
                    u_loss, u_acc = self.sess.run([self.utility_loss,self.utility_acc],
                                                                 feed_dict=feed_dict)
                for _ in np.arange(self.n_u):
                    trash, u_loss, u_acc = self.sess.run([self.utility_solver,self.utility_loss,self.utility_acc],
                                                                 feed_dict=feed_dict)

                feed_dict = {self.raw_input_image_ph: imgs, self.utility_gt_ph : utility_gt, self.lr_u_ph : self.lr_u }
                u_raw_loss, u_raw_acc = self.sess.run([self.utility_raw_loss,self.utility_raw_acc],
                                                                  feed_dict=feed_dict)

                self.stored_utility_losses[counter, :] = [u_raw_loss, u_raw_acc, u_loss, u_acc]


                # ---------------------
                #  Test filter
                # ---------------------

                feed_dict= {self.input_image_ph: imgs, self.lr_c_ph: self.lr_c,
                                        self.utility_gt_ph: pred_u_raw, self.secret_prior_ph: self.secret_prior[:imgs.shape[0],...],
                                        self.budget_ph : self.budget_val, self.lambda_ph:self.lambda_val}
                comb_loss_v, util_loss_v, sec_loss_v = self.sess.run(
                                [self.combined_loss, self.utility_combined_loss , self.secret_combined_loss], feed_dict=feed_dict)

                self.stored_filtered_losses[counter, :3] = [comb_loss_v, util_loss_v, sec_loss_v]
                
                # ---------------------
                #  Write summary
                # ---------------------
                feed_dict={
                        self.scalar_summary['utility_acc_ph']:u_acc, self.scalar_summary['utility_loss_ph']:u_loss,
                        self.scalar_summary['utility_raw_acc_ph']:u_raw_acc, self.scalar_summary['utility_raw_loss_ph']:u_raw_loss,
                        self.scalar_summary['secret_acc_ph']:s_acc, self.scalar_summary['secret_loss_ph']:s_loss,
                        self.scalar_summary['filter_CIB_loss_ph']:comb_loss_v,
                        self.scalar_summary['filter_censored_info_loss_ph']:util_loss_v,
                        self.scalar_summary['filter_information_penalty_loss_ph']: sec_loss_v}
                scalar_sum = self.sess.run(
                    self.scalar_summary['merged_training_losses'], 
                    feed_dict=feed_dict
                )
                self.writer.add_summary(scalar_sum, counter)
                image_sum = self.sess.run(
                    self.image_summary['merged_image_summary'], 
                    feed_dict={self.input_image_ph: imgs}
                )
                self.writer.add_summary(image_sum, counter)

                if eval_interval is not None:
                    if counter % eval_interval == eval_interval-1:
                        self.test_validation(eval_interval)

#                 if counter % sample_interval == sample_interval-1:
#                     self.plot_losses(counter)

                if (counter % (sample_interval*5) == 0) & save_image: 
                    self.sample_images(counter)
                


    def sample_images(self, epoch):
        r, c = 4, 4
        idx = np.random.randint(self.data_len)
        data_batch = self.data_generator.__getitem__(idx)

        imgs = data_batch[0][0]
        feed_dict = {self.input_image_ph: imgs}
        gen_imgs = self.sess.run(self.filter_output, feed_dict=feed_dict)

        # Rescale images 0 - 1
        #         gen_imgs = 0.5 * gen_imgs + 0.5
        plt.figure()
        fig, axs = plt.subplots(r, c, figsize=(12,12))
        cnt = 0
        for i in range(r):
            for j in range(c):
                aux = gen_imgs[cnt, :, :, 0]
                aux = aux[20:-20,20:-20]
                aux -=aux.min()
                aux/=aux.max()
                axs[i, j].imshow(aux, cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/budget_{:.2e}_{:06d}.png".format(self.budget_val,epoch))
        plt.close()
        
        save_dic = {}
        save_dic['loss_val'] = self.stored_filtered_val_losses[0:self.nplot_val, :]
        save_dic['loss_tr'] = self.stored_filtered_train_losses[0:self.nplot_val, :]
        save_dic['loss_tr_u'] = self.stored_utility_train_losses[0:self.nplot_val, :]
        save_dic['loss_tr_s'] = self.stored_secret_train_losses[0:self.nplot_val, :]
        save_dic['loss_val_u'] = self.stored_utility_val_losses[0:self.nplot_val, :]
        save_dic['loss_val_s'] = self.stored_secret_val_losses[0:self.nplot_val, :]
        save_dic['infos'] = self.stored_infos[0:self.nplot_val, :]
        
        savepath = 'checkpoint/summary_losses_budget_{:.2e}.pkl'.format(self.budget_val)
        with open(savepath, 'wb') as f:
            pickle.dump(save_dic, f)
        

    def test_validation(self,eval_interval):
        
        (class_loss, class_acc,
         combined_loss, information)=self.evaluate_generator(self.dg_val,n_val=10)
        
        #Store information on validation
        self.stored_infos[self.nplot_val,0] = information[0] # IUX
        self.stored_infos[self.nplot_val,1] = information[1] # IUY
        self.stored_infos[self.nplot_val,2] = information[0] - information[1] # IUX|Y
        self.stored_infos[self.nplot_val,3] = information[2] # ISY

        # Store losses on validation
        self.stored_filtered_val_losses[self.nplot_val, :3] = combined_loss
        self.stored_secret_val_losses[self.nplot_val, :2] = [class_loss[2],class_acc[2]]
        self.stored_utility_val_losses[self.nplot_val, :2] = [class_loss[0],class_acc[0]]
        
        # Store losses on training set
        self.stored_filtered_train_losses[self.nplot_val, :] = np.mean(
            self.stored_filtered_losses[self.nplot_val*eval_interval:(self.nplot_val+1)*eval_interval,:],axis = 0)
        self.stored_secret_train_losses[self.nplot_val, :] = np.mean(
            self.stored_secret_losses[self.nplot_val*eval_interval:(self.nplot_val+1)*eval_interval,:],axis = 0)
        self.stored_utility_train_losses[self.nplot_val, :] = np.mean(
            self.stored_utility_losses[self.nplot_val*eval_interval:(self.nplot_val+1)*eval_interval,:],axis = 0)

        self.nplot_val += 1
        
        # ---------------------
        #  Write summary
        # ---------------------
        scalar_sum = self.sess.run(
            self.scalar_summary['merged_training_losses'], 
            feed_dict={
                self.scalar_summary['utility_acc_ph']:class_acc[0], self.scalar_summary['utility_loss_ph']:class_loss[0],
                self.scalar_summary['utility_raw_acc_ph']:class_acc[1], self.scalar_summary['utility_raw_loss_ph']:class_loss[1],
                self.scalar_summary['secret_acc_ph']:class_acc[2], self.scalar_summary['secret_loss_ph']:class_loss[2],
                self.scalar_summary['filter_CIB_loss_ph']:combined_loss[0],
                self.scalar_summary['filter_censored_info_loss_ph']:combined_loss[1],
                self.scalar_summary['filter_information_penalty_loss_ph']: combined_loss[2]}
        )
        self.val_writer.add_summary(scalar_sum, self.nplot_val*eval_interval)
        
        
        

        total_filter_loss=combined_loss[0]
        if total_filter_loss < self.best_eval:
            self.best_eval = total_filter_loss
            
        named_path = os.path.join(self.checkpoint_dir, "{:s}_{:.5f}".format(self.model_name, self.budget_val))
        classifier_type_path = 'batch_norm' if self.use_batchnorm_classifiers else 'instance_norm'
        named_class_path = os.path.join(named_path, classifier_type_path)
        filter_type_path = 'batch_norm' if self.use_batchnorm_filter else 'instance_norm'
        named_filter_path = os.path.join(named_path, filter_type_path)
        
        
        self.checkpoint_dir_f = os.path.join(named_filter_path, 'filter')
        self.checkpoint_dir_u = os.path.join(named_class_path, 'utility_filtered')
        self.checkpoint_dir_s = os.path.join(named_class_path, 'secret')
           
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(named_path):
            os.makedirs(named_path)
        if not os.path.exists(named_class_path):
            os.makedirs(named_class_path)
        if not os.path.exists(named_filter_path):
            os.makedirs(named_filter_path)
        if not os.path.exists(self.checkpoint_dir_f):
            os.makedirs(self.checkpoint_dir_f)
        if not os.path.exists(self.checkpoint_dir_u):
            os.makedirs(self.checkpoint_dir_u)
        if not os.path.exists(self.checkpoint_dir_s):
            os.makedirs(self.checkpoint_dir_s)
            
        self.utility_saver.save(self.sess,os.path.join(self.checkpoint_dir_u,'utility_filtered'),
                    global_step=1)
        self.filter_saver.save(self.sess,os.path.join(self.checkpoint_dir_f,'filter'),
                    global_step=1)
        self.secret_saver.save(self.sess,os.path.join(self.checkpoint_dir_s,'secret'),
            global_step=1)


    def plot_losses(self, epoch):
        clear_output(wait=True)
        plt.figure(figsize=(15, 12))
        plt.subplot(621)
        plt.semilogy(self.stored_secret_train_losses[0:self.nplot_val, 2], label='filtered_secret_loss')
        plt.semilogy(self.stored_secret_val_losses[0:self.nplot_val, 0], label='filtered_secret_loss')
        plt.legend(loc='best')

        plt.subplot(622)
        plt.plot(self.stored_secret_train_losses[0:self.nplot_val, 3], label='filtered_secret_acc')
        plt.plot(self.stored_secret_val_losses[0:self.nplot_val, 1], label='filtered_secret_acc val')
        plt.legend(loc='best')

        plt.subplot(623)
        plt.semilogy(self.stored_utility_train_losses[0:self.nplot_val, 2], label='filtered_utility_loss')
        plt.semilogy(self.stored_utility_val_losses[0:self.nplot_val, 0], label='filtered_utility_loss val')
        plt.legend(loc='best')

        plt.subplot(624)
        plt.plot(self.stored_utility_train_losses[0:self.nplot_val, 3], label='filtered_utility_acc')
        plt.plot(self.stored_utility_val_losses[0:self.nplot_val, 1], label='filtered_utility_acc val')
        plt.legend(loc='best')

        plt.subplot(613)
        plt.semilogy(self.stored_filtered_train_losses[0:self.nplot_val, 0], label='full filter loss')
        plt.semilogy(self.stored_filtered_val_losses[0:self.nplot_val, 0], label='full filter val loss')
        plt.legend(loc='best')
        
        plt.subplot(614)
        plt.semilogy((self.stored_filtered_train_losses[0:self.nplot_val, 1]), label='utility filter loss')
        plt.semilogy((self.stored_filtered_val_losses[0:self.nplot_val, 1]), label='utility filter val loss')
        plt.legend(loc='best')
        
        plt.subplot(615)

        plt.semilogy((self.stored_filtered_train_losses[0:self.nplot_val, 2]), label='secret filter loss')
        plt.semilogy((self.stored_filtered_val_losses[0:self.nplot_val, 2]), label='secret filter val loss')
        plt.legend(loc='best')
        
        plt.subplot(616)
        
        plt.semilogy((self.stored_filtered_train_losses[0:self.nplot_val, 2]) +(self.stored_filtered_train_losses[0:self.nplot_val, 1]), label='bound filter loss')
        plt.semilogy((self.stored_filtered_val_losses[0:self.nplot_val, 2]) + (self.stored_filtered_val_losses[0:self.nplot_val, 1]), label='bound filter val loss')
        plt.legend(loc='best')

        plt.show()


        
    def make_trainable(self, net,state):
        #net.trainable=state
        #for layer in net.layers:
        #    layer.trainable = state
            
        net.trainable = state
        for l in net.layers:
            l.trainable = state
            if hasattr(l, 'layers'):
                self.make_trainable(l, state)
                
    def load(self, train_checkpoint_dir,pretrain_checkpoint_dir,load_type='pretrain',budget_val=None):
        print(" [*] Reading checkpoint...")
        named_path_train = os.path.join(train_checkpoint_dir, "{:s}_{:.5f}".format(self.model_name, budget_val))
        named_path_pretrain = os.path.join(pretrain_checkpoint_dir, "{:s}".format(self.model_name))
        
        
        checkpoint_dir_ur = os.path.join(named_path_pretrain,
                                             'batch_norm' if self.use_batchnorm_classifiers else 'instance_norm', 'utility_raw')
        if load_type=='train':
            checkpoint_dir_uf = os.path.join(named_path_train,
                                             'batch_norm' if self.use_batchnorm_classifiers else 'instance_norm', 'utility_filtered')
            checkpoint_dir_s = os.path.join(named_path_train,
                                            'batch_norm' if self.use_batchnorm_classifiers else 'instance_norm', 'secret')
            checkpoint_dir_f = os.path.join(named_path_train,
                                            'batch_norm' if self.use_batchnorm_filter else 'instance_norm', 'filter')
        elif load_type=='pretrain':
            checkpoint_dir_uf = os.path.join(named_path_pretrain,
                                             'batch_norm' if self.use_batchnorm_classifiers else 'instance_norm', 'utility_filtered')
            checkpoint_dir_s = os.path.join(named_path_pretrain,
                                            'batch_norm' if self.use_batchnorm_classifiers else 'instance_norm', 'secret')
            checkpoint_dir_f = os.path.join(named_path_pretrain,
                                            'batch_norm' if self.use_batchnorm_filter else 'instance_norm','filter')
        
        success=True
        #Restore Utility raw
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir_ur)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.utility_raw_saver.restore(self.sess, os.path.join(checkpoint_dir_ur, ckpt_name))
            print("Utility Raw restore success")
        else:
            print("Utility Raw restore failed")
            success= False
        #Restore Utility filtered
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir_uf)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.utility_saver.restore(self.sess, os.path.join(checkpoint_dir_uf, ckpt_name))
            print("Utility Filtered restore success")
        else:
            print("Utility Filtered restore failed")
            success= False
        #Restore secret
        print(checkpoint_dir_s)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir_s)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.secret_saver.restore(self.sess, os.path.join(checkpoint_dir_s, ckpt_name))
            print("Secret restore success")
        else:
            print("Secret restore failed")
            success= False
        
        #Restore filter
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir_f)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.filter_saver.restore(self.sess, os.path.join(checkpoint_dir_f, ckpt_name))
            print("Filter restore success")
        else:
            print("Filter restore failed")
            success= False
        
        return success


    def evaluate_generator(self,dg,n_val=None):
        dg.shuffle()
        if n_val is None:
            n_val = len(dg)
        else:
            n_val = int(np.minimum(len(dg),n_val))   
        
        #classifier losses [u,ur,s]
        class_loss = np.zeros([n_val,3])
        #classifier accuracies [u,ur,s]
        class_acc = np.zeros([n_val,3])
        #combined losses [F, sc, uc]
        combined_loss = np.zeros([n_val,3])
        #information [IUX, IUY, ISY]
        information = np.zeros([n_val,3])
#         print(n_val)
#         print(dg)
        
        for idx in np.arange(n_val):
            # get data batch
            data_batch = dg.__getitem__(idx)
            imgs = data_batch[0][0]
            utility_gt = data_batch[1][0]
            secret_gt = data_batch[1][1] 
            
            #Get filtered images            
            feed_dict = {self.input_image_ph: imgs}
            gen_imgs = self.sess.run(self.filter_output, feed_dict=feed_dict)
            
            #get classifier losses and output probabilities
            feed_dict = {self.raw_input_image_ph: gen_imgs, self.secret_gt_ph : secret_gt, self.utility_gt_ph : utility_gt}
            u_loss, s_loss, u_acc, s_acc, pUY, pSY = self.sess.run(
                        [self.utility_loss, self.secret_loss, self.utility_acc,
                         self.secret_acc, self.utility_output, self.secret_output],
                        feed_dict=feed_dict)
            
            feed_dict = {self.raw_input_image_ph: imgs,  self.utility_gt_ph : utility_gt}
            ur_loss, ur_acc, pUX = self.sess.run(
                        [self.utility_raw_loss, self.utility_raw_acc, self.utility_raw_output],
                        feed_dict=feed_dict)
            
            # Compute mutual information
            IUX = computeMI(pUX,self.utility_prior[:pUX.shape[0],:])
            IUY = computeMI(pUY,self.utility_prior[:pUY.shape[0],:])
            ISY = computeMI(pSY,self.secret_prior[:pSY.shape[0],:])
            
            # Compute filter losses
            feed_dict= {self.input_image_ph: imgs, self.utility_gt_ph: pUX,
                        self.secret_prior_ph: self.secret_prior[:pUX.shape[0],...], self.budget_ph : self.budget_val,
                        self.lambda_ph:self.lambda_val}
            f_loss, u_c_loss, s_c_loss = self.sess.run(
                                [self.combined_loss, self.utility_combined_loss,
                                 self.secret_combined_loss], feed_dict=feed_dict)
            
            #storing and bookkeeping
            #classifier losses [u,ur,s]
            class_loss[idx,:] = [u_loss, ur_loss, s_loss]
            #classifier accuracies [u,ur,s]
            class_acc[idx,:] = [u_acc, ur_acc, s_acc]
            #combined losses [F, sc, uc]
            combined_loss[idx,:] = [f_loss, u_c_loss, s_c_loss]
            #information [IUX, IUY, ISY]
            information[idx,:] = [IUX, IUY, ISY]
            

        return class_loss.mean(0), class_acc.mean(0), combined_loss.mean(0), information.mean(0)