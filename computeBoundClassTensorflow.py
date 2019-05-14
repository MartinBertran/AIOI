import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


class boundClass():
    
    def __init__(self,Pus_m, lambda_base=10):
        self.Pus_m = Pus_m
        self.lambda_base=lambda_base
        (self.nu, self.ns ,
         self.Pus_u, self.Pus_s,
         self.Pu, self.Ps,
         self.Pus,
        self.IUS, self.HU) = self.compute_marginals(self.Pus_m)
        self.nus = self.nu*self.ns
        
    def instantiate_tensorflow_variables(self,nu,ns,ny):
        def xavier_init(size):
            in_dim = size[0]
            xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
            return tf.random_normal(shape=size, stddev=xavier_stddev)
        
        self.Pus_ph =  tf.placeholder(tf.float32,  shape = (nu*ns,1), name = 'Pus')
        self.Pus_u_ph =  tf.placeholder(tf.float32,  shape = (nu*ns,nu), name = 'Pus_u')
        self.Pus_s_ph =  tf.placeholder(tf.float32,  shape = (nu*ns,ns), name = 'Pus_s')
        self.Pu_ph =  tf.placeholder(tf.float32,  shape = (1,nu), name = 'Pu')
        self.Ps_ph =  tf.placeholder(tf.float32,  shape = (1,ns), name = 'Ps')
        self.budget_ph =  tf.placeholder(tf.float32,  shape = (), name = 'budget')
        self.lambda_ph =  tf.placeholder(tf.float32,  shape = (), name = 'lambda')
        self.lr_ph = tf.placeholder(tf.float32, shape=(), name = 'lr')


        D_y = tf.Variable(xavier_init([ny, nu*ns]), name='A_base_y')
        self.T = tf.nn.softmax(    D_y,    axis=0,    name="Py_us",)

        Py = tf.matmul(self.T,self.Pus_ph, name="Py")
        Py_u = tf.matmul(self.T,self.Pus_u_ph, name="Py_u")
        Py_s = tf.matmul(self.T,self.Pus_s_ph, name="Py_s")

        #print(D_y.shape, self.T.shape)
        #print(Py.shape, Py_u.shape, Py_s.shape)

        iy_u = Py_u * tf.log(Py_u/Py)
        self.IY_U = tf.reduce_sum(self.Pu_ph*iy_u)

        #print(iy_u.shape, self.IY_U.shape)

        iy_s = Py_s * tf.log(Py_s/Py)
        self.IY_S = tf.reduce_sum(self.Ps_ph*iy_s)

        #print(iy_s.shape, self.IY_S.shape)

        self.L = -self.IY_U + self.lambda_ph*tf.square(tf.nn.relu(self.IY_S-self.budget_ph))
        
        loss_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_ph)
        self.loss_solver = loss_optimizer.minimize(self.L, var_list=[D_y])
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess= tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        self.opt_vars = loss_optimizer.variables()
        #reset_optimizer_op = tf.variables_initializer(opt_vars)
    
    def get_curve(self, budget_list=None, n_y_list=None):

        if n_y_list is None:
            n_y_list =[self.nu,np.ceil(2.5*self.nu)]
        
        if budget_list is None:
            budget_list = np.linspace(0,self.IUS,4)[:-1]

        results_multi_lst = []

        ISY_multi_arr = np.zeros([len(n_y_list), len(budget_list)])
        IUY_multi_arr = np.zeros([len(n_y_list), len(budget_list)])

        for idx_y in np.arange(len(n_y_list)):
            ny=int(n_y_list[idx_y])
            print("ny",ny)
            self.instantiate_tensorflow_variables(self.nu,self.ns,ny)

            for idx in np.arange(len(budget_list)):
                
                budget = budget_list[idx]
                print("budget",budget)
                if budget==0:
                    lambda_val =10* self.lambda_base
                else:
                    lambda_val = self.lambda_base    

                #reset optimizer and load feed dictionary
                tf.variables_initializer(self.opt_vars)

                feed_dict={self.Pus_ph : self.Pus.reshape([-1,1]),
                           self.Pus_u_ph: self.Pus_u,
                           self.Pus_s_ph: self.Pus_s,
                           self.Pu_ph: self.Pu.reshape([1,-1]),
                           self.Ps_ph: self.Ps.reshape([1,-1]),
                           self.lambda_ph:lambda_val,
                           self.budget_ph:budget,
                           self.lr_ph:1e-2}
                #iterate
                for it in np.arange(10000):
                    T_val, IY_U_val, IY_S_val,L_val, _ = self.sess.run([
                        self.T,self.IY_U, self.IY_S,self.L, self.loss_solver],feed_dict=feed_dict)

                results_multi_lst +=[T_val]
                ISY_multi_arr[idx_y,idx] =IY_S_val
                IUY_multi_arr[idx_y,idx] =IY_U_val
            

        self.ISY_multi_arr=ISY_multi_arr
        self.IUY_multi_arr=IUY_multi_arr
        self.results_multi_lst = results_multi_lst
        self.n_y_list=n_y_list
        
        return ISY_multi_arr, IUY_multi_arr, results_multi_lst, self.HU, self.IUS, n_y_list
    
    def plot_everything(self):
        plt.figure(figsize=(9,9))
        for _ in np.arange(self.ISY_multi_arr.shape[0]):
            plt.plot(
                    np.concatenate([self.HU- self.IUY_multi_arr[_,:],[0]]),
                    np.concatenate([self.ISY_multi_arr[0,:],[self.IUS]]), '+-' ,label=self.n_y_list[_])
        plt.plot([self.IUS,0],[0,self.IUS], label='I(U;S)-I(U;S|X)')
        plt.ylabel('I(S;Y)')
        plt.xlabel('I(U;X|Y)')
        plt.grid()
        plt.legend()
        plt.show()


    def compute_marginals(self,Pus_m):
        def get_mi(p):
            pu = np.sum(p,axis = 1)
            ps = np.sum(p,axis = 0)
            ps = np.dot(np.ones([p.shape[0],1]),ps[np.newaxis,:])
            hu = -np.sum(pu[pu>0]*np.log(pu[pu>0]))
            pops = p/ps
            hu_s = -np.sum(p[p>0]*np.log(pops[p>0]))
            return hu-hu_s, hu

        nu,ns = Pus_m.shape
        Pus=Pus_m.flatten()

        # Get Pus_u matrix de nusxnu que tenga Pus|u, idem con Pus_s
        Pus_u = np.zeros([Pus.size,nu])
        Pus_s = np.zeros([Pus.size,ns])

        Pu = Pus_m.sum(1)
        Ps = Pus_m.sum(0)

        for u in np.arange(nu):
            aux = np.zeros(Pus_m.shape)
            aux[u,:] = Pus_m[u,:]
            aux/= aux.sum()
            Pus_u[:,u]= aux.flatten()

        for s in np.arange(ns):
            aux = np.zeros(Pus_m.shape)
            aux[:,s] = Pus_m[:,s]
            aux/= aux.sum()
            Pus_s[:,s]= aux.flatten()

            IUS, HU = get_mi(Pus_m)
        return nu, ns ,Pus_u, Pus_s, Pu, Ps, Pus, IUS,HU
        

