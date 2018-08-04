# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:15:18 2018

@author: yibo sun
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys,os
import math,random
from extractminibatch import extractminibatch

import matplotlib.pyplot as plt

def weight_init(shape,mean = 0.0,stddev = 0.1,seed = 1,name='weights',init_method='gauss'):
    if init_method.lower() == 'gauss':
        return tf.Variable(tf.truncated_normal(shape=shape,
                                               mean=mean,
                                               stddev=stddev,
                                               seed = seed),
                            name = name)
    elif init_method.lower() == 'uniform':
        return tf.get_variable(name=name,shape=shape,
                               initializer = tf.contrib.layers.xavier_initializer())

def bias_init(shape,name='biases'):
    return tf.Variable(tf.constant(0.0,shape=shape),name=name)

class rbm:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 learning_rate = 0.01,
                 momentum = 0.0,
                 rbm_type = 'bbrbm',
                 init_method = 'gauss',
                 CD_k=1,
                 wPenalty = 0.0001,
                 cost_Fun = 'mse',
                 relu_hidden = False,
                 relu_visible = False
                 ):
        # check parameters
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0,1]')
            
        # RBM training parameter
        self.rbm_type = rbm_type
        if rbm_type.lower() == 'gbrbm':
            self.sig = 1.0;
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.CD_k = CD_k
        self.relu_hidden = relu_hidden
        self.relu_visible = relu_visible
        self.lr = learning_rate
        self.momentum = momentum
        
        # RBM parameters 
        self.w = weight_init([n_visible,n_hidden],
                        mean=0.,
                        stddev=math.sqrt(2.0/(n_visible+n_hidden)),
                        name='w',
                        init_method=init_method)
        self.hidden_bias = bias_init([n_hidden],'hb')
        self.visible_bias = bias_init([n_visible],'vb')
        
        self.v_w = tf.Variable(tf.zeros([n_visible,n_hidden]),dtype=tf.float32)
        self.v_hb = tf.Variable(tf.zeros(n_hidden),dtype=tf.float32)
        self.v_vb = tf.Variable(tf.zeros(n_visible),dtype=tf.float32)
        
        # fliping index for computing pseudo likelihood
        self.i = 0
        
        # other parameters for training
        self.wPenalty = wPenalty
        self.plot = False
        self.cost_fun = cost_Fun
        
    def actV2H(self,vis):
        if self.rbm_type.lower() == 'bbrbm':
            h_active = tf.matmul(vis,self.w) + self.hidden_bias
        elif self.rbm_type.lower() == 'gbrbm':
            h_active = tf.matmul(vis/self.sig,self.w) + self.hidden_bias
        else:
            raise TypeError('Error Type of rbm, should be bbrbm or gbrbm!')
            
        if self.relu_hidden is True:
            h_prob = tf.nn.relu(h_active)
        else:
            h_prob = tf.nn.sigmoid(h_active)
        return h_active,h_prob
    def actH2V(self,hid):
        if self.rbm_type.lower() == 'bbrbm':
            v_active = tf.matmul(hid,tf.transpose(self.w)) + self.visible_bias
        elif self.rbm_type.lower() == 'gbrbm':
            v_active = tf.matmul(hid*self.sig,tf.transpose(self.w)) + self.visible_bias
        else:
            raise TypeError('Error Type of rbm, should be bbrbm or gbrbm!')
        
        if self.relu_visible is True:
            v_prob = tf.nn.relu(v_active)
        else:
            if self.rbm_type.lower() == 'bbrbm':
                v_prob = tf.nn.sigmoid(v_active)
            elif self.rbm_type.lower() == 'gbrbm':
                v_prob = v_active
            else:
                raise TypeError('Error Type of rbm, should be bbrbm or gbrbm!')
        return v_active,v_prob
    
    def sample_h_given_v(self,vis):
        hidact,hidprob = self.actV2H(vis)
        if self.relu_hidden is True:
            hid_samp = tf.nn.relu(hidact+tf.random_normal(tf.shape(hidact)))
        else:
            hid_samp = tf.nn.relu(tf.sign(hidprob - tf.random_uniform(tf.shape(hidprob))))
        return hidprob,hid_samp
    
    def sample_v_given_h(self,hid):
        v_act,v_prob = self.actH2V(hid)
        if self.relu_visible is True:
            v_samp = tf.nn.relu(v_act+tf.random_normal(tf.shape(v_act)))
        else:
            if self.rbm_type.lower() == 'bbrbm':
                v_samp = tf.nn.relu(tf.sign(v_prob - tf.random_uniform(tf.shape(v_prob))))
            elif self.rbm_type.lower() == 'gbrbm':
                v_samp = tf.random_normal(tf.shape(v_prob),v_prob)
            else:
                raise TypeError('Error Type of rbm, should be bbrbm or gbrbm!')
        return v_prob,v_samp
    
    def CDk(self,visibles):
        # gibbs sample - CD method
        v_sample = visibles
        h0_state,h_sample = self.sample_h_given_v(v_sample)
        
        w_positive = tf.matmul(tf.transpose(visibles),h0_state)
        
        # gibbs
        for i in range(self.CD_k):
            v_state,v_sample = self.sample_v_given_h(h_sample)
            h_state,h_sample = self.sample_h_given_v(v_state)
        
        w_negative = tf.matmul(tf.transpose(v_state),h_state)
        
        w_grad = tf.divide(tf.subtract(w_positive,w_negative),tf.to_float(tf.shape(visibles)[0]))
        hb_grad = tf.reduce_mean(h0_state-h_state,0)
        vb_grad = tf.reduce_mean(visibles-v_state,0)
        
        if self.rbm_type.lower() == 'gbrbm':
            w_grad = tf.divide(w_grad,self.sig)
            vb_grad = tf.divide(vb_grad,self.sig**2)
        
        return w_grad,hb_grad,vb_grad
    
    def rbm_train(self,visibles):
        w_grad,hb_grad,vb_grad = self.CDk(visibles)
        # compute new velocities
        v_w = self.momentum * self.v_w + self.lr * (w_grad - self.wPenalty * self.w)
        v_hb = self.momentum * self.v_hb + self.lr * hb_grad
        v_vb = self.momentum * self.v_vb + self.lr * vb_grad
        
        # update rbm parameters
        update_w = tf.assign(self.w,self.w + v_w)
        update_hb = tf.assign(self.hidden_bias,self.hidden_bias+v_hb)
        update_vb = tf.assign(self.visible_bias,self.visible_bias+v_vb)
        
        # update vlocities
        update_v_w = tf.assign(self.v_w,v_w)
        update_v_hb = tf.assign(self.v_hb,v_hb)
        update_v_vb = tf.assign(self.v_vb,v_vb)
        
        return [update_w,update_hb,update_vb,update_v_w,update_v_hb,update_v_vb]

    def reconstruct(self,visibles):
        _,h_samp = self.sample_h_given_v(visibles)
        # reconstruct phase
        for i in range(self.CD_k):
            v_recon,_ = self.sample_v_given_h(h_samp)
            h_state,h_samp = self.sample_h_given_v(v_recon)
        
        recon_error = self.error(visibles,v_recon)
        return v_recon,recon_error
    
    def error(self,vis,vis_recon):
        if self.cost_fun.lower() == 'mse':
            # mean squared error (linear regression)
            loss = tf.reduce_mean(tf.square(vis - vis_recon))
        elif self.cost_fun.lower() == 'expll':
            # exponential log likelihood (poisson regression)
            loss = tf.reduce_mean(vis_recon - vis*tf.log(vis_recon))
        elif self.cost_fun.lower() == 'xent':
            # cross entropy error (binary classification/Logistic regression)
            loss = -tf.reduce_mean(vis*tf.log(vis_recon)+(1-vis)*tf.log(1-vis_recon+1e-5))
        elif self.cost_fun.lower() == 'mcxent':
            # multi-class (>2) cross entropy (classification) used by softmax
            loss = -tf.reduce_mean(vis*tf.log(vis_recon))
        else:
            raise TypeError("Error type of cost function, should be mse(*)/expll/xent/mcxent")
        return loss
    
    def pretrain(self,data_x,batch_size=1,n_epoches=1,data_y=None):
        # return errors in training phase
        assert n_epoches > 0 and batch_size > 0
        # define the TF variables
        n_data = data_x.shape[0]
        x_in = tf.placeholder(tf.float32,shape=[None,self.n_visible])
        if data_y is not None:
            n_out = data_y.shape[1]
            y_out = tf.placeholder(tf.float32,shape=[None,n_out])
            
        rbm_pretrain = self.rbm_train(x_in)
        x_re,x_loss = self.reconstruct(x_in)
        
        n_batches = n_data // batch_size
        
        if n_batches == 0:
            n_batches = 1
        
        # deep copy
        data_x_cpy = data_x.copy()
        inds = np.arange(n_data)
        
        # whether or not plot
        if self.plot is True:
            plt.ion() # start the interactive mode of plot
            plt.figure(1)
            
        
        errs = []
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            mean_cost = []
            for epoch in range(n_epoches):
                # shuffle
                np.random.shuffle(inds)
                mean_cost = []
                for b in range(n_batches):
                    batch_x = extractminibatch(inds,b,batch_size,data_x_cpy)
                    sess.run(rbm_pretrain,feed_dict = {x_in:batch_x})
                    cost = sess.run(x_loss,feed_dict={x_in:batch_x})
                    mean_cost.append(cost)
                errs.append(np.mean(mean_cost))
#                print('Epoch %d Cost %g' % (epoch, np.mean(mean_cost)))
                print('Epoch %d Cost %g' % (epoch, errs[-1]))
                
                # plot ? 
                if plt.fignum_exists(1):
                    plt.plot(range(epoch+1),errs,'-r')
            self.train_error = errs
            return errs
    
    def free_energy(self,visibles):
        # ref:http://deeplearning.net/tutorial/rbm.html
        first_term = tf.matmul(visibles,tf.reshape(self.visible_bias,
                                                   [tf.shape(self.visible_bias)[0],1]))
        second_term = tf.reduce_sum(tf.log(1+tf.exp(self.hidden_bias+tf.matmul(visibles,self.w))),axis=1)
        
        return -first_term - second_term
    
    def pseudo_likelihood(self,visibles):
        # binarize the input image by rounding to nearest integer
        x = tf.round(visibles)
        # calculate free energy for the given bit configuration
        x_fe = self.free_energy(x)
        
        split0,split1,split2 = tf.split(x,[self.i,1,tf.shape(x)[1]-self.i-1],1)
        xi = tf.concat([split0,1-split1,split2],1)
        self.i = (self.i+1)%self.n_visible
        xi_fe = self.free_energy(xi)
        return tf.reduce_mean(self.n_visible*tf.log(tf.nn.sigmoid(xi_fe-x_fe)),axis=0)
    
    def rbmup(self,visibles):
        _,x_up = self.actV2H(visibles)
        if self.relu_hidden is True:
            [x_mean,x_std] = tf.nn.moments(x_up,axes=[0])
            x_up = tf.nn.batch_normalization(x_up,x_mean,x_std,0,1,1e-10)
            
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(x_up)
        
    # return model's weights
    def get_weights(self):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            return sess.run(self.w),sess.run(self.hidden_bias),sess.run(self.visible_bias)
            
    # save the current model
    def save_model_weights(self,save_path=None,name=None):
        if save_path is None:
            save_path = os.getcwd()
        if name is None:
            name = 'rbm'
        filename = os.path.join(save_path,name+'.meta')
        saver = tf.train.Saver({
                                name + '_w':self.w,
                                name + '_hb':self.hidden_bias,
                                name + '_vb':self.visible_bias})
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            return saver.save(sess,filename)
    
    # load the model parameters (include weight and bias)
    def load_model_weights(self,w,hidden_bias,visible_bias):
        self.w.assign(w)
        self.hidden_bias.assign(hidden_bias)
        self.visible_bias.assign(visible_bias)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
    
    #restore weights from file
    def restore_weights(self,restore_path,name):
        if name is None:
            name = 'rbm'
        filename = os.path.join(restore_path,name+'.meta')
        saver = tf.train.Saver({
                                name + '_w':self.w,
                                name + '_hb':self.hidden_bias,
                                name + '_vb':self.visible_bias})
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver.restore(sess,filename)