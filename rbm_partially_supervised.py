#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:45:21 2018

@author: yibo
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import math
import timeit
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
        low = -1 * np.sqrt(6.0/np.sum(shape))
        high = 1 * np.sqrt(6.0/np.sum(shape))
        return tf.Variable(tf.random_uniform(shape = shape,
                                             minval = low,
                                             maxval = high,
                                             dtype = tf.float32))

def bias_init(shape,name='biases'):
    return tf.Variable(tf.constant(0.0,shape=shape),name=name)

def leaky_ReLU(feature_in,leaky=0.01,name="Leaky_ReLU"):
    return tf.maximum(leaky*feature_in,feature_in)

def active_function(activation = "relu"):
    if activation.lower() == "relu":
        return tf.nn.relu
    elif activation.lower() == "sigmoid":
        return tf.nn.sigmoid
    elif activation.lower() == "tanh":
        return tf.nn.tanh
    elif activation.lower() == "softplus":
        return tf.nn.softplus
    elif activation.lower() == "linear":
        return lambda x:x
    elif activation.lower() == "lrelu":
        return leaky_ReLU
    else:
        raise ValueError("Error type of activation function, should be \
                         relu(*)/sigmoid/tanh/softplus/linear")
def output_function(output_fun = "linear"):
    if output_fun.lower() == "linear":
        return lambda x:x
    elif output_fun.lower() == "softmax":
        return tf.nn.softmax
    elif output_fun.lower() == "sigmoid":
        return tf.nn.sigmoid
    else:
        raise ValueError("Error type of Output function, should be \
                         linear(regression problem) or softmax(classification problem)")

def cost_function(cost_fun = "mse"):
    if cost_fun.lower() == "mse":
        # mean squared error (linear regression)
        return tf.losses.mean_squared_error
    elif cost_fun.lower() == "expll":
        # exponential log likelihood (poisson regression)
        return tf.nn.log_poisson_loss
    elif cost_fun.lower() == "xent":
        # cross entropy (binary classification)
        return tf.nn.sigmoid_cross_entropy_with_logits
    elif cost_fun.lower() == "mcxent":
        # multi-class cross entropy (classification)->softmax
        return tf.nn.softmax_cross_entropy_with_logits
    elif cost_fun.lower() == 'class':
        return lambda y,y_: tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(y_,1)),'float')
    else:
        raise ValueError('Error type of Cost Function, should be \
                         mse(*)/expll/xent/mcxent/class(just for calculate class accuracy)')

def optimizer_fun(optimizer_method = 'sgd'):
    if optimizer_method.lower() == 'sgd':
        return tf.train.MomentumOptimizer
    elif optimizer_method.lower() == 'adam':
        return tf.train.AdamOptimizer
    elif optimizer_method.lower() == 'adagrad':
        return tf.train.AdagradDAOptimizer
    elif optimizer_method.lower() == 'rmsprop':
        return tf.train.RMSPropOptimizer
    else:
        raise ValueError('Error type of Optimizer Function, should be \
                         sgd(*)/adam/adagrad/rmsprop')

class rbm_partially_sup:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 n_out,
                 learning_rate = 0.01,
                 momentum = 0.0,
                 rbm_type = 'bbrbm',
                 init_method = 'gauss',
                 CD_k=1,
                 wPenalty = 0.0001,
                 cost_Fun = 'mse',
                 dropout = 0.,
                 relu_hidden = False,
                 relu_visible = False,
                 BP_optimizer_method = 'sgd',
                 BP_activation = "relu",
                 BP_output_fun = "linear",
                 BP_cost_fun = "mse",
                 rbm_name = 'rbm',
                 filepath = "saved_model_rbm"
                 ):
        # check parameters
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0,1]')
        assert 0. <= dropout <= 1.
            
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
        self.init_method = init_method
        
        # RBM parameters 
        self.w = weight_init([n_visible,n_hidden],
                        mean=0.,
                        stddev=math.sqrt(2.0/(n_visible+n_hidden)),
                        init_method=init_method)
        self.hidden_bias = bias_init([n_hidden])
        self.visible_bias = bias_init([n_visible])
        
        self.v_w = tf.Variable(tf.zeros([n_visible,n_hidden]),dtype=tf.float32)
        self.v_hb = tf.Variable(tf.zeros(n_hidden),dtype=tf.float32)
        self.v_vb = tf.Variable(tf.zeros(n_visible),dtype=tf.float32)
        
        # output layer part: weights and bias
        self.outlayer_w = weight_init([n_hidden,n_out],
                        mean=0.,
                        stddev=math.sqrt(2.0/(self.n_hidden+n_out)),
                        init_method=self.init_method)
        self.outlayer_bias = bias_init([n_out])
        
        self.BP_optimizer_method = optimizer_fun(BP_optimizer_method)
        self.BP_activation_fun = active_function(BP_activation)
        self.BP_output_fun = output_function(BP_output_fun)
        self.BP_cost_fun = cost_function(BP_cost_fun)
        
        # fliping index for computing pseudo likelihood
        self.i = 0
        
        # other parameters for training
        self.dropout = dropout
        self.wPenalty = wPenalty
        self.plot = False
        self.cost_fun = cost_Fun
        
        # path to save and restore model parameters
        if os.path.exists(filepath) is False:
            os.mkdir(filepath)
        self._save_path = filepath + "/" + "./" + rbm_name + ".ckpt"
        
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
        if self.dropout > 0:
            hid_samp = tf.nn.dropout(hid_samp,self.dropout) # dropout
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
    
    def CDk(self,visibles,targets):
        # gibbs sample - CD method
        v_sample = visibles
        h0_state,h_sample = self.sample_h_given_v(v_sample)
        
        BP_train = self.partially_supervised_part(visibles,targets)
        
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
    
    def partially_supervised_part(self,visibles,targets):        
        # The BP suoervised training part
        hidden_layer_output = tf.add(tf.matmul(visibles,self.w),self.hidden_bias)
        if self.dropout > 0:
            hidden_layer_output = tf.nn.dropout(hidden_layer_output,self.dropout)
        hidden_layer_output = self.BP_activation_fun(hidden_layer_output)
        
        output_layer = self.BP_output_fun(tf.add(tf.matmul(hidden_layer_output,self.outlayer_w),self.outlayer_bias))
        
        if self.BP_cost_fun == tf.losses.mean_squared_error:
            loss = self.BP_cost_fun(labels = targets,predictions = output_layer)
        elif self.BP_cost_fun == tf.nn.log_poisson_loss:
            loss = tf.reduce_mean(self.BP_cost_fun(targets = targets,log_input = output_layer))
        else:
            loss = tf.reduce_mean(self.BP_cost_fun(labels = targets, \
                                                   logits = tf.add(tf.matmul(hidden_layer_output,self.outlayer_w),
                                                                   self.outlayer_bias)))
        
        # L2 regularization
        l2_reg = self.wPenalty * tf.reduce_mean((tf.nn.l2_loss(self.w) + \
                                                    tf.nn.l2_loss(self.outlayer_w)))
        loss = loss + l2_reg
            
        # The optimizer
        if self.BP_optimizer_method == tf.train.MomentumOptimizer:
            optimizer = self.BP_optimizer_method(learning_rate = self.lr,momentum=self.momentum).minimize(loss)
        else:
            optimizer = self.BP_optimizer_method(learning_rate = self.lr).minimize(loss)
        print("Partially Supervised Training Layer!")
        return [optimizer,loss]
    
    def rbm_train(self,visibles,targets):
        w_grad,hb_grad,vb_grad = self.CDk(visibles,targets)
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
    
    def pretrain(self,data_x,data_y,batch_size=1,n_epoches=1):
        # return errors in training phase
        assert n_epoches > 0
        assert batch_size > 0
        # define the TF variables
        n_data = data_x.shape[0]
        x_in = tf.placeholder(tf.float32,shape=[None,self.n_visible])

        n_out = data_y.shape[1]
        y_out = tf.placeholder(tf.float32,shape=[None,n_out])
            
        rbm_pretrain = self.rbm_train(x_in,y_out)
        x_re,x_loss = self.reconstruct(x_in)
        
        n_batches = n_data // batch_size
        
        if n_batches == 0:
            n_batches = 1
        
        # deep copy
        data_x_cpy = data_x.copy()
        data_y_cpy = data_y.copy()
        inds = np.arange(n_data)
        
        # whether or not plot
        if self.plot is True:
            plt.ion() # start the interactive mode of plot
            plt.figure(1)
        
        # Define the saver
        all_saver = tf.train.Saver({'RBM_w':self.w,
                                'RBM_vb':self.visible_bias,
                                'RBM_hb':self.hidden_bias})
        
        # for save the trained parameters
        model_weights = []
        model_bias = []
        
        
        errs = []
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            mean_cost = []
            for epoch in range(n_epoches):
                start_time = timeit.default_timer()
                # shuffle
                np.random.shuffle(inds)
                mean_cost = []
                for b in range(n_batches):
                    batch_x = extractminibatch(inds,b,batch_size,data_x_cpy)
                    batch_y = extractminibatch(inds,b,batch_size,data_y_cpy)
                    sess.run(rbm_pretrain,feed_dict = {x_in:batch_x,y_out:batch_y})
                    cost = sess.run(x_loss,feed_dict={x_in:batch_x})
                    mean_cost.append(cost)
                
                errs.append(np.mean(mean_cost))
#                print('Epoch %d Cost %g' % (epoch, np.mean(mean_cost)))
                # timeit
                end_time = timeit.default_timer()
                training_time = (end_time-start_time)
                time_str = "%.4f" % (training_time)
                str1 = ('Epoch %d. Cost %g' % (epoch, errs[-1]))
                str_out = str1 + ". Consuming time: " + time_str +"s. "
                print(str_out)
                # plot ? 
                if plt.fignum_exists(1):
                    plt.plot(range(epoch+1),errs,'-r')
            # get the trained parameters
            model_weights.append(self.w.eval())
            model_bias.append(self.visible_bias.eval())
            model_bias.append(self.hidden_bias.eval())

            # save the model
            all_saver.save(sess,self._save_path)
        
        # save the trained results
        self.train_error = errs
        # input->weights,visible bias, hidden bias
        self.load_model_weights(model_weights[-1],model_bias[0],model_bias[1])
        self.model_weights = model_weights
        self.model_bias = model_bias
        
        return errs
    
    # load the trained model parameters (include weight and bias) after the train session
    def load_model_weights(self,w,vb,hb):
        # re init the weight and bias according to the trained parameters
        self.w.__init__(w)
        self.visible_bias.__init__(vb)
        self.hidden_bias.__init__(hb)
        
        print("(RBM) The weights and bias of this model have been successed loaded")
    
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
        if self.dropout > 0:
            x_up = tf.nn.dropout(x_up,1.0) # inference phase using dropout
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(x_up)
        
    # return model's weights
    def get_weights(self):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            return sess.run(self.w),sess.run(self.hidden_bias),sess.run(self.visible_bias)
    
    #restore weights from file
    def restore_weights(self,restore_file_path = None):
        if restore_file_path is None:
            restore_file_path = self._save_path
        
        # for save the parameters
        model_weights = []
        model_bias = []
        
        # Define the saver
        saver = tf.train.Saver({"RBM_w":self.w,
                                "RBM_vb":self.visible_bias,
                                "RBM_hb":self.hidden_bias})
    
        with tf.Session() as sess:
            
            saver.restore(sess,restore_file_path)
            model_weights.append(self.w.eval())
            model_bias.append(self.visible_bias.eval())
            model_bias.append(self.hidden_bias.eval())
        
        # registe the weight and bias to current model
        self.load_model_weights(model_weights[-1],model_bias[0],model_bias[1])
        
        self.model_weights = model_weights
        self.model_bias = model_bias
        print("(RBM) Restroe the parameter done!!")
        