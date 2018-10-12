#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:25:20 2018

@author: yibo
"""
from __future__ import print_function

import os
import tensorflow as tf
from rbm_py3 import rbm
from rbm_partially_supervised import rbm_partially_sup

class dbn_partially_sup(object):
    
    """Deep Belief Network (DBN)
     The DBN is obtained by stacking several RBMs on top of each other. The hidden
     layer of the RBM at layer 'i' becomes the input of the RBM at layer 'i+1'. The
     first layer RBM gets input of the network, and the hidden layer of the last of
     the RBM represents the output. When used for classification, adding a logistic
     regression layer (softmax) on the top, for regression, adding a linear layer on th 
     top.
    """
    def __init__(self,n_in,hidden_layers_sizes,n_out,opts):
        assert n_in > 0
        
        self._sizes = [n_in] + hidden_layers_sizes
        self.n_out = n_out
        self.train_params = opts
        self.rbm_list = []
        
        # path to save and restore model parameters
        if os.path.exists('saved_model_DBN') is False:
            os.mkdir('saved_model_DBN')
        
        for i in range(len(self._sizes)-1):
            exec("rbm_%d = rbm(self._sizes[i],\
                               self._sizes[i+1],\
                               opts['learning_rate'],\
                               opts['momentum'],\
                               opts['rbm_type'],\
                               opts['weight_init_type'],\
                               opts['CDk'],\
                               opts['wPenalty'],\
                               opts['costFun'],\
                               opts['dropout'],\
                               opts['relu_hidden'],\
                               opts['relu_visible'],\
                               'rbm_%d',\
                               'saved_model_DBN')"%(i,i))
            
            exec("rbm_%d.plot=opts['plot']"%(i))
            exec("self.rbm_list.append(rbm_%d)"%(i))
        
        if opts['rbm_type'] == 'gbrbm':
            for rbm_tmp in self.rbm_list:
                rbm_tmp.sig = opts['gbrbm_param']['sig']
                
        if opts['partially_supervised_type'].lower() == 'regression':
            supervised_layer = opts['partially_supervised_layer']
            for i in supervised_layer:
                exec("rbm_%d = rbm_partially_sup(self._sizes[i],\
                                             self._sizes[i+1],\
                                             n_out,\
                                             opts['learning_rate'],\
                                             opts['momentum'],\
                                             opts['rbm_type'],\
                                             opts['weight_init_type'],\
                                             opts['CDk'],\
                                             opts['wPenalty'],\
                                             opts['costFun'],\
                                             opts['dropout'],\
                                             opts['relu_hidden'],\
                                             opts['relu_visible'],\
                                             opts['BP_optimizer_method'],\
                                             opts['BP_activation'],\
                                             opts['BP_output_fun'],\
                                             opts['BP_cost_fun'],\
                                             'rbm_%d',\
                                             'saved_model_DBN')"%(i,i))
                
                exec("self.rbm_list[i] = rbm_%d"%(i))
    
    def train(self,data_x,data_y):
        # Train the DBN layer by layer
        opts = self.train_params
        
        batch_size = opts['batchsize']
        epochs = opts['epochs']
        RBM_num = 0
        X = data_x.copy()
        Y = data_y.copy()
        for rbm_tmp in self.rbm_list:
            RBM_num += 1
            print("Training the %d RBM"%RBM_num)
            rbm_tmp.pretrain(data_x=X,data_y = Y,batch_size=batch_size,n_epoches=epochs)
            X = rbm_tmp.rbmup(X)
            if opts['x_val'] is not None:
                opts['x_val'] = rbm_tmp.rbmup(opts['x_val'])
        
    def restore_weights(self):
        # loading the trained parameters from file
        for rbm_tmp in self.rbm_list:
            rbm_tmp.restore_weights()
