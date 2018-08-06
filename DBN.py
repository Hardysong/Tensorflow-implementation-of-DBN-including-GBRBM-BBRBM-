#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 22:21:35 2018

@author: yibo
"""
from __future__ import print_function

import tensorflow as tf
from rbm_py3 import rbm

class dbn(object):
    
    """Deep Belief Network (DBN)
     The DBN is obtained by stacking several RBMs on top of each other. The hidden
     layer of the RBM at layer 'i' becomes the input of the RBM at layer 'i+1'. The
     first layer RBM gets input of the network, and the hidden layer of the last of
     the RBM represents the output. When used for classification, adding a logistic
     regression layer (softmax) on the top, for regression, adding a linear layer on th 
     top.
    """
    def __init__(self,n_in,hidden_layers_sizes,opts):
        assert n_in > 0
        
        self._sizes = [n_in] + hidden_layers_sizes
        self.train_params = opts
        self.rbm_list = []
         
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
                               opts['relu_visible'])"%(i+1))
            exec("rbm_%d.plot=opts['plot']"%(i+1))
            exec("self.rbm_list.append(rbm_%d)"%(i+1))
        
        if opts['rbm_type'] == 'gbrbm':
            for rbm_i in self.rbm_list:
                rbm_i.sig = opts['gbrbm_param']['sig']
    
    def train(self,data_x):
        # Train the DBN layer by layer
        opts = self.train_params
        
        batch_size = opts['batchsize']
        epochs = opts['epochs']
         
        X = data_x.copy()
        for rbm_i in self.rbm_list:
             rbm_i.pretrain(data_x=X,batch_size=batch_size,n_epoches=epochs)
             X = rbm_i.rbmup(X)
             if opts['x_val'] is not None:
                 opts['x_val'] = rbm_i.rbmup(opts['x_val'])
        
        if opts['saveDir'] is not None:
            print("Save current Model....")
            for i in len(self.rbm_list):
                self.rbm_list[i].save_model_weights(save_path=opts['saveDir'],name="rbm_%d"%i)
