#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 22:22:21 2018

@author: yibo
"""

def dbncreateopts(rbmclass='gbrbm',numepochs=1,batchsize=1):
    # Dict that contain the training parameters of DBM and RBM
    dbnopts = {}
    
    if rbmclass.lower() == 'bbrbm':
        dbnopts['rbm_type'] = 'bbrbm'
        
    elif rbmclass.lower() == 'gbrbm':
        dbnopts['rbm_type'] = 'gbrbm'
        gbrbm = {}
        gbrbm['sig'] = 1.
        dbnopts['gbrbm_param'] = gbrbm
               
    else:
        raise ValueError('The type of RBM must be gbrbm or bbrbm')
    
    dbnopts['epochs'] = numepochs
    dbnopts['batchsize'] = batchsize
    dbnopts['learning_rate'] = 0.01
    dbnopts['momentum'] = 0.9
    dbnopts['wPenalty'] = 0.0001
           
    dbnopts['relu_hidden'] = False
    dbnopts['relu_visible'] = False
    dbnopts['weight_init_type'] = 'gauss' # gauss(Kaiming He (2015)'s method) or uniform(xavier)
    dbnopts['dropout'] = 0
    dbnopts['costFun'] = 'mse' #include mse/expll/xent/mcxent/
    dbnopts['CDk'] = 1 # number of gibbs sample step
    dbnopts['plot'] = False
    
    dbnopts['y_train'] = None # target. If included, this information will be add
                               #during pre-train phase (reference: Bengio et al., 2007)
    dbnopts['partially_supervised'] = True # use target (y) to constrain the pretraining
                                           # phase, let the weight trained in pre train
                                           # phase have the information of target
                                           # this manner could improve the representative of
                                           # weight and imporve the accuracy
    dbnopts['partially_supervised_type'] = 'classification' # or 'regression'
    dbnopts['partially_supervised_layer'] = [1] # which layer or layers to perform partially supervised pre-train
    dbnopts['partially_supervised_outputFun'] = 'linear' # active function of partially supervise
                                                           #'linear','exp','sigmoid','softmax','tanh','softrect'
    dbnopts['partially_supervised_costFun'] = 'mse' # available cost function include: 
                                                    # mse,expll,xent,mcxent
    
    dbnopts['x_val'] = None #for validation
    dbnopts['y_val'] = None
    
    dbnopts['saveEvery'] = 100 # epoch, save training weight every of epoch
    dbnopts['saveDir'] = None # Path to save the model
    
    # print the dict 
    for key in dbnopts:
        print(key+': '+str(dbnopts[key]))
    
    return dbnopts