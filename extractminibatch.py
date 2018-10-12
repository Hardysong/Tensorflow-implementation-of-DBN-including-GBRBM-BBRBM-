#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 18:48:01 2018

    extract minibatch
    input: 
        kk: random permuation of i.e. kk = np.random.shuffle
        minibatch_num: current minibatch
        batchsize: minibatch size
        data: data to be extracted
    output:
        data extracted from data

@author: yibo
"""

import numpy as np

def extractminibatch(kk,minibatch_num,batchsize,data):
    batch_start = minibatch_num *  batchsize
    batch_end = (minibatch_num+1) * batchsize
    n_samples = data.shape[0]
    if (batch_end + batchsize) <= n_samples:
        idx = kk[batch_start:batch_end]
        batch = data[idx]
    else:
        batch = data[kk[batch_start:]]
        
    return batch