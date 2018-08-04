# Tensorflow-implementation-of-Restricted-Boltzmann-Machine-including-GBRBM-BBRBM-
The application of RBM through Matlab have several drawbacks. One serious problem is the computation time is very time-consuming. It is because that the matlab is difficult to run in cloud server.So I wrap the code of rbm based on matlab to python based on tensorflow. We can use jupyter notebook to run the code for training or inference quickly.

test application:
  import numpy as np
  import tensorflow as tf
  import os,sys
  from rbm_py3 import rbm
  import time
  from matplotlib import pyplot as plt
  from math import *
  from tensorflow.examples.tutorials.mnist import input_data
  # temp, for debug
  sess = tf.InteractiveSession()

  mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

  data = mnist.train.next_batch(50000)
  data_x = data[0]
  data_y = data[1]

  my_rbm = rbm(784,100,
             learning_rate=0.01,
             momentum=0.8,
             rbm_type='gbrbm',
             relu_hidden = True,
             init_method = 'uniform'
             )

  my_rbm.plot=True
  my_rbm.pretrain(data_x,batch_size=100,n_epoches=10)
  
  my_rbm.save_model_weights()

  restore_path = os.getcwd()
  re_rbm = rbm(784,100,learning_rate=0.01,momentum=0.8,rbm_type='gbrbm',relu_hidden = True)
  re_rbm.restore_weights(restore_path,'rbm')

  x_up = my_rbm.rbmup(data_x)



references:

  https://github.com/meownoid/tensorfow-rbm
  https://github.com/lyy1994/generative-models/blob/master/RBM/RBM.ipynb
  http://deeplearning.net/tutorial/rbm.html
