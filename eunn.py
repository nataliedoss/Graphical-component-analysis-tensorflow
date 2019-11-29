from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import numpy as np




def generate_index_fft(s):
    """
    generate the index lists for eunn to prepare weight matrices 
    and perform efficient rotations
    This function works for fft case
    """      
    def ind_s(k):
        if k==0:
            return np.array([[1,0]])
        else:
            temp = np.array(range(2**k))
            list0 = [np.append(temp + 2**k, temp)]
            list1 = ind_s(k-1)
            for i in range(k):
                list0.append(np.append(list1[i],list1[i] + 2**k))
            return list0

    t = ind_s(int(math.log(s/2, 2)))

    ind_exe = []
    for i in range(int(math.log(s, 2))):
        ind_exe.append(tf.constant(t[i]))

    ind_param = []
    for i in range(int(math.log(s, 2))):
        ind = np.array([])
        for j in range(2**i):
            ind = np.append(ind, np.array(range(0, s, 2**i)) + j).astype(np.int32)

        ind_param.append(tf.constant(ind))
    
    return ind_exe, ind_param


def fft_param(num_units):
    
    phase_init = tf.random_uniform_initializer(-3.14, 3.14)
    capacity = int(math.log(num_units, 2))

    theta = tf.get_variable("theta", [capacity, num_units//2], 
        initializer=phase_init)
    cos_theta = tf.cos(theta)
    sin_theta = tf.sin(theta)
        
    cos_list = tf.concat([cos_theta, cos_theta], axis=1)
    sin_list = tf.concat([sin_theta, -sin_theta], axis=1)
        
    ind_exe, index_fft = generate_index_fft(num_units)

    v1 = tf.stack([tf.gather(cos_list[i,:], index_fft[i]) for i in range(capacity)])
    v2 = tf.stack([tf.gather(sin_list[i,:], index_fft[i]) for i in range(capacity)])

    D = None

    diag = D

    return v1, v2, ind_exe, diag




def eunn_feedforward(x):
    """
    feedforward layer using eunn 
    input: x of shape [None, 2^N]
    output: same shape as x

    Note: 
    1. use different scope if this layer is called multiple times
    2. only support fft style in the paper
    3. only support real number, so it is actually orthogonal matrix

    """
    d = int(x.shape[-1])
    capacity = int(math.log(d, 2))
    v1, v2, ind, _ = fft_param(d)
    h = x
    for i in range(capacity):
        diag = h * v1[i]
        off = h * v2[i]
        h = diag + tf.gather(off, ind[i], axis=1)
    return h



