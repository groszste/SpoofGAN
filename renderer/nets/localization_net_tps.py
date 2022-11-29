from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from nets.tps_STN import ElasticTransformer


def parametric_relu(x, shape):
    with tf.variable_scope('p_re_lu'):
        alphas = tf.get_variable('alpha', shape,
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * tf.minimum(x, 0)
    return pos + neg

def prelu_keras(x):
    if len(x.shape) == 4: 
        return tf.keras.layers.PReLU(shared_axes=[1,2]).apply(x)
    else:
        return tf.keras.layers.PReLU().apply(x)

activation = parametric_relu

def inference(images, random_z, out_size, is_training=False, 
                weight_decay=0.0, reuse=None):

    n0 = 4*4

    with tf.variable_scope('LocalizationNet', reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                with slim.arg_scope([slim.conv2d], padding='SAME'):
          
                    # 256 x 256
                    print('LocalizationNetSM input shape:', [dim.value for dim in images.shape])
                    with tf.variable_scope('Conv1'):
                        net = slim.conv2d(images, 32, 3, stride=2)
                    print('module 1 shape:', [dim.value for dim in net.shape]) # 128

                    with tf.variable_scope('Conv2'):
                        net = slim.conv2d(net, 64, 3, stride=2)
                    print('module 2 shape:', [dim.value for dim in net.shape]) # 64

                    with tf.variable_scope('Conv3'):
                        net = slim.conv2d(net, 128, 3, stride=2)
                    print('module 3 shape:', [dim.value for dim in net.shape]) # 32

                    with tf.variable_scope('Conv4'):
                        net = slim.conv2d(net, 256, 3, stride=2)
                    print('module 4 shape:', [dim.value for dim in net.shape]) # 16

                    with tf.variable_scope('Conv5'):
                        net = slim.conv2d(net, 512, 3, stride=2)
                    print('module 5 shape:', [dim.value for dim in net.shape]) # 8

                    with tf.variable_scope('Conv6'):
                        net = slim.conv2d(net, 1024, 3, stride=2)
                    print('module 6 shape:', [dim.value for dim in net.shape]) # 8
                    
                    with tf.variable_scope('Pooling'):
                        kernel_sz = [8, 8]
                        net = tf.squeeze(slim.max_pool2d(net, kernel_sz, stride=1))
                    print('post-pooling shape:', [dim.value for dim in net.shape])

                    with tf.variable_scope('FC1'):
                        net = slim.fully_connected(net, 512)
                        net = tf.concat([net, random_z], axis=1)
                    print('module fc1 shape:', [dim.value for dim in net.shape])

                    with tf.variable_scope('FC2'):
                        net = slim.fully_connected(net, 512)

                    with tf.variable_scope('FC3'):
                        net = slim.fully_connected(net, 2*n0, activation_fn=None) # 200
                    print('module final shape:', [dim.value for dim in net.shape])

        stl = ElasticTransformer(out_size, param_dim=2*n0)

        displacements = tf.tanh(net) * 0.05
        #displacements = net

        images_transformed = stl.transform(images, displacements)

    return images_transformed, displacements
