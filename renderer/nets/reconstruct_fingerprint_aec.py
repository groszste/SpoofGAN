from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import layer_norm, instance_norm

batch_norm_params = {
    'decay': 0.995,
    'epsilon': 0.001,
    'updates_collections': None,
    'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
}

gf_dim = 64

def leaky_relu(x):
    return tf.maximum(0.2 * x, x)

def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2d'):
        s = x.shape
        if len(s) == 4:
            x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
            x = tf.tile(x, [1, 1, factor, 1, factor, 1])
            x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        else:
            print("you are screwed"); sys.exit(0)           
        return x


def padding(x, pad, pad_type='zero'):
    if pad_type == 'zero':
        return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    if pad_type == 'reflect':
        return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0,]], mode='REFLECT')
    else:
        raise ValueError('Unknown pad type: {}'.format(pad_type))

def padding2d(x, pad):
    return tf.pad(x, [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]])

def conv_recon(x, *args, pad=1, **kwargs):
    with slim.arg_scope([slim.conv2d], padding='VALID'):
        if type(pad) is list:
            x = padding2d(x, pad)
        else:
            x = padding(x, pad)
        return slim.conv2d(x, *args, **kwargs)

def deconv_recon(x, *args, pad=1, **kwargs):
    with slim.arg_scope([slim.conv2d_transpose], padding='SAME'):
        x = padding(x, pad)
        return slim.conv2d(x, *args, **kwargs)


def generator(image_batch, keep_prob=1.0, phase_train=True, weight_decay=0.0, reuse=None, scope='Encoder'):
    k = 64
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
             activation_fn=tf.nn.relu,
             normalizer_fn=instance_norm,
             normalizer_params=None,
             weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
             weights_regularizer=slim.l2_regularizer(weight_decay)):
         with tf.variable_scope(scope, [image_batch], reuse=reuse):                   
            with slim.arg_scope([slim.dropout, slim.batch_norm], is_training=phase_train):
                    print('{} input shape: '.format(scope), [dim.value for dim in image_batch.shape])
                    k = 64
                    net = conv_recon(image_batch, k, kernel_size=3, stride=2, scope='conv0')
                    print('conv0 shape: ', [dim.value for dim in net.shape])
                    net = conv_recon(net, 2 * k, kernel_size=3, stride=2, scope='conv1')
                    print('conv1 shape: ', [dim.value for dim in net.shape])
                    net = conv_recon(net, 4 * k, kernel_size=3, stride=2, scope='conv2')
                    print('conv2 shape: ', [dim.value for dim in net.shape])
                    net = conv_recon(net, 6 * k, kernel_size=3, stride=2, scope='conv3')
                    print('conv3 shape: ', [dim.value for dim in net.shape])
                    encoded = conv_recon(net, 8 * k, kernel_size=3, stride=2, scope='conv4')
                    print('conv4 shape: ', [dim.value for dim in encoded.shape])
                    print("ENCODED SHAPE = {}".format(encoded.shape))

    scope = 'Decoder'
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
             activation_fn=tf.nn.relu,
             normalizer_fn=layer_norm,
             normalizer_params=None,
             weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
             weights_regularizer=slim.l2_regularizer(weight_decay)):
         with tf.variable_scope(scope, [encoded], reuse=reuse):                 
            with slim.arg_scope([slim.dropout, slim.batch_norm], is_training=phase_train):
                with slim.arg_scope([slim.fully_connected], normalizer_fn=layer_norm, normalizer_params=None): 
                    net = upscale2d(encoded, 2)
                    net = conv_recon(net, 6 * k, kernel_size=3, scope='deconv0_1')
                    print('deconv0 shape:', [dim.value for dim in net.shape])

                    net = upscale2d(net, 2)
                    net = conv_recon(net, 4 * k, kernel_size=3, scope='deconv1_1')
                    print('deconv1 shape:', [dim.value for dim in net.shape])

                    net = upscale2d(net, 2)
                    net = conv_recon(net, 2 * k, kernel_size=3, scope='deconv2_1')
                    print('deconv2 shape:', [dim.value for dim in net.shape])

                    net = upscale2d(net, 2)
                    net = conv_recon(net, k, kernel_size=3, scope='deconv3_1')
                    print('deconv3 shape:', [dim.value for dim in net.shape])

                    net = upscale2d(net, 2)
                    net = conv_recon(net, 32, kernel_size=3, scope='deconv4_1')
                    print('deconv4 shape:', [dim.value for dim in net.shape])

                    net = conv_recon(net, 1, kernel_size=3, activation_fn=None, normalizer_fn=None, scope='output')
                    net = tf.nn.tanh(net, name='output')
                    print('output:', [dim.value for dim in net.shape])
                    
                    return net
