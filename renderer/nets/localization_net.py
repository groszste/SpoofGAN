from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import atanh
from functools import partial
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import instance_norm, layer_norm

from nets.sparse_image_warp import sparse_image_warp


batch_norm_params = {
    'decay': 0.995,
    'epsilon': 0.001,
    'updates_collections': None,
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}
 

def leaky_relu(x):
    return tf.maximum(0.2*x, x)


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        return x


def padding(x, pad, pad_type='reflect'):
    if pad_type == 'zero' :
        return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    if pad_type == 'reflect' :
        return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
    else:
        raise ValueError('Unknown pad type: {}'.format(pad_type))

def conv(x, *args, pad=1, **kwargs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='VALID'):
        x = padding(x, pad)
        return slim.conv2d(x, *args, **kwargs)

def deconv(x, *args, pad=1, **kwargs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='VALID'):
        x = padding(x, pad)
        return slim.conv2d_transpose(x, *args, **kwargs)


def warp(images_rendered, random_z, image_size=(512,512), keep_prob=1.0, 
                    phase_train=True, weight_decay=0.0, reuse=None, scope='WarpController'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.dropout, slim.batch_norm], is_training=phase_train):
                with slim.arg_scope([slim.fully_connected],
                    normalizer_fn=layer_norm, normalizer_params=None):
                    print('{} input shape:'.format(scope), [dim.value for dim in images_rendered.shape])
                        
                    batch_size = tf.shape(random_z)[0]
                    h, w = tuple(image_size)
                    k = 64                      

                    warp_input = tf.identity(images_rendered, name='warp_input')

                    net = slim.fully_connected(random_z, 128, scope='fc1')
                    print('module fc1 shape:', [dim.value for dim in net.shape])

                    num_ldmark = 16

                    # Predict the control points
                    ldmark_mean = (np.random.normal(0,50, (num_ldmark,2)) + np.array([[0.5*h,0.5*w]])).flatten()
                    ldmark_mean = tf.Variable(ldmark_mean.astype(np.float32), name='ldmark_mean')
                    print('ldmark_mean shape:', [dim.value for dim in ldmark_mean.shape])

                    ldmark_pred = slim.fully_connected(net, num_ldmark*2, 
                        weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
                        normalizer_fn=None, activation_fn=None, biases_initializer=None, scope='fc_ldmark')
                    ldmark_pred = ldmark_pred + ldmark_mean
                    print('ldmark_pred shape:', [dim.value for dim in ldmark_pred.shape])
                    ldmark_pred = tf.identity(ldmark_pred, name='ldmark_pred')
             

                    # Predict the displacements
                    ldmark_diff = slim.fully_connected(net, num_ldmark*2, 
                        normalizer_fn=None,  activation_fn=None, scope='fc_diff')
                    print('ldmark_diff shape:', [dim.value for dim in ldmark_diff.shape])
                    ldmark_diff = tf.identity(ldmark_diff, name='ldmark_diff')
                    ldmark_diff = tf.identity(1.0 * ldmark_diff, name='ldmark_diff_scaled')



                    src_pts = tf.reshape(ldmark_pred, [-1, num_ldmark ,2])
                    dst_pts = tf.reshape(ldmark_pred + ldmark_diff, [-1, num_ldmark, 2])

                    diff_norm = tf.reduce_mean(tf.norm(src_pts-dst_pts, axis=[1,2]))

                    images_transformed, dense_flow = sparse_image_warp(warp_input, src_pts, dst_pts,
                            regularization_weight = 1e-6, num_boundary_points=0)
                    dense_flow = tf.identity(dense_flow, name='dense_flow')

            return images_transformed, images_rendered, ldmark_pred, ldmark_diff
