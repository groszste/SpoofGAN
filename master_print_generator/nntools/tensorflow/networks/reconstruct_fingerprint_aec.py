"""Main training file for face recognition
"""
# MIT License
# 
# Copyright (c) 2017 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from scipy import misc
import sys
import time
import imp
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from functools import partial

from .. import utils as tfutils 
from .. import losses as tflosses
from .. import watcher as tfwatcher
from scipy.special import expit
import os

class ReconFingerprintAEC:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)
            
    def initialize(self, config, num_classes):
        '''
            Initialize the graph from scratch according config.
        '''
        with self.graph.as_default():
            with self.sess.as_default():
                G_grad_splits = []
                D_grad_splits = []
                average_dict = {}
                concat_dict = {}
                
                def insert_dict(_dict, k,v):
                    if k in _dict: _dict[k].append(v)
                    else: _dict[k] = [v]

                # Set up placeholders
                h, w = config.image_size
                channels = config.channels
                self.disc_counter = config.disc_counter

                self.mode = config.mode

                summaries = []

                self.input = tf.placeholder(tf.float32, shape=[None, 512, 512, 1], name='input')
                self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                self.phase_train = tf.placeholder(tf.bool, name='phase_train')
                self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                self.setup_network_model(config, num_classes)

                self.reconstructed_fingerprint = self.generator(self.input)

                summaries.append(tf.summary.image('recon', self.reconstructed_fingerprint))
                summaries.append(tf.summary.image('input', self.input))

                # COMPUTE L2 LOSS
                diff = self.input - self.reconstructed_fingerprint
                l2_loss = tf.nn.l2_loss(diff)
                l1_loss = tf.reduce_mean(tf.abs(diff))
                
                self.g_loss = l2_loss * (1.0 / (512.0 * 512.0))

                insert_dict(average_dict, 'g_loss', self.g_loss)

                E_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
                G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')
                all_vars = E_vars + G_vars

                # origin lr = 0.0001
                self.train_G_op = tf.train.AdamOptimizer(0.001, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=all_vars)

                for k,v in average_dict.items():
                    v = tfutils.average_tensors(v)
                    average_dict[k] = v
                    tfwatcher.insert(k,v)
                    if 'loss' in k:
                        summaries.append(tf.summary.scalar('losses/' + k, v))
                    elif 'acc' in k:
                        summaries.append(tf.summary.scalar('acc/' + k, v))
                    else:
                        tf.summary(k, v)
                for k,v in concat_dict.items():
                    v = tf.concat(v, axis=0, name='merged_' + k)
                    concat_dict[k] = v
                    tfwatcher.insert(k, v)

                save_variables = tf.trainable_variables()
                self.update_global_step_op = tf.assign_add(self.global_step, 1)
                summaries.append(tf.summary.scalar('learning_rate', self.learning_rate))
                self.summary_op = tf.summary.merge(summaries)

                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(save_variables, max_to_keep=5)

                self.watch_list = tfwatcher.get_watchlist()
    
    def setup_network_model(self, config, num_classes):
        network_models = imp.load_source('network_model', config.network)
        self.generator = partial(network_models.generator,
                keep_prob = self.keep_prob,
                phase_train = self.phase_train,
                weight_decay = config.weight_decay,
                reuse=tf.AUTO_REUSE,
                scope='Encoder')

    def train(self, image_batch, learning_rate, num_classes, keep_prob):
        h,w,c = image_batch.shape[1:]

        # now compute the losses
        feed_dict = {
                    self.input: image_batch,
                    self.learning_rate: learning_rate,
                    self.keep_prob: keep_prob,
                    self.phase_train: True,
                    }
        #for i in range(25):
        for i in range(1):
            _ = self.sess.run(self.train_G_op, feed_dict=feed_dict)
        
        wl,sm, step = self.sess.run([tfwatcher.get_watchlist(), self.summary_op, self.update_global_step_op], feed_dict = feed_dict)
        return wl, sm, step

    def restore_model(self, *args, **kwargs):
        trainable_variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tfutils.restore_model(self.sess, trainable_variables, *args, **kwargs)

    def save_model(self, model_dir, global_step):
        tfutils.save_model(self.sess, self.saver, model_dir, global_step)

    def reconstruction(self, images, batch_size=128):
        feed_dict = {
                    self.input: images,
                    self.keep_prob: 1.0,
                    self.phase_train: False,
                }
        out_fingerprints = self.sess.run(self.G, feed_dict=feed_dict)
        return out_fingerprints


    def load_model(self, *args, **kwargs):
        print('load_model')
        tfutils.load_model(self.sess, *args, **kwargs)
        self.phase_train = self.graph.get_tensor_by_name('phase_train:0')
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
        self.G = self.graph.get_tensor_by_name('Decoder/output:0')
        self.input = self.graph.get_tensor_by_name('input:0')

def inception_arg_scope(weight_decay=0.00004,

                        use_batch_norm=True,

                        batch_norm_decay=0.9997,

                        batch_norm_epsilon=0.001):

  """Defines the default arg scope for inception models.

  Args:

    weight_decay: The weight decay to use for regularizing the model.

    use_batch_norm: "If `True`, batch_norm is applied after each convolution.

    batch_norm_decay: Decay for batch norm moving average.

    batch_norm_epsilon: Small float added to variance to avoid dividing by zero

      in batch norm.

  Returns:

    An `arg_scope` to use for the inception models.

  """

  batch_norm_params = {

      # Decay for the moving averages.

      'decay': batch_norm_decay,

      # epsilon to prevent 0s in variance.

      'epsilon': batch_norm_epsilon,

      # collection containing update_ops.

      'updates_collections': tf.GraphKeys.UPDATE_OPS,

  }

  if use_batch_norm:

    normalizer_fn = slim.batch_norm

    normalizer_params = batch_norm_params

  else:

    normalizer_fn = None

    normalizer_params = {}

  # Set weight_decay for weights in Conv and FC layers.

  with slim.arg_scope([slim.conv2d, slim.fully_connected],

                      weights_regularizer=slim.l2_regularizer(weight_decay)):

    with slim.arg_scope(

        [slim.conv2d],

        weights_initializer=slim.variance_scaling_initializer(),

        activation_fn=tf.nn.relu,

        normalizer_fn=normalizer_fn,

        normalizer_params=normalizer_params) as sc:

             return sc 

def standardize_images(images, standard):
    if standard=='mean_scale':
        mean = 127.5
        std = 128.0
    elif standard=='scale':
        mean = 0.0
        std = 255.0
    images_new = images.astype(np.float32)
    images_new = (images_new - mean) / std
    return images_new

def resize(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    shape_new = get_new_shape(images, size)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    for i in range(n):
        images_new[i] = misc.imresize(images[i], (h,w))

    return images_new

def center_crop(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    assert (_h>=h and _w>=w)

    y = int(round(0.5 * (_h - h)))
    x = int(round(0.5 * (_w - w)))

    images_new = images[:, y:y+h, x:x+w]

    return images_new

def get_new_shape(images, size):
    w, h = tuple(size)
    shape = list(images.shape)
    shape[1] = h
    shape[2] = w
    shape = tuple(shape)
    return shape

def pad_image(img_data, output_width, output_height):
    height, width = img_data.shape
    output_img = np.ones((output_height, output_width), dtype=np.int32) * 255
    margin_h = (output_height - height) // 2
    margin_w = (output_width - width) // 2
    output_img[margin_h:margin_h+height, margin_w:margin_w+width] = img_data
    return output_img
