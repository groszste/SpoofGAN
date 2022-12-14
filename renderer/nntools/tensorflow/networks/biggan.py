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
from nntools import tensorflow as tftools
from .. import utils as tfutils 
from .. import losses as tflosses
from .. import watcher as tfwatcher
from tensorflow.contrib.opt import MovingAverageOptimizer
from scipy.special import expit

import os
import cv2
import random

class BigGAN:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)
            
    def initialize(self, config):
        '''
            Initialize the graph from scratch according config.
        '''
        self.config = config
        with self.graph.as_default():
            with self.sess.as_default():

                # Set up placeholders
                h, w = config.image_size
                batch_size = config.batch_size
                #batch_size = None
                channels = config.channels
                summaries = []

                self.images = tf.placeholder(tf.float32, shape=[None, h,w,channels],
                        name='images')
                self.binary_images = tf.placeholder(tf.float32, shape=[None, 256 , 256, channels], name='binary_images')
                self.random_z = tf.placeholder(tf.float32, shape=[None, config.z_dim], name='random_z')
                self.deepprint_gt_tfeats = tf.placeholder(tf.float32, shape=[None, 96], name='deepprint_gt_tfeats')
                self.deepprint_gt_mfeats = tf.placeholder(tf.float32, shape=[None, 96], name='deepprint_gt_mfeats')
                self.learning_rate_g = tf.placeholder(tf.float32, name='learning_rate_g')
                self.learning_rate_d = tf.placeholder(tf.float32, name='learning_rate_d')
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                self.phase_train = tf.placeholder(tf.bool, name='phase_train')
                self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                # create splits for different gpus
                image_splits = tf.split(self.images, config.num_gpus)
                binary_splits = tf.split(self.binary_images, config.num_gpus)
                dp_t_splits = tf.split(self.deepprint_gt_tfeats, config.num_gpus)
                dp_m_splits = tf.split(self.deepprint_gt_mfeats, config.num_gpus)
                z_splits = tf.split(self.random_z, config.num_gpus)
                g_grads_splits = []
                d_grads_splits = []
                g_split_dict = {}
                d_split_dict = {}
                def insert_dict_g(k,v):
                    if k in g_split_dict: g_split_dict[k].append(v)
                    else: g_split_dict[k] = [v]
                def insert_dict_d(k,v):
                    if k in d_split_dict: d_split_dict[k].append(v)
                    else: d_split_dict[k] = [v]

                g_loss_list = []
                d_loss_list = []
                for i in range(config.num_gpus):
                    scope_name = '' if i==0 else 'gpu_%d' % i
                    with tf.name_scope(scope_name):
                        with tf.variable_scope('', reuse=i>0):
                            with tf.device('/gpu:%d' % i):
                                # tower inputs
                                images = tf.identity(image_splits[i], name='inputs')
                                binary_images  = tf.identity(binary_splits[i], name='binary_inputs')
                                z_split = tf.identity(z_splits[i], name='z_inputs')
                                if i == 0:
                                    self.inputs = binary_images
                                    self.image_inputs = images
                                    self.z_inputs = z_split
                                dp_t = tf.identity(dp_t_splits[i], name='dp_t_inputs')
                                dp_m = tf.identity(dp_m_splits[i], name='dp_m_inputs')

                                self.setup_network_model(config)

                                # load the tensorflow model graph def for minutiae aec
                                self.load_ridge_extractor()

                                # load the tensorflow model graph def for deepprint
                                self.load_deepprint()

                                # # load spoofbuster model
                                # self.load_spoofbuster()

                                # get the generated image
                                G = self.generator(binary_images, z_split)
                                
                                # # Get SpoofBuster embeddings
                                # # pad G and input images to 750x800 and resize to 299x299
                                # images_resized = tf.image.resize(tf.image.pad_to_bounding_box(images, offset_height=238, offset_width=288, target_height=750, target_width=800), (299,299))
                                # G_resized = tf.image.resize(tf.image.pad_to_bounding_box(G, offset_height=238, offset_width=288, target_height=750, target_width=800), (299,299))
 
                                # # get prelogits
                                # prelogits, _ = self.spoofbuster(images_resized)
                                # prelogits_G, _ = self.spoofbuster(G_resized)

                                # prelogits = tf.squeeze(prelogits, [1, 2])
                                # prelogits_G = tf.squeeze(prelogits_G, [1, 2])
                                
                                # # loss to bring prelogits together
                                # spoofbuster_loss = config.sb_loss_weight * tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(prelogits - prelogits_G), axis=1))
                                # g_loss_list.append(spoofbuster_loss)
                                # insert_dict_g('g_sb_loss', spoofbuster_loss)

 
                                # extract generated binary ridges
                                binary_ridge_generated_512x512 = self.ridge_aec(image_batch=G, keep_prob=1.0, phase_train=False)
                                self.binary_ridge_generated = tf.image.resize_images(binary_ridge_generated_512x512, (256, 256))

                                # save a tensor for testing
                                if i == 0:
                                    self.rendered = tf.identity(G, name='rendered')
                                    summaries.append(tf.summary.image('original_MSP_images', images))
                                    summaries.append(tf.summary.image('rendered msp images', self.rendered))
                                    summaries.append(tf.summary.image('groundtruth_binary_images', binary_images))
                                    summaries.append(tf.summary.image('generated_binary_images', self.binary_ridge_generated))

                                # extract generated deepprint features
                                self.DeepPrint_input = tf.image.central_crop(tf.image.resize_images(G, (480, 480)), .9333) # 448 x 448
                                deepprint_generated_tfeats, _, end_points = self.deepprint(self.DeepPrint_input)
                                self.deepprint_generated_mfeats = tf.math.l2_normalize(end_points['minutiae_net'], axis=1)
                                self.deepprint_generated_tfeats = tf.math.l2_normalize(deepprint_generated_tfeats, axis=1)

				                # extract distorted deepprint features
                                """
                                distorted_input = tf.image.central_crop(tf.image.resize_images(distorted_G, (480, 480)), .9333) # 448 x 448
                                distorted_tfeats, _, end_points = self.deepprint(distorted_input)
                                distorted_mfeats = tf.math.l2_normalize(end_points['minutiae_net'], axis=1)
                                distorted_tfeats = tf.math.l2_normalize(distorted_tfeats, axis=1)
                                """

                                # get the adversarial loss
                                self.D_real = self.discriminator(images)
                                #self.D_fake = self.discriminator(distorted_G)
                                self.D_fake = self.discriminator(G)

                                d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels = tf.ones_like(self.D_real)))
                                d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels = tf.zeros_like(self.D_fake)))
                                g_adv_loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
                                self.d_loss = d_loss_real + d_loss_fake
                                d_loss_list.append(self.d_loss)
                                g_loss_list.append(g_adv_loss)

                                # get the regularization loss
                                reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='generator'), name='reg_loss')
                                g_loss_list.append(reg_loss)

                                # get the image level (content) loss
                                image_loss = config.img_weight * tf.nn.l2_loss(self.binary_ridge_generated - binary_images) * (1.0 / (256.0 * 256.0))
                                g_loss_list.append(image_loss)

                                # next compute the deep print loss
                                deepprint_tloss = config.dp_t_w * tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(dp_t - self.deepprint_generated_tfeats), axis=1))
                                deepprint_mloss = config.dp_m_w * tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(dp_m - self.deepprint_generated_mfeats), axis=1))
                                g_loss_list.append(deepprint_tloss)
                                g_loss_list.append(deepprint_mloss)

                                insert_dict_g('g_adv_loss', g_adv_loss)
                                insert_dict_g('g_reg_loss', reg_loss)
                                insert_dict_g('g_dp_t_loss', deepprint_tloss)
                                insert_dict_g('g_dp_m_loss', deepprint_mloss)
                                insert_dict_g('g_img_loss', image_loss)
                                insert_dict_d('d_loss', self.d_loss)

                                trainable_variables = [t for t in tf.trainable_variables()]
                                g_total_loss = tf.add_n(g_loss_list, name='g_total_loss')
                                d_total_loss = tf.add_n(d_loss_list, name='d_total_loss')
                                g_grads_split = tf.gradients(g_total_loss, trainable_variables)
                                d_grads_split = tf.gradients(d_total_loss, trainable_variables)
                                g_grads_splits.append(g_grads_split)
                                d_grads_splits.append(d_grads_split)

                g_grads = tfutils.average_grads(g_grads_splits)
                d_grads = tfutils.average_grads(d_grads_splits)
                for k,v in g_split_dict.items():
                    v = tfutils.average_tensors(v)
                    tfwatcher.insert(k, v)
                    if 'loss' in k:
                        summaries.append(tf.summary.scalar('losses/' + k, v))
                        #tf.summary.scalar('losses/' + k, v)
                    else:
                        summaries.append(tf.summary.scalar(k, v))
                        #tf.summary.scalar(k, v)
                for k,v in d_split_dict.items():
                    v = tfutils.average_tensors(v)
                    tfwatcher.insert(k, v)
                    if 'loss' in k:
                        summaries.append(tf.summary.scalar('losses/' + k, v))
                        #tf.summary.scalar('losses/' + k, v)
                    else:
                        #tf.summary.scalar(k, v)
                        summaries.append(tf.summary.scalar(k, v))

                # Training Operaters
                g_grads_filtered = []
                d_grads_filtered = []
                G_vars = []
                D_vars = []
                for i, t in enumerate(trainable_variables):
                    if 'discriminator' in t.name:
                        D_vars.append(t)
                        d_grads_filtered.append(d_grads[i])
                    elif 'generator' in t.name or 'LocalizationNet' in t.name:
                        G_vars.append(t)
                        g_grads_filtered.append(g_grads[i])

                self.train_G_op = tfutils.apply_gradient(G_vars, g_grads_filtered, config.optimizer_g,
                                        self.learning_rate_g, config.learning_rate_multipliers)
                self.train_D_op = tfutils.apply_gradient(D_vars, d_grads_filtered, config.optimizer_d,
                                        self.learning_rate_d, config.learning_rate_multipliers)
                
                update_global_step_op = tf.assign_add(self.global_step, 1)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                update_ops = update_ops + [update_global_step_op]
                self.update_ops = tf.group(*update_ops)
                
                self.summary_op = tf.summary.merge(summaries)
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())

                save_vars = tf.trainable_variables()
                for t in tf.global_variables():
                    if t not in save_vars:
                        print(f"not already included: {t.name}")
                        save_vars.append(t)

                self.saver = tf.train.Saver(save_vars, max_to_keep=5)

                # load ridge extraction AEC here
                ridge_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')
                ridge_saver = tf.train.Saver(ridge_vars)
                model_path = 'code/biggan/vloss_generation_binary2fing/log/binary_image/final/ckpt-6000'
                ridge_saver.restore(self.sess, model_path)
                print("finished loading Ridge AEC vars")

                # load the deepprint variables which have been saved beore
                deepprint_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionV4')
                deepprint_saver = tf.train.Saver(deepprint_vars)
                deepprint_saver.restore(self.sess, 'code/biggan/vloss_generation_warp/DeepPrint/ckpt-0')
                print("finished loading deepprint checkpoint")

                # # load the spoofbuster variables which have been saved beore
                # spoofbuster_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionV3')
                # spoofbuster_saver = tf.train.Saver(spoofbuster_vars)
                # spoofbuster_saver.restore(self.sess, 'code/biggan/vloss_generation_binary2fing/log/spoofbuster/whole_image/ckpt-120000')
                # print("finished loading spoofbuster checkpoint")
                
                self.watch_list = tfwatcher.get_watchlist()
    
    def setup_network_model(self, config):
        network_models = imp.load_source('network_model', config.network)
        self.generator = partial(network_models.generator,
                is_training=self.phase_train,
                reuse=tf.AUTO_REUSE)
         
        self.discriminator = partial(network_models.discriminator,
                is_training=self.phase_train,
                reuse=tf.AUTO_REUSE)

    def load_ridge_extractor(self):
        print("Loading Ride Extractor!!!")
        ridge_aec = imp.load_source('network_model', 'code/biggan/vloss_generation_binary2fing/nets/reconstruct_fingerprint_aec.py')
        self.ridge_aec = partial(ridge_aec.generator)
        print("finished loading module")

    def load_deepprint(self):
        print("Loading DeepPrint!!!")
        deepprint_model = imp.load_source('network_model', 'code/biggan/vloss_generation_binary2fing/nets/inception_deep_mtl.py')
        self.deepprint = partial(deepprint_model.inference,
                keep_probability = 1.0,
                phase_train=False,
                bottleneck_layer_size=96,
                reuse=tf.AUTO_REUSE)

    def load_spoofbuster(self):
        print("Loading SpoofBuster!!!")
        spoofbuster_model = imp.load_source('network_model', 'code/biggan/vloss_generation_binary2fing/nets/inception_v3.py')
        self.spoofbuster = partial(spoofbuster_model.inference,
                keep_probability = 1.0,
                phase_train=False,
                num_classes=None)

    def get_deepprint_feats(self, image_batch):
        # helper function to get deepprint features from an image batch
        image_batch = np.squeeze(image_batch)
        image_batch = image_batch * 127.5 + 128
        image_batch = resize(image_batch, (480, 480))
        image_batch = center_crop(image_batch, (448, 448))
        image_batch = standardize_images(image_batch, 'mean_scale')
        image_batch = image_batch[:,:,:,None]
        feed_dict = {
            self.DeepPrint_input: image_batch[0:2, :, :, :]
        }
        deepprint_mfeats1 = self.sess.run(self.deepprint_generated_mfeats, feed_dict=feed_dict)
        deepprint_tfeats1 = self.sess.run(self.deepprint_generated_tfeats, feed_dict=feed_dict)
        feed_dict = {
            self.DeepPrint_input: image_batch[2:, :, :, :]
        }
        deepprint_mfeats2 = self.sess.run(self.deepprint_generated_mfeats, feed_dict=feed_dict)
        deepprint_tfeats2 = self.sess.run(self.deepprint_generated_tfeats, feed_dict=feed_dict)
        
        # concat
        deepprint_mfeats = np.concatenate([deepprint_mfeats1, deepprint_mfeats2], axis=0)
        deepprint_tfeats = np.concatenate([deepprint_tfeats1, deepprint_tfeats2], axis=0)

        return deepprint_mfeats, deepprint_tfeats

    def train(self, image_batch, minutiae_batch, learning_rate_g, learning_rate_d):
        h,w,c = image_batch.shape[1:]

        deepprint_mfeats, deepprint_tfeats = self.get_deepprint_feats(image_batch)

        texture_noise = np.random.normal(size=(image_batch.shape[0], self.config.z_dim))
        
        # now compute the losses
        feed_dict = {
                    self.images: image_batch,
                    self.random_z: texture_noise,
                    self.binary_images: minutiae_batch,
                    self.phase_train: True,
                    self.deepprint_gt_tfeats: deepprint_tfeats,
                    self.deepprint_gt_mfeats: deepprint_mfeats,
                    self.learning_rate_g: self.config.g_lr,
                    self.learning_rate_d: self.config.d_lr,
                    }


        for i in range(self.config.g_iters):
            _ = self.sess.run(self.train_G_op, feed_dict=feed_dict)
        
        for i in range(self.config.d_iters):
            _ = self.sess.run([self.train_D_op], feed_dict = feed_dict)
        
        # run update ops
        wl,sm, _ = self.sess.run([tfwatcher.get_watchlist(), self.summary_op, self.update_ops], feed_dict = feed_dict)
        # get the current step
        step = self.sess.run([self.global_step], feed_dict = {})
        step = step[0]

        return wl, sm, step

    def restore_model(self, *args, **kwargs):
        trainable_variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tfutils.restore_model(self.sess, trainable_variables, *args, **kwargs)
        # restore_vars = self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # tfutils.restore_model(self.sess, restore_vars, *args, **kwargs)

    def save_model(self, model_dir, global_step):
        tfutils.save_model(self.sess, self.saver, model_dir, global_step)

    def generate_images2(self, minutiae, images, batch_size=1):
        batch_size = 1 # hacky
        #print("old generate_images for fgan is commented out")
        num_images = minutiae.shape[0]
        h,w,c = tuple(self.images.shape[1:])
        result = np.ndarray((num_images, h,w,c), dtype=np.float32)
        for start_idx in range(0, num_images, 1):
            end_idx = min(num_images, start_idx + batch_size)
            minu = minutiae[start_idx:end_idx]
            im = images[start_idx:end_idx]
            minu = np.concatenate([minu, minu], axis=0)
            im = np.concatenate([im, im], axis=0)
            texture_noise = np.random.normal(size=(4, self.config.z_dim))
            feed_dict = {
                    self.inputs: minu,
                    self.random_z: texture_noise,
                    self.image_inputs:im,
                    self.phase_train: False
                }
            result[start_idx:end_idx] = self.sess.run(self.rendered, feed_dict=feed_dict)[0, :, :, :]
        return result

    def generate_images(self, binary_image):
        num_images = binary_image.shape[0]
        texture_noise = np.random.normal(size=(num_images, 128))
        feed_dict = {
                    self.binary_inputs: binary_image,
                    self.random_z: texture_noise,
                    self.phase_train: True,
                    #self.keep_prob: 1.0
        }
        result = self.sess.run(self.rendered, feed_dict=feed_dict)
        return result


    def load_model(self, *args, **kwargs):
        print('load_model')
        tfutils.load_model(self.sess, *args, **kwargs)
        self.phase_train = self.graph.get_tensor_by_name('phase_train:0')
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
        self.rendered = self.graph.get_tensor_by_name('rendered:0')
        self.binary_inputs = self.graph.get_tensor_by_name('binary_inputs:0')
        self.random_z = self.graph.get_tensor_by_name('z_inputs:0')

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
