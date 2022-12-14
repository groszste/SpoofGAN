import cv2
import imp
import numpy as np
import os
import random
from scipy import misc
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

from functools import partial
from tensorflow.contrib.opt import MovingAverageOptimizer
from scipy.special import expit

from .. import losses as tflosses
from .. import utils as tfutils 
from .. import watcher as tfwatcher

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
                #batch_size = config.batch_size
                batch_size = None
                channels = config.channels
                summaries = []

                self.binary_images = tf.placeholder(tf.float32, shape=[batch_size, h , w, channels], name='binary_images')
                self.random_z = tf.placeholder(tf.float32, shape=[batch_size, config.z_dim], name='random_z')
                self.learning_rate_g = tf.placeholder(tf.float32, name='learning_rate_g')
                self.learning_rate_d = tf.placeholder(tf.float32, name='learning_rate_d')
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                self.phase_train = tf.placeholder(tf.bool, name='phase_train')
                self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                # create splits for different gpus
                binary_splits = tf.split(self.binary_images, config.num_gpus)
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
                                binary_images  = tf.identity(binary_splits[i], name='binary_inputs')
                                z_split = tf.identity(z_splits[i], name='z_inputs')
                                if i == 0:
                                    self.inputs = z_split

                                self.setup_network_model(config)

                                # get the generated image
                                G = self.generator(z_split)
                                
                                # save a tensor for testing
                                if i == 0:
                                    self.rendered = tf.identity(G, name='rendered')
                                    summaries.append(tf.summary.image('groundtruth_binary_images (input)', binary_images))
                                    summaries.append(tf.summary.image('rendered (output)', self.rendered))

                                # get the adversarial loss
                                self.D_real = self.discriminator(binary_images)
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
                                reg_loss_d = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='discriminator'), name='reg_loss_d')
                                d_loss_list.append(reg_loss_d)

                                insert_dict_g('g_adv_loss', g_adv_loss)
                                insert_dict_g('g_reg_loss', reg_loss)
                                insert_dict_d('d_loss', self.d_loss)
                                insert_dict_d('d_reg_loss', reg_loss_d)

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
                    else:
                        summaries.append(tf.summary.scalar(k, v))
                for k,v in d_split_dict.items():
                    v = tfutils.average_tensors(v)
                    tfwatcher.insert(k, v)
                    if 'loss' in k:
                        summaries.append(tf.summary.scalar('losses/' + k, v))
                    else:
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
                self.saver = tf.train.Saver(trainable_variables, max_to_keep=None)

                self.watch_list = tfwatcher.get_watchlist()
    
    def setup_network_model(self, config):
        network_models = imp.load_source('network_model', config.network)
        self.generator = partial(network_models.generator,
                is_training=self.phase_train,
                reuse=tf.AUTO_REUSE)
         
        self.discriminator = partial(network_models.discriminator,
                is_training=self.phase_train,
                reuse=tf.AUTO_REUSE)


    def train(self, image_batch, minutiae_batch, learning_rate_g, learning_rate_d):
        h,w,c = image_batch.shape[1:]

        z_noise = np.random.normal(size=(image_batch.shape[0], self.config.z_dim))
        
        # now compute the losses
        feed_dict = {
                    self.binary_images: image_batch,
                    self.random_z: z_noise,
                    self.phase_train: True,
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

    def save_model(self, model_dir, global_step):
        tfutils.save_model(self.sess, self.saver, model_dir, global_step)

    def generate_images(self, random_z, batch_size=1):
        num_images = random_z.shape[0]
        h,w,c = 256, 256, 1
        result = np.ndarray((num_images, h,w,c), dtype=np.float32)
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(num_images, start_idx + batch_size)
            z = random_z[start_idx:end_idx]
            feed_dict = {
                    self.inputs: z,
                    self.phase_train: True
                }
            result[start_idx:end_idx] = self.sess.run(self.rendered, feed_dict=feed_dict)[0, :, :, :]
        return result

    def load_model(self, *args, **kwargs):
        print('load_model')
        tfutils.load_model(self.sess, *args, **kwargs)
        self.phase_train = self.graph.get_tensor_by_name('phase_train:0')
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
        self.rendered = self.graph.get_tensor_by_name('rendered:0')
        self.inputs = self.graph.get_tensor_by_name('z_inputs:0')

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
