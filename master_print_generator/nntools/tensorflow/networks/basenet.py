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

import sys
import os
import time
import imp
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2
import math

from .. import utils as tfutils 
from .. import losses as tflosses
from .. import watcher as tfwatcher

class BaseNetwork:
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
                # Set up placeholders
                w, h = config.image_size
                channels = config.channels
                batch_size = None
                image_batch_placeholder = tf.placeholder(tf.float32, shape=[batch_size, h, w, channels], name='image_batch')
                label_batch_placeholder = tf.placeholder(tf.int32, shape=[batch_size], name='label_batch')
                minutiae_label_batch_placeholder = tf.placeholder(tf.float32, shape=[batch_size, 128, 128, 6], name='minutiae_label_batch')
                learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
                keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob')
                phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
                global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                image_splits = tf.split(image_batch_placeholder, config.num_gpus)
                label_splits = tf.split(label_batch_placeholder, config.num_gpus)
                minutiae_splits = tf.split(minutiae_label_batch_placeholder, config.num_gpus)
                grads_splits = []
                split_dict = {}
                def insert_dict(k,v):
                    if k in split_dict: split_dict[k].append(v)
                    else: split_dict[k] = [v]
                        
                for i in range(config.num_gpus):
                    scope_name = '' if i==0 else 'gpu_%d' % i
                    with tf.name_scope(scope_name):
                        with tf.variable_scope('', reuse=i>0):
                            with tf.device('/gpu:%d' % i):
                                images = tf.identity(image_splits[i], name='inputs')
                                labels = tf.identity(label_splits[i], name='labels')
                                minutiae_labels = tf.identity(minutiae_splits[i], name='minutiae_labels')
                                splits = tf.split(minutiae_labels, 2, axis=3)
                                # Save the first channel for testing
                                if i == 0:
                                    self.inputs = images
                                    tf.summary.image('input image', images)
                                    for s_index, split in enumerate(splits):
                                        tf.summary.image('input map{}'.format(s_index), split)
                                
                                # Build networks
                                if config.localization_net is not None:
                                    localization_net = imp.load_source('localization_net', config.localization_net)
                                    #images, splits, theta, cropped_images = localization_net.inference(images, splits, config.image_size, 
                                    #                phase_train_placeholder,
                                    #                weight_decay = 0.0)
                                    images, splits, theta, imgs2 = localization_net.inference(images, splits, config.image_size, 
                                                    phase_train_placeholder,
                                                    weight_decay = 0.0)
                                    self.thetas = theta
                                    images = tf.identity(images, name='transformed_image')
                                    self.images = images

                                    minutiae_labels = tf.concat(splits, axis=3) # restack the transformed minutiae labels
                                    print("TRANSFORMED MLABELS SHAPE: {}".format(minutiae_labels.shape))

                                    
                                    if i == 0:
                                        tf.summary.image('transformed_image', images)
                                        tf.summary.image('full_transformed', imgs2)
                                        tf.summary.image('cropped_map0', tf.squeeze(splits[0]))
                                        tf.summary.image('cropped_map1', tf.squeeze(splits[1]))
                                else:
                                    self.images = images

                                network = imp.load_source('network', config.network)
                                prelogits, aux_logits, end_points = network.inference(images, keep_prob_placeholder, phase_train_placeholder,
                                                        bottleneck_layer_size = config.embedding_size, 
                                                        weight_decay = config.weight_decay, 
                                                        model_version = config.model_version)
                                prelogits = tf.identity(prelogits, name='prelogits')

                                embeddings = tf.nn.l2_normalize(prelogits, dim=1, name='embeddings')
                                if i == 0:
                                    self.outputs = tf.identity(prelogits, name='outputs')
                                    self.moutputs = tf.identity(end_points['minutiae_net'], name='MinutiaeFeatures')

                                # Build all losses
                                loss_list = []

                                # MSE loss
                                if 'minutiae_map'in config.losses.keys():
                                    mmap = end_points['Conv2d_MinutiaeMap']
                                    if i == 0:
                                        print("mmap shape: {}".format(mmap.shape))
                                        split0, split1 = tf.split(mmap, 2, axis=3)
                                        tf.summary.image('estimated map0', split0)
                                        tf.summary.image('estimated map1', split1)
                                    minutiae_map = tf.identity(mmap, name='minutiae_map')
                                    minutiae_labels = tf.stop_gradient(minutiae_labels)
                                    #mse = 0.125 * tf.losses.mean_squared_error(minutiae_labels, minutiae_map)
                                    mse = 0.095 * tf.losses.mean_squared_error(minutiae_labels, minutiae_map)
                                    #mse = tf.losses.mean_squared_error(minutiae_labels, minutiae_map)
                                    loss_list.append(mse)
                                    insert_dict('minutiae_map_loss', mse)                                   
                                # minutiae feature softmax
                                if 'minutiae_softmax' in config.losses.keys():
                                    minutiae_prelogits = tf.identity(end_points['minutiae_net'], name="minutiae_prelogits")
                                    minutiae_logits = slim.fully_connected(minutiae_prelogits, num_classes, 
                                                                    weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                                                    weights_initializer=slim.xavier_initializer(),
                                                                    biases_initializer=tf.constant_initializer(0.0),
                                                                    activation_fn=None, scope='Minutiae_Logits')
                                    cross_entropy = config.losses['minutiae_softmax']['weight'] * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    labels=labels, logits=minutiae_logits), name='cross_entropy')
                                    loss_list.append(cross_entropy)
                                    insert_dict('minutiae_softmax_loss', cross_entropy)

                                    # add center loss with softmax
                                    closs = 0.05 * tflosses.center_loss(minutiae_prelogits, labels, num_classes, scope='CenterLossMinu')
                                    loss_list.append(closs)
                                    insert_dict('minutiae_closs', closs)
                                # Orignal Softmax
                                if 'softmax' in config.losses.keys():
                                    logits = slim.fully_connected(prelogits, num_classes, 
                                                                    weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                                                    weights_initializer=slim.xavier_initializer(),
                                                                    biases_initializer=tf.constant_initializer(0.0),
                                                                    activation_fn=None, scope='Logits')
                                    cross_entropy = config.losses['softmax']['weight'] * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    labels=labels, logits=logits), name='cross_entropy')
                                    loss_list.append(cross_entropy)
                                    insert_dict('sloss', cross_entropy)

                                    # add center loss with softmax
                                    closs = 0.05 * tflosses.center_loss(prelogits, labels, num_classes, scope='CenterLossTex')
                                    loss_list.append(closs)
                                    insert_dict('closs', closs)
                                

                                if 'aux-softmax' in config.losses.keys():
                                    aux_logits = tf.identity(aux_logits, name='aux_logits')
                                    aux_logits = slim.fully_connected(aux_logits, num_classes, 
                                                                    weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                                                    weights_initializer=slim.xavier_initializer(),
                                                                    biases_initializer=tf.constant_initializer(0.0),
                                                                    activation_fn=None, scope='Aux_logits')
                                    cross_entropy = config.losses['aux-softmax']['weight'] * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    labels=labels, logits=aux_logits), name='aux_cross_entropy')
                                    loss_list.append(cross_entropy)
                                    insert_dict('aux_loss', cross_entropy)
                                if 'diam-softmax' in config.losses.keys():
                                    diam_loss = tflosses.diam_softmax(prelogits, labels, 38291, 'texture')
                                    loss_list.append(diam_loss)
                                    insert_dict('t_diam_loss', diam_loss)
                                if 'diam-softmax-minutiae' in config.losses.keys():
                                    minutiae_prelogits = tf.identity(end_points['minutiae_net'], name="minutiae_prelogits")
                                    diam_loss = tflosses.diam_softmax(minutiae_prelogits, labels, 38291, 'minutiae')
                                    loss_list.append(diam_loss)
                                    insert_dict('m_diam_loss', diam_loss)                                 
                                # Triplet Loss
                                if 'triplet' in config.losses.keys():
                                    triplet_loss = tflosses.triplet_semihard_loss(labels, embeddings, **config.losses['triplet'])
                                    loss_list.append(triplet_loss)
                                    insert_dict('loss', triplet_loss)
                                # Contrastive Loss
                                if 'contrastive' in config.losses.keys():
                                    contrastive_loss = tflosses.contra = tflosses.contrastive_loss(labels, embeddings, **config.losses['contrastive'])
                                    loss_list.append(contrastive_loss)
                                    insert_dict('loss', contrastive_loss)

                                # L2-Softmax
                                if 'cosine' in config.losses.keys():
                                    logits, cosine_loss = losses.cosine_softmax(prelogits, labels, num_classes, 
                                                            gamma=config.losses['cosine']['gamma'], 
                                                            weight_decay=config.weight_decay)
                                    loss_list.append(cosine_loss)
                                    insert_dict('closs', cosine_loss)
                                # A-Softmax
                                if 'angular' in config.losses.keys():
                                    a_cfg = config.losses['angular']
                                    angular_loss = a_cfg['weight'] * tflosses.angular_softmax(prelogits, labels, num_classes, 
                                                            global_step, a_cfg['m'], a_cfg['lamb_min'], a_cfg['lamb_max'],
                                                            weight_decay=config.weight_decay)
                                    loss_list.append(angular_loss)
                                    insert_dict('aloss', angular_loss)
                                if 'minutiae_angular' in config.losses.keys():
                                    minutiae_prelogits = tf.identity(end_points['minutiae_net'], name="minutiae_prelogits")
                                    a_cfg = config.losses['minutiae_angular']
                                    minutiae_angular_loss = a_cfg['weight'] * tflosses.angular_softmax(minutiae_prelogits, labels, num_classes, 
                                                            global_step, a_cfg['m'], a_cfg['lamb_min'], a_cfg['lamb_max'],
                                                            weight_decay=config.weight_decay, scope='MinutiaeAngularSoftmax')
                                    loss_list.append(minutiae_angular_loss)
                                    insert_dict('minutiae_aloss', minutiae_angular_loss)
                                if 'aux_angular' in config.losses.keys():
                                    aux_prelogits = tf.identity(aux_logits, name='aux_logits')
                                    a_cfg = config.losses['aux_angular']
                                    aux_angular_loss = a_cfg['weight'] * tflosses.angular_softmax(aux_prelogits, labels, num_classes, 
                                                            global_step, a_cfg['m'], a_cfg['lamb_min'], a_cfg['lamb_max'],
                                                            weight_decay=config.weight_decay, scope='AuxAngularSoftmax')
                                    loss_list.append(aux_angular_loss)
                                    insert_dict('aux_aloss', aux_angular_loss)
                                # AM-Softmax
                                if 'am_softmax' in config.losses.keys():
                                    amloss = tflosses.am_softmax(prelogits, labels, num_classes, 
                                                            global_step, weight_decay=config.weight_decay,
                                                            **config.losses['am_softmax'])
                                    loss_list.append(amloss)
                                    insert_dict('loss', amloss)
                                # Split Loss
                                if 'split' in config.losses.keys():
                                    split_losses = tflosses.split_softmax(prelogits, labels, num_classes, 
                                                            global_step, **config.losses['split'])
                                    loss_list.append(split_losses)
                                    insert_dict('sploss', split_losses)
                                if 'pair' in config.losses.keys():
                                    pair_losses = tflosses.pair_loss(prelogits, labels, num_classes, 
                                                            global_step, gamma=config.losses['pair']['gamma'],  
                                                            m=config.losses['pair']['m'],
                                                            weight_decay=config.weight_decay)
                                    loss_list.extend(pair_losses)
                                    insert_dict('loss', pair_losses[0])
                                if 'triplet_avg' in config.losses.keys():
                                    tcfg = config.losses['triplet_avg']
                                    triplet_loss = tflosses.triplet_loss_avghard(labels, prelogits, tcfg['margin'], normalize=True, scope='TripletLoss')
                                    loss_list.append(triplet_loss)
                                    insert_dict('triplet_loss', triplet_loss)
                                if 'minutiae_triplet_avg' in config.losses.keys():
                                    tcfg = config.losses['minutiae_triplet_avg']
                                    minutiae_prelogits = tf.identity(end_points['minutiae_net'], name="minutiae_prelogits")
                                    triplet_loss = tflosses.triplet_loss_avghard(labels, minutiae_prelogits, tcfg['margin'], normalize=True, scope='MinutiaeTripletLoss')
                                    loss_list.append(triplet_loss)
                                    insert_dict('minutiae_triplet_loss', triplet_loss)
                                if 'aux_triplet_avg' in config.losses.keys():
                                    tcfg = config.losses['aux_triplet_avg']
                                    aux_prelogits = tf.identity(aux_logits, name='aux_logits')
                                    triplet_loss = tflosses.triplet_loss_avghard(labels, aux_prelogits, tcfg['margin'], normalize=True, scope='AuxTripletLoss')
                                    loss_list.append(triplet_loss)
                                    insert_dict('aux_triplet_loss', triplet_loss)


                                # Collect all losses
                                reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
                                loss_list.append(reg_loss)
                                insert_dict('reg_loss', reg_loss)

                                total_loss = tf.add_n(loss_list, name='total_loss')
                                grads_split = tf.gradients(total_loss, tf.trainable_variables())
                                grads_splits.append(grads_split)



                # Merge the splits
                grads = tfutils.average_grads(grads_splits)
                for k,v in split_dict.items():
                    v = tfutils.average_tensors(v)
                    tfwatcher.insert(k, v)
                    if 'loss' in k:
                        tf.summary.scalar('losses/' + k, v)
                    else:
                        tf.summary.scalar(k, v)


                # Training Operaters
                apply_gradient_op = tfutils.apply_gradient(tf.trainable_variables(), grads, config.optimizer,
                                        learning_rate_placeholder, config.learning_rate_multipliers)

                update_global_step_op = tf.assign_add(global_step, 1)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                train_ops = [apply_gradient_op, update_global_step_op] + update_ops
                train_op = tf.group(*train_ops)

                tf.summary.scalar('learning_rate', learning_rate_placeholder)
                summary_op = tf.summary.merge_all()

                # Initialize variables
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

                # Keep useful tensors
                self.image_batch_placeholder = image_batch_placeholder
                self.label_batch_placeholder = label_batch_placeholder 
                self.minutiae_label_batch_placeholder = minutiae_label_batch_placeholder
                self.learning_rate_placeholder = learning_rate_placeholder 
                self.keep_prob_placeholder = keep_prob_placeholder 
                self.phase_train_placeholder = phase_train_placeholder 
                self.global_step = global_step
                self.train_op = train_op
                self.summary_op = summary_op
                


    def train(self, image_batch, label_batch, minutiae_batch, learning_rate, keep_prob):
        feed_dict = {self.image_batch_placeholder: image_batch,
                    self.label_batch_placeholder: label_batch,
                    self.minutiae_label_batch_placeholder: minutiae_batch,
                    self.learning_rate_placeholder: learning_rate,
                    self.keep_prob_placeholder: keep_prob,
                    self.phase_train_placeholder: True,}
        _, wl, sm = self.sess.run([self.train_op, tfwatcher.get_watchlist(), self.summary_op], feed_dict = feed_dict)
        step = self.sess.run(self.global_step)

        return wl, sm, step
    
    def restore_model(self, *args, **kwargs):
        trainable_variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tfutils.restore_model(self.sess, trainable_variables, *args, **kwargs)

    def save_model(self, model_dir, global_step):
        tfutils.save_model(self.sess, self.saver, model_dir, global_step)
        

    def load_model(self, *args, **kwargs):
        tfutils.load_model(self.sess, *args, **kwargs)
        self.phase_train_placeholder = self.graph.get_tensor_by_name('phase_train:0')
        self.keep_prob_placeholder = self.graph.get_tensor_by_name('keep_prob:0')
        #self.inputs = self.graph.get_tensor_by_name('image_batch:0')
        self.inputs = self.graph.get_tensor_by_name('inputs:0')
        self.outputs = self.graph.get_tensor_by_name('outputs:0')
        self.moutputs = self.graph.get_tensor_by_name('InceptionV4/InceptionV4/MinutiaeFeatures/BiasAdd:0')
        self.maps = self.graph.get_tensor_by_name('minutiae_map:0')
        #self.moutputs = self.graph.get_tensor_by_name('InceptionV4/InceptionV4/MinutiaeFeatures/BiasAdd:0')
        self.images = self.graph.get_tensor_by_name('transformed_image:0')

    def to_lite(self):
      #converter = tf.contrib.lite.TFLiteConverter.from_session(self.sess, [self.inputs, self.phase_train_placeholder, self.keep_prob_placeholder], [self.outputs, self.moutputs])
      converter = tf.contrib.lite.TFLiteConverter.from_session(self.sess, [self.inputs], [self.outputs, self.moutputs])
      tflite_model = converter.convert()
      open("converted_model.tflite", "wb").write(tflite_model)

    def extract_feature(self, images, batch_size, verbose=False):
        num_images = images.shape[0] if type(images)==np.ndarray else len(images)
        num_features = self.outputs.shape[1]
        result = np.ndarray((num_images, num_features), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            inputs = images[start_idx:end_idx]
            feed_dict = {self.inputs: inputs,
                        self.phase_train_placeholder: False,
                    self.keep_prob_placeholder: 1.0}
            result[start_idx:end_idx] = self.sess.run(self.outputs, feed_dict=feed_dict)
        return result

    def extract_maps(self, images, batch_size, verbose=False):
        result = []
        num_images = images.shape[0] if type(images)==np.ndarray else len(images)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            inputs = images[start_idx:end_idx]
            feed_dict = {self.inputs: inputs,
                        self.phase_train_placeholder: False,
                    self.keep_prob_placeholder: 1.0}
            batch_maps = self.sess.run(self.maps, feed_dict=feed_dict)
            result.extend(batch_maps)
        return result

    def extract_multifeatures(self, images, batch_size, verbose=False):
        num_images = images.shape[0] if type(images)==np.ndarray else len(images)
        num_features = self.outputs.shape[1]
        result = np.ndarray((num_images, num_features), dtype=np.float32)
        result2 = np.ndarray((num_images, num_features), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            inputs = images[start_idx:end_idx]
            feed_dict = {self.inputs: inputs,
                        self.phase_train_placeholder: False,
                    self.keep_prob_placeholder: 1.0}
            outputs, moutputs = self.sess.run([self.outputs, self.moutputs], feed_dict=feed_dict)
            result[start_idx:end_idx] = outputs
            result2[start_idx:end_idx] = moutputs
        return result, result2
    
    """
    def extract_theta(self, images, batch_size, verbose=False):
        num_images = images.shape[0] if type(images)==np.ndarray else len(images)
        num_thetas = self.thetas.shape[1]
        result = np.ndarray((num_images, num_thetas), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            inputs = images[start_idx:end_idx]
            feed_dict = {self.inputs: inputs,
                        self.phase_train_placeholder: False,
                    self.keep_prob_placeholder: 1.0}
            result[start_idx:end_idx] = self.sess.run(self.thetas, feed_dict=feed_dict)
        return result
    """
    
    def transformed_images(self, images, batch_size, verbose=False):
        result = []
        num_images = images.shape[0] if type(images)==np.ndarray else len(images)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            inputs = images[start_idx:end_idx]
            feed_dict = {self.inputs: inputs,
                        self.phase_train_placeholder: False,
                    self.keep_prob_placeholder: 1.0}
            transformed_images = self.sess.run(self.images, feed_dict=feed_dict)
            num = transformed_images.shape[0]
            for b in range(0, num):
              img = transformed_images[b, :, :, :] * 127.5 + 127.5
              result.append(img)
        return result

        
