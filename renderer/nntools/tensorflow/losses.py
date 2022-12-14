import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from . import watcher as tfwatcher
from .metric_loss_ops import triplet_semihard_loss, npairs_loss, lifted_struct_loss

def center_loss(features, labels, num_classes, alpha=0.5, coef=0.05, scope='CenterLoss', reuse=None):
    num_features = features.shape[1].value
    batch_size = tf.shape(features)[0]
    with tf.variable_scope(scope, reuse=reuse):
        centers = tf.get_variable('centers', shape=(num_classes, num_features),
                # initializer=slim.xavier_initializer(),
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES],
                dtype=tf.float32)

        centers_batch = tf.gather(centers, labels)
        diff_centers = centers_batch - features

        loss = coef * 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(diff_centers), axis=1), name='center_loss')

        # Update centers
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        diff_centers = diff_centers / tf.cast(appear_times, tf.float32)
        diff_centers = alpha * diff_centers
        centers_update_op = tf.scatter_sub(centers, labels, diff_centers)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)

        return loss

def diam_softmax(prelogits, label, num_classes, scope,
                    scale='auto', m=1.0, alpha=0.5, reuse=None):
    ''' Implementation of DIAM-Softmax, AM-Softmax with Dynamic Weight Imprinting (DWI), proposed in:
            Y. Shi and A. K. Jain. DocFace+: ID Document to Selfie Matching. arXiv:1809.05620, 2018.
        The weights in the DIAM-Softmax are dynamically updated using the mean features of training samples.
    '''
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('AM-Softmax_{}'.format(scope), reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, num_features),
                initializer=slim.xavier_initializer(),
                trainable=False,
                collections=[tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES],
                dtype=tf.float32)
        _scale = tf.get_variable('_scale', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(0.0),
                trainable=True,
                dtype=tf.float32)

        # Normalizing the vecotors
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)
        weights_normed = tf.nn.l2_normalize(weights, dim=1)

        # Label and logits between batch and examplars
        label_mat_glob = tf.one_hot(label, num_classes, dtype=tf.float32)
        label_mask_pos_glob = tf.cast(label_mat_glob, tf.bool)
        label_mask_neg_glob = tf.logical_not(label_mask_pos_glob)

        logits_glob = tf.matmul(prelogits_normed, tf.transpose(weights_normed))
        # logits_glob = -0.5 * euclidean_distance(prelogits_normed, tf.transpose(weights_normed))
        logits_pos_glob = tf.boolean_mask(logits_glob, label_mask_pos_glob)
        logits_neg_glob = tf.boolean_mask(logits_glob, label_mask_neg_glob)

        logits_pos = logits_pos_glob
        logits_neg = logits_neg_glob

        if scale == 'auto':
            # Automatic learned scale
            scale = tf.log(tf.exp(0.0) + tf.exp(_scale))
        else:
            # Assigned scale value
            assert type(scale) == float

        # Losses
        _logits_pos = tf.reshape(logits_pos, [batch_size, -1])
        _logits_neg = tf.reshape(logits_neg, [batch_size, -1])

        _logits_pos = _logits_pos * scale
        _logits_neg = _logits_neg * scale
        _logits_neg = tf.reduce_logsumexp(_logits_neg, axis=1)[:,None]

        loss_ = tf.nn.softplus(m + _logits_neg - _logits_pos)
        loss = tf.reduce_mean(loss_, name='diam_softmax')

        # Dynamic weight imprinting
        # We follow the CenterLoss to update the weights, which is equivalent to
        # imprinting the mean features
        weights_batch = tf.gather(weights, label)
        diff_weights = weights_batch - prelogits_normed
        unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        diff_weights = diff_weights / tf.cast(appear_times, tf.float32)
        diff_weights = alpha * diff_weights
        weights_update_op = tf.scatter_sub(weights, label, diff_weights)
        with tf.control_dependencies([weights_update_op]):
            weights_update_op = tf.assign(weights, tf.nn.l2_normalize(weights,dim=1))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, weights_update_op)
        
        return loss

def euclidean_distance(X, Y, sqrt=False):
    '''Compute the distance between each X and Y.

    Args: 
        X: a (m x d) tensor
        Y: a (d x n) tensor
    
    Returns: 
        diffs: an m x n distance matrix.
    '''
    with tf.name_scope('EuclideanDistance'):
        XX = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
        YY = tf.reduce_sum(tf.square(Y), 0, keep_dims=True)
        XY = tf.matmul(X, Y)
        diffs = XX + YY - 2*XY
        diffs = tf.maximum(0.0, diffs)
        if sqrt == True:
            diffs = tf.sqrt(diffs)
    return diffs

def cosine_softmax(prelogits, label, num_classes, weight_decay, gamma=16.0, reuse=None):
    
    nrof_features = prelogits.shape[1].value
    
    with tf.variable_scope('Logits', reuse=reuse):
        weights = tf.get_variable('weights', shape=(nrof_features, num_classes),
                regularizer=slim.l2_regularizer(weight_decay),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.1),
                dtype=tf.float32)
        alpha = tf.get_variable('alpha', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)

        weights_normed = tf.nn.l2_normalize(weights, dim=0)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)

        if gamma == 'auto':
            gamma = tf.nn.softplus(alpha)
        else:
            assert type(gamma) == float
            gamma = tf.constant(gamma)

        logits = gamma * tf.matmul(prelogits_normed, weights_normed)


    cross_entropy =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
            labels=label, logits=logits), name='cross_entropy')

    tf.summary.scalar('gamma', gamma)
    tf.add_to_collection('watch_list', ('gamma', gamma))

    return logits, cross_entropy

def norm_loss(prelogits, alpha, reuse=None):
    with tf.variable_scope('NormLoss', reuse=reuse):
        sigma = tf.get_variable('sigma', shape=(),
            # regularizer=slim.l2_regularizer(weight_decay),
            initializer=tf.constant_initializer(0.1),
            trainable=True,
            dtype=tf.float32)
    prelogits_norm = tf.reduce_sum(tf.square(prelogits), axis=1)
    # norm_loss = alpha * tf.square(tf.sqrt(prelogits_norm) - sigma)
    norm_loss = alpha * prelogits_norm
    norm_loss = tf.reduce_mean(norm_loss, axis=0, name='norm_loss')

    # tf.summary.scalar('sigma', sigma)
    # tf.add_to_collection('watch_list', ('sigma', sigma))
    return norm_loss

def angular_softmax(prelogits, label, num_classes, global_step, 
            m, lamb_min, lamb_max, weight_decay, scope='AngularSoftmax', reuse=None):
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    lamb_min = lamb_min
    lamb_max = lamb_max
    lambda_m_theta = [
        lambda x: x**0,
        lambda x: x**1,
        lambda x: 2.0*(x**2) - 1.0,
        lambda x: 4.0*(x**3) - 3.0*x,
        lambda x: 8.0*(x**4) - 8.0*(x**2) + 1.0,
        lambda x: 16.0*(x**5) - 20.0*(x**3) + 5.0*x
    ]

    with tf.variable_scope(scope, reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_features, num_classes),
                regularizer=slim.l2_regularizer(1e-4),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.1),
                trainable=True,
                dtype=tf.float32)
        lamb = tf.get_variable('lambda', shape=(),
                initializer=tf.constant_initializer(lamb_max),
                trainable=False,
                dtype=tf.float32)
        prelogits_norm  = tf.sqrt(tf.reduce_sum(tf.square(prelogits), axis=1, keep_dims=True))
        weights_normed = tf.nn.l2_normalize(weights, dim=0)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)

        # Compute cosine and phi
        cos_theta = tf.matmul(prelogits_normed, weights_normed)
        cos_theta = tf.minimum(1.0, tf.maximum(-1.0, cos_theta))
        theta = tf.acos(cos_theta)
        cos_m_theta = lambda_m_theta[m](cos_theta)
        k = tf.floor(m*theta / 3.14159265)
        phi_theta = tf.pow(-1.0, k) * cos_m_theta - 2.0 * k

        cos_theta = cos_theta * prelogits_norm
        phi_theta = phi_theta * prelogits_norm

        lamb_new = tf.maximum(lamb_min, lamb_max/(1.0+0.1*tf.cast(global_step, tf.float32)))
        update_lamb = tf.assign(lamb, lamb_new)
        
        # Compute loss
        with tf.control_dependencies([update_lamb]):
            label_dense = tf.one_hot(label, num_classes, dtype=tf.float32)

            logits = cos_theta
            logits -= label_dense * cos_theta * 1.0 / (1.0+lamb)
            logits += label_dense * phi_theta * 1.0 / (1.0+lamb)
            
            cross_entropy =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                labels=label, logits=logits), name='cross_entropy')

        tf.add_to_collection('watch_list', ('lamb', lamb))

    return cross_entropy


def am_softmax(prelogits, label, num_classes, global_step, weight_decay, 
                scale=16.0, m=1.0, alpha='auto', reuse=None):
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    weights_trainable = alpha=='auto'
    with tf.variable_scope('AM-Softmax', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, num_features),
                # regularizer=slim.l2_regularizer(weight_decay),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.0),
                # initializer=tf.constant_initializer(0),
                trainable=weights_trainable,
                dtype=tf.float32)
        _scale = tf.get_variable('_scale', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)

        tf.add_to_collection('classifier_weights', weights)

        # Normalizing the vecotors
        weights_normed = tf.nn.l2_normalize(weights, dim=1)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)

        # Label and logits between batch and examplars
        label_mat_glob = tf.one_hot(label, num_classes, dtype=tf.float32)
        label_mask_pos_glob = tf.cast(label_mat_glob, tf.bool)
        label_mask_neg_glob = tf.logical_not(label_mask_pos_glob)

        dist_mat_glob = tf.matmul(prelogits_normed, tf.transpose(weights_normed))
        dist_pos_glob = tf.boolean_mask(dist_mat_glob, label_mask_pos_glob)
        # dist_mat_glob = tf.matmul(prelogits_normed, tf.transpose(tf.stop_gradient(weights_normed)))
        dist_neg_glob = tf.boolean_mask(dist_mat_glob, label_mask_neg_glob)

        logits_glob = dist_mat_glob
        logits_pos_glob = tf.boolean_mask(logits_glob, label_mask_pos_glob)
        logits_neg_glob = tf.boolean_mask(logits_glob, label_mask_neg_glob)

        logits_pos = logits_pos_glob
        logits_neg = logits_neg_glob

        if scale == 'auto':
            # Automatic learned scale
            scale = tf.log(tf.exp(1.0) + tf.exp(_scale))
        else:
            # Assigned scale value
            assert type(scale) == float
            scale = tf.constant(scale)

        # Losses
        _logits_pos = tf.reshape(logits_pos, [batch_size, -1])
        _logits_neg = tf.reshape(logits_neg, [batch_size, -1])

        _logits_pos = _logits_pos * scale
        _logits_neg = _logits_neg * scale
        _logits_neg = tf.reduce_logsumexp(_logits_neg, axis=1)[:,None]

        loss_ = tf.nn.relu(m + _logits_neg - _logits_pos)
        loss = tf.reduce_mean(loss_, name='am_softmax')

        # Update centers
        if not weights in tf.trainable_variables():
            weights_batch = tf.gather(weights, label)
            diff_centers = weights_batch - prelogits_normed
            unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
            appear_times = tf.gather(unique_count, unique_idx)
            appear_times = tf.reshape(appear_times, [-1, 1])
            diff_centers = diff_centers / tf.cast((1 + appear_times), tf.float32)
            diff_centers = alpha * diff_centers
    
            # decay_op = tf.assign(weights, 0.99*weights)
            # with tf.control_dependencies([decay_op]):
            # new_centers = tf.nn.l2_normalize(weights_batch - diff_centers, dim=1)
            # centers_update_op = tf.scatter_update(weights, label, new_centers)

            centers_update_op = tf.scatter_sub(weights, label, diff_centers)
            with tf.control_dependencies([centers_update_op]):
                centers_update_op = tf.assign(weights, tf.nn.l2_normalize(weights,dim=1))
            centers_update_op = tf.group(centers_update_op)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)

        # Analysis
        tfwatcher.insert('scale', scale)

    return loss

def am_softmax_imprint(prelogits, label, num_classes, global_step, weight_decay, 
                scale=16.0, m=1.0, alpha='auto', reuse=None):
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('AM-Softmax', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, num_features),
                regularizer=slim.l2_regularizer(weight_decay),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.0),
                # initializer=tf.constant_initializer(0),
                trainable=False,
                dtype=tf.float32)
        _scale = tf.get_variable('_scale', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)

        tf.add_to_collection('classifier_weights', weights)

        # Normalizing the vecotors
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)

        # Update centers
        if not weights in tf.trainable_variables():
            weights_batch = tf.gather(weights, label)
            diff_centers = weights_batch - prelogits_normed
            unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
            appear_times = tf.gather(unique_count, unique_idx)
            appear_times = tf.reshape(appear_times, [-1, 1])
            diff_centers = diff_centers / tf.cast((1 + appear_times), tf.float32)
            diff_centers = alpha * diff_centers
    
            centers_update_op = tf.scatter_sub(weights, label, diff_centers)
            with tf.control_dependencies([centers_update_op]):
                centers_update_op = tf.assign(weights, tf.nn.l2_normalize(weights,dim=1))
            # centers_update_op = tf.group(centers_update_op)
            # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)

        with tf.control_dependencies([centers_update_op]):
            weights_normed = tf.nn.l2_normalize(weights, dim=1)

            idx_odd = tf.range(start=0, limit=batch_size-1, delta=2)
            idx_even = tf.range(start=1, limit=batch_size, delta=2)
            idx = tf.reshape(tf.stack([idx_even, idx_odd], axis=1), [batch_size])
            prelogits_reorder = tf.stop_gradient(tf.gather(prelogits_normed, idx))

            # Label and logits between batch and examplars
            label_mat_glob = tf.one_hot(label, num_classes, dtype=tf.float32)
            label_mask_pos_glob = tf.cast(label_mat_glob, tf.bool)
            label_mask_neg_glob = tf.logical_not(label_mask_pos_glob)

            logits_glob = tf.matmul(prelogits_normed, tf.transpose(weights_normed))
            # logits_pos_glob = tf.reduce_sum(prelogits_normed * prelogits_reorder, axis=1)
            # tfwatcher.insert('dimp', tf.shape(logits_pos_glob)[0])
            # print('Positive shape: {}'.format(logits_pos_glob.shape))
            logits_pos_glob = tf.boolean_mask(logits_glob, label_mask_pos_glob)
            logits_neg_glob = tf.boolean_mask(logits_glob, label_mask_neg_glob)

            logits_pos = logits_pos_glob
            logits_neg = logits_neg_glob

            if scale == 'auto':
                # Automatic learned scale
                scale = tf.log(tf.exp(1.0) + tf.exp(_scale))
            else:
                # Assigned scale value
                assert type(scale) == float
                scale = tf.constant(scale)

            # Losses
            _logits_pos = tf.reshape(logits_pos, [batch_size, -1])
            _logits_neg = tf.reshape(logits_neg, [batch_size, -1])

            _logits_pos = _logits_pos * scale
            _logits_neg = _logits_neg * scale
            _logits_neg = tf.reduce_logsumexp(_logits_neg, axis=1)[:,None]

            loss_ = tf.nn.relu(m + _logits_neg - _logits_pos)
            loss = tf.reduce_mean(loss_, name='am_softmax')


        # Analysis
        tfwatcher.insert('scale', scale)

    return loss


def batch_norm(x):
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 1e-8,
        'center': True,
        'scale': True,
        'updates_collections': None,
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
        'param_initializers': {'gamma': tf.constant_initializer(0.1)},
    }
    x_normed = slim.batch_norm(x, **batch_norm_params)
    return x_normed

def split_softmax(prelogits, label, num_classes, 
                global_step, weight_decay, gamma=16.0, m=1.0, reuse=None):
    nrof_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('SplitSoftmax', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, nrof_features),
                regularizer=slim.l2_regularizer(weight_decay),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.5),
                # initializer=tf.constant_initializer(0),
                trainable=True,
                dtype=tf.float32)
        alpha = tf.get_variable('alpha', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)
        beta = tf.get_variable('beta', shape=(),
                # regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(0.0),
                trainable=True,
                dtype=tf.float32)

        # Normalizing the vecotors
        # weights_normed = weights
        weights_normed = tf.nn.l2_normalize(weights, dim=1)
        # prelogits_normed = prelogits
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)
        # norm_ = tf.norm(prelogits_normed, axis=1)
        # tfwatcher.insert('pnorm', tf.reduce_mean(norm_))
        

        coef = 1.0
        # Label and logits between batch and examplars
        label_mat_glob = tf.one_hot(label, num_classes, dtype=tf.float32)
        label_mask_pos_glob = tf.cast(label_mat_glob, tf.bool)
        label_mask_neg_glob = tf.logical_not(label_mask_pos_glob)
        # label_exp_batch = tf.expand_dims(label, 1)
        # label_exp_glob = tf.expand_dims(label_history, 1)
        # label_mat_glob = tf.equal(label_exp_batch, tf.transpose(label_exp_glob))
        # label_mask_pos_glob = tf.cast(label_mat_glob, tf.bool)
        # label_mask_neg_glob = tf.logical_not(label_mat_glob)

        # dist_mat_glob = euclidean_distance(prelogits_normed, tf.transpose(weights_normed), False)
        dist_mat_glob = tf.matmul(prelogits_normed, tf.transpose(weights_normed))
        dist_pos_glob = tf.boolean_mask(dist_mat_glob, label_mask_pos_glob)
        dist_neg_glob = tf.boolean_mask(dist_mat_glob, label_mask_neg_glob)

        logits_glob = coef * dist_mat_glob
        logits_pos_glob = tf.boolean_mask(logits_glob, label_mask_pos_glob)
        logits_neg_glob = tf.boolean_mask(logits_glob, label_mask_neg_glob)


        # Label and logits within batch
        label_exp_batch = tf.expand_dims(label, 1)
        label_mat_batch = tf.equal(label_exp_batch, tf.transpose(label_exp_batch))
        label_mask_pos_batch = tf.cast(label_mat_batch, tf.bool)
        label_mask_neg_batch = tf.logical_not(label_mask_pos_batch)
        mask_non_diag = tf.logical_not(tf.cast(tf.eye(batch_size), tf.bool))
        label_mask_pos_batch = tf.logical_and(label_mask_pos_batch, mask_non_diag)

        # dist_mat_batch = euclidean_distance(prelogits_normed, tf.transpose(prelogits_normed), False)
        dist_mat_batch = tf.matmul(prelogits_normed, tf.transpose(prelogits_normed))
        dist_pos_batch = tf.boolean_mask(dist_mat_batch, label_mask_pos_batch)
        dist_neg_batch = tf.boolean_mask(dist_mat_batch, label_mask_neg_batch)

        logits_batch =  coef * dist_mat_batch
        logits_pos_batch = tf.boolean_mask(logits_batch, label_mask_pos_batch)
        logits_neg_batch = tf.boolean_mask(logits_batch, label_mask_neg_batch)
        

        logits_pos = logits_pos_glob
        logits_neg = logits_neg_glob


        if gamma == 'auto':
            # gamma = tf.nn.softplus(alpha)
            gamma = tf.log(tf.exp(1.0) + tf.exp(alpha))
        elif type(gamma) == tuple:
            t_min, decay = gamma
            epsilon = 1.0
            t = tf.maximum(t_min, 1.0/(epsilon + decay*tf.cast(global_step, tf.float32)))
            gamma = 1.0 / t
        else:
            assert type(gamma) == float
            gamma = tf.constant(gamma)

        # Losses

        t_pos = (beta)
        t_neg = (beta)

        _logits_pos = tf.reshape(logits_pos, [batch_size, -1])
        _logits_neg = tf.reshape(logits_neg, [batch_size, -1])

        _logits_pos = _logits_pos # * gamma
        _logits_neg = _logits_neg # * gamma
        # _logits_neg = tf.reduce_logsumexp(_logits_neg, axis=1)[:,None]
        
        # _logits_neg = -tf.reduce_max(-_logits_neg, axis=1)[:,None]
        
        tfwatcher.insert('lneg', tf.reduce_mean(_logits_neg))
        tfwatcher.insert('lpos', tf.reduce_mean(_logits_pos))
        

        num_violate = tf.reduce_sum(tf.cast(tf.greater(m + _logits_neg - _logits_pos, 0.), tf.float32), axis=1, keep_dims=True)
        loss =  tf.reduce_sum(tf.nn.relu(m + _logits_neg - _logits_pos), axis=1, keep_dims=True) / (num_violate + 1e-8)
        tfwatcher.insert('nv', tf.reduce_mean(num_violate))
        
        # loss = tf.nn.softplus(m + _logits_neg - _logits_pos)

        loss = tf.reduce_mean(30*loss, name='split_loss')


        # Update centers
        if not weights in tf.trainable_variables():
            weights_batch = tf.gather(weights, label)
            diff_centers = weights_batch - prelogits_normed
            unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
            appear_times = tf.gather(unique_count, unique_idx)
            appear_times = tf.reshape(appear_times, [-1, 1])
            diff_centers = diff_centers / tf.cast((1 + appear_times), tf.float32)
            diff_centers = 0.5 * diff_centers
            centers_update_op = tf.scatter_sub(weights, label, diff_centers)
            with tf.control_dependencies([centers_update_op]):
                centers_update_op = tf.assign(weights, tf.nn.l2_normalize(weights,dim=1))
            # centers_decay_op = tf.assign_sub(weights, 2*weight_decay*weights)# weight decay
            centers_update_op = tf.group(centers_update_op)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)

        # Analysis
        tf.summary.scalar('gamma', gamma)
        # tf.summary.scalar('alpha', alpha)
        # tf.summary.scalar('beta', beta)
        tfwatcher.insert('gamma', gamma)
        # tfwatcher.insert('beta', beta)

    return loss

def centers_by_label(features, label):
    # Compute centers within batch
    unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
    num_centers = tf.size(unique_label)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    weighted_prelogits = features / tf.cast(appear_times, tf.float32)
    centers = tf.unsorted_segment_sum(weighted_prelogits, unique_idx, num_centers)
    return centers, unique_label, unique_idx, unique_count


def pair_loss(prelogits, label, num_classes, 
                global_step, weight_decay, gamma=16.0, m=1.0, reuse=None):
    nrof_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('PairLoss', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, nrof_features),
                regularizer=slim.l2_regularizer(weight_decay),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.0),
                # initializer=tf.constant_initializer(0),
                trainable=True,
                dtype=tf.float32)
        alpha = tf.get_variable('alpha', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)
        beta = tf.get_variable('beta', shape=(),
                # regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(0.0),
                trainable=True,
                dtype=tf.float32)

        # Normalizing the vecotors
        weights_normed = tf.nn.l2_normalize(weights, dim=1)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)
        # weights_normed = weights
        # prelogits_normed = prelogits

        prelogits_reshape = tf.reshape(prelogits_normed, [-1,2,tf.shape(prelogits_normed)[1]])
        prelogits_tmp = prelogits_reshape[:,0,:]
        prelogits_pro = prelogits_reshape[:,1,:]
    
        dist_mat_batch = -euclidean_distance(prelogits_tmp, tf.transpose(prelogits_pro), False)
        # dist_mat_batch = tf.matmul(prelogits_tmp, tf.transpose(prelogits_pro))
        
        logits_mat_batch = dist_mat_batch

        num_pairs = tf.shape(prelogits_reshape)[0]
        label_mask_pos_batch = tf.cast(tf.eye(num_pairs), tf.bool)
        label_mask_neg_batch = tf.logical_not(label_mask_pos_batch)
        dist_pos_batch = tf.boolean_mask(dist_mat_batch, label_mask_pos_batch)
        dist_neg_batch = tf.boolean_mask(dist_mat_batch, label_mask_neg_batch)

        logits_pos_batch = tf.boolean_mask(logits_mat_batch, label_mask_pos_batch)
        logits_neg_batch = tf.boolean_mask(logits_mat_batch, label_mask_neg_batch)

        logits_pos = logits_pos_batch
        logits_neg = logits_neg_batch
    
        dist_pos = dist_pos_batch
        dist_neg = dist_neg_batch


        if gamma == 'auto':
            # gamma = tf.nn.softplus(alpha)
            gamma = tf.log(tf.exp(1.0) + tf.exp(alpha))
        elif type(gamma) == tuple:
            t_min, decay = gamma
            epsilon = 1.0
            t = tf.maximum(t_min, 1.0/(epsilon + decay*tf.cast(global_step, tf.float32)))
            gamma = 1.0 / t
        else:
            assert type(gamma) == float
            gamma = tf.constant(gamma)

        hinge_loss = lambda x: tf.nn.relu(1.0 + x)
        margin_func = hinge_loss

        # Losses
        losses = []

        t_pos = (beta)
        t_neg = (beta)

        _logits_pos = tf.reshape(logits_pos, [num_pairs, -1])
        _logits_neg_1 = tf.reshape(logits_neg, [num_pairs, -1])
        _logits_neg_2 = tf.reshape(logits_neg, [-1, num_pairs])

        _logits_pos = _logits_pos
        _logits_neg_1 = tf.reduce_max(_logits_neg_1, axis=1)[:,None]
        _logits_neg_2 = tf.reduce_max(_logits_neg_2, axis=0)[:,None]
        _logits_neg = tf.maximum(_logits_neg_1, _logits_neg_2)
        # _logits_neg = tf.reduce_logsumexp(_logits_neg, axis=1)[:,None]

        loss_pos = tf.nn.relu(m + _logits_neg - _logits_pos) * 0.5
        loss_neg = tf.nn.relu(m + _logits_neg - _logits_pos) * 0.5
        loss = tf.reduce_mean(loss_pos + loss_neg)
        loss = tf.identity(loss, name='pair_loss')
        losses.extend([loss])
        tfwatcher.insert('ploss', loss)

        # Analysis
        tf.summary.scalar('gamma', gamma)
        tf.summary.scalar('alpha', alpha)
        tf.summary.scalar('beta', beta)
        tf.summary.histogram('dist_pos', dist_pos)
        tf.summary.histogram('dist_neg', dist_neg)

        tfwatcher.insert('gamma', gamma)

    return losses



def pair_loss_twin(prelogits_tmp, prelogits_pro, label_tmp, label_pro, num_classes, 
                global_step, weight_decay, gamma=16.0, m=1.0, reuse=None):
    num_features = prelogits_tmp.shape[1].value
    batch_size = tf.shape(prelogits_tmp)[0] + tf.shape(prelogits_pro)[0]
    with tf.variable_scope('PairLoss', reuse=reuse):
        alpha = tf.get_variable('alpha', shape=(),
                # regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)
        beta = tf.get_variable('beta', shape=(),
                # regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(0.0),
                trainable=True,
                dtype=tf.float32)

        # Normalizing the vecotors
        prelogits_tmp = tf.nn.l2_normalize(prelogits_tmp, dim=1)
        prelogits_pro = tf.nn.l2_normalize(prelogits_pro, dim=1)
    
        dist_mat_batch = -euclidean_distance(prelogits_tmp, tf.transpose(prelogits_pro),False)
        # dist_mat_batch = tf.matmul(prelogits_tmp, tf.transpose(prelogits_pro))
        
        logits_mat_batch = dist_mat_batch

        num_pairs = tf.shape(prelogits_tmp)[0]

        # label_tmp_batch = tf.expand_dims(label_tmp, 1)
        # label_pro_batch = tf.expand_dims(label_pro, 1)
        # label_mat_batch = tf.equal(label_tmp_batch, tf.transpose(label_pro_batch))
        # label_mask_pos_batch = tf.cast(label_mat_batch, tf.bool)
        label_mask_pos_batch = tf.cast(tf.eye(num_pairs), tf.bool)
        label_mask_neg_batch = tf.logical_not(label_mask_pos_batch)

        logits_pos = tf.boolean_mask(logits_mat_batch, label_mask_pos_batch)
        logits_neg_1 = tf.boolean_mask(logits_mat_batch, label_mask_neg_batch)
        logits_neg_2 = tf.boolean_mask(tf.transpose(logits_mat_batch), label_mask_neg_batch)

        if gamma == 'auto':
            # gamma = tf.nn.softplus(alpha)
            gamma = tf.log(tf.exp(1.0) + tf.exp(alpha))
        elif type(gamma) == tuple:
            t_min, decay = gamma
            epsilon = 1.0
            t = tf.maximum(t_min, 1.0/(epsilon + decay*tf.cast(global_step, tf.float32)))
            gamma = 1.0 / t
        else:
            assert type(gamma) == float
            gamma = tf.constant(gamma)

        # Losses
        losses = []

        t_pos = (beta)
        t_neg = (beta)

        _logits_pos = tf.reshape(logits_pos, [num_pairs, -1])
        _logits_neg_1 = tf.reshape(logits_neg_1, [num_pairs, -1])
        _logits_neg_2 = tf.reshape(logits_neg_2, [num_pairs, -1])
        
        
        _logits_neg_1 = tf.reduce_max(_logits_neg_1, axis=1)[:,None]
        _logits_neg_2 = tf.reduce_max(_logits_neg_2, axis=1)[:,None]
        _logits_neg = tf.maximum(_logits_neg_1, _logits_neg_2)
        # _logits_neg = tf.concat([_logits_neg_1, _logits_neg_2], axis=1)
        # _logits_neg = tf.reduce_logsumexp(_logits_neg, axis=1)[:,None]


        num_violate = tf.reduce_sum(tf.cast(tf.greater(m + _logits_neg - _logits_pos, 0.), tf.float32), axis=1, keep_dims=True)

        loss_1 = tf.reduce_sum(tf.nn.relu(m + _logits_neg - _logits_pos), axis=1, keep_dims=True) * 0.5 # / (num_violate + 1e-8)
        loss_2 = tf.reduce_sum(tf.nn.relu(m + _logits_neg - _logits_pos), axis=1, keep_dims=True) * 0.5 # / (num_violate + 1e-8)
        loss = tf.reduce_mean(loss_1 + loss_2)
        loss = tf.identity(loss, name='pair_loss')
        losses.extend([loss])

        # Analysis
        tf.summary.scalar('gamma', gamma)
        tf.summary.scalar('alpha', alpha)
        tf.summary.scalar('beta', beta)
        tf.summary.histogram('dist_pos', _logits_pos)
        tf.summary.histogram('dist_neg', _logits_neg)

        tfwatcher.insert("gamma", gamma)

    return losses

def l2centers(features, label, centers, coef):
    centers_batch = tf.gather(centers, label)
    loss = tf.reduce_mean(coef * tf.reduce_sum(tf.square(features - centers_batch), axis=1), name='l2centers')
    
    return loss

def cosine_regression(features, targets, coef):
    features = tf.nn.l2_normalize(features, dim=1)
    targets = tf.nn.l2_normalize(targets, dim=1)

    loss = coef * tf.reduce_mean(tf.reduce_sum(tf.square(features - targets), axis=1))
    
    return loss

def triplet_loss(labels, embeddings, margin):

    with tf.name_scope('TripletLoss'):

        embeddings = tf.nn.l2_normalize(embeddings, dim=1)

        batch_size = tf.shape(embeddings)[0]
        num_features = embeddings.shape[1].value


        diag_mask = tf.eye(batch_size, dtype=tf.bool)
        non_diag_mask = tf.logical_not(diag_mask)    

        dist_mat = euclidean_distance(embeddings, tf.transpose(embeddings), False)
        

        label_mat = tf.equal(labels[:,None], labels[None,:])
        label_mask_pos = tf.logical_and(non_diag_mask, label_mat)
        label_mask_neg = tf.logical_and(non_diag_mask, tf.logical_not(label_mat))

        
        dist_pos = tf.boolean_mask(dist_mat, label_mask_pos) 
        dist_neg = tf.boolean_mask(dist_mat, label_mask_neg)
        
        dist_pos = tf.reshape(dist_pos, [batch_size, -1])
        dist_neg = tf.reshape(dist_neg, [batch_size, -1])


        # Hard Negative Mining
        dist_neg = -tf.reduce_max(-dist_neg, axis=1, keep_dims=True)
        
        loss = tf.nn.relu(dist_pos - dist_neg + margin)
        loss = tf.reduce_mean(loss, name='TripletLoss')
    
    return loss

def contrastive_loss(labels, embeddings, margin):

    with tf.name_scope('ContrastiveLoss'):
        embeddings = tf.nn.l2_normalize(embeddings, dim=1)

        batch_size = tf.shape(embeddings)[0]
        num_features = embeddings.shape[1].value


        diag_mask = tf.eye(batch_size, dtype=tf.bool)
        non_diag_mask = tf.logical_not(diag_mask)    

        dist_mat = euclidean_distance(embeddings, tf.transpose(embeddings), False)
        

        label_mat = tf.equal(labels[:,None], labels[None,:])
        label_mask_pos = tf.logical_and(non_diag_mask, label_mat)
        label_mask_neg = tf.logical_and(non_diag_mask, tf.logical_not(label_mat))

        
        dist_pos = tf.boolean_mask(dist_mat, label_mask_pos) 
        dist_neg = tf.boolean_mask(dist_mat, label_mask_neg)
        
        dist_pos = tf.reshape(dist_pos, [batch_size, -1])
        dist_neg = tf.reshape(dist_neg, [batch_size, -1])
        dist_neg = tf.reduce_min(dist_neg, axis=1, keep_dims=True)

        loss_pos = tf.reduce_mean(dist_pos)
        loss_neg = tf.reduce_mean(tf.nn.relu(margin - dist_neg))
        
        loss = tf.identity(0.5*loss_pos + 0.5*loss_neg, name='contrastive_loss')
    
    return loss

def triplet_loss_avghard(labels, embeddings, margin, normalize=False, scope='TripletLoss'):

    with tf.name_scope(scope):

        if normalize:
            embeddings = tf.nn.l2_normalize(embeddings, dim=1)

        batch_size = tf.shape(embeddings)[0]
        num_features = embeddings.shape[1].value


        diag_mask = tf.eye(batch_size, dtype=tf.bool)
        non_diag_mask = tf.logical_not(diag_mask)

        dist_mat = euclidean_distance(embeddings, tf.transpose(embeddings), False)

        label_mat = tf.equal(labels[:,None], labels[None,:])

        dist_mat_tile = tf.tile(tf.reshape(dist_mat, [batch_size, 1, -1]), [1, batch_size ,1])
        dist_mat_tile = tf.reshape(dist_mat_tile, [-1, batch_size])

        label_mat_tile = tf.tile(tf.reshape(label_mat, [batch_size, 1, -1]), [1, batch_size, 1])
        label_mat_tile = tf.reshape(label_mat_tile, [-1, batch_size])


        dist_flatten = tf.reshape(dist_mat, [-1, 1])
        label_flatten = tf.reshape(label_mat, [-1])

        loss = dist_flatten - dist_mat_tile + margin

        valid = tf.cast(tf.logical_and(tf.logical_not(label_mat_tile), tf.greater(loss, 0.0)), tf.float32)
        valid_count = tf.reduce_sum(valid, axis=1) + 1e-8

        loss = tf.nn.relu(loss)

        loss = tf.reduce_sum(loss * valid, axis=1) / valid_count

        loss = tf.boolean_mask(loss, label_flatten)

        loss = tf.reduce_mean(loss)

        return loss
