from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def inference(images, keep_probability, phase_train=True, weight_decay=0.0, reuse=None):
  with tf.variable_scope('Unet', reuse=tf.AUTO_REUSE) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=phase_train):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'):
        print("keep probability: {}".format(keep_probability))

        print('Unet input shape:', [dim.value for dim in images.shape])
        # net = slim.dropout(net, keep_probability, scope='Dropout_1b')
        net1 = slim.conv2d(images, 64, [3, 3], stride=1, padding='SAME', scope='Conv1')
        print('module 1 shape:', [dim.value for dim in net1.shape])

        net = slim.conv2d(net1, 128, [3, 3], stride=2, padding='SAME', scope='Conv2')
        print('module 2 shape:', [dim.value for dim in net.shape])

        net3 = slim.conv2d(net, 128, [3, 3], stride=1, padding='SAME', scope='Conv3')
        print('module 3 shape:', [dim.value for dim in net3.shape])

        net = slim.conv2d(net3, 256, [3, 3], stride=2, padding='SAME', scope='Conv4')
        print('module 4 shape:', [dim.value for dim in net.shape])

        net5 = slim.conv2d(net, 256, [3, 3], stride=1, padding='SAME', scope='Conv5')
        print('module 5 shape:', [dim.value for dim in net5.shape])

        net = slim.conv2d(net5, 512, [3, 3], stride=2, padding='SAME', scope='Conv6')
        print('module 6 shape:', [dim.value for dim in net.shape])

        net7 = slim.conv2d(net, 512, [3, 3], stride=1, padding='SAME', scope='Conv7')
        print('module 7 shape:', [dim.value for dim in net7.shape])

        net = slim.conv2d(net7, 512, [3, 3], stride=2, padding='SAME', scope='Conv8')
        print('module 8 shape:', [dim.value for dim in net.shape])

        net = slim.conv2d(net, 512, [3, 3], stride=1, padding='SAME', scope='Conv9')
        print('module 9 shape:', [dim.value for dim in net.shape])

        net = slim.conv2d(net, 512, [3, 3], stride=1, padding='SAME', scope='Conv10')
        print('module 10 shape:', [dim.value for dim in net.shape])

        net = slim.layers.conv2d_transpose(net, 512, [3, 3], stride=2, scope='DeConv2d_11')
        print('module 11 shape:', [dim.value for dim in net.shape])

        net = tf.concat([net, net7], 3)
        net = slim.conv2d(net, 512, [3, 3], stride=1, padding='SAME', scope='Conv12')
        print('module 12 shape:', [dim.value for dim in net.shape])

        net = slim.layers.conv2d_transpose(net, 256, [3, 3], stride=2, scope='DeConv2d_13')
        print('module 13 shape:', [dim.value for dim in net.shape])

        net = tf.concat([net, net5], 3)
        net = slim.conv2d(net, 256, [3, 3], stride=1, padding='SAME', scope='Conv14')
        print('module 14 shape:', [dim.value for dim in net.shape])

        net = slim.layers.conv2d_transpose(net, 128, [3, 3], stride=2, scope='DeConv2d_15')
        print('module 15 shape:', [dim.value for dim in net.shape])

        net = tf.concat([net, net3], 3)
        net = slim.conv2d(net, 128, [3, 3], stride=1, padding='SAME', scope='Conv16')
        print('module 16 shape:', [dim.value for dim in net.shape])

        net = slim.layers.conv2d_transpose(net, 64, [3, 3], stride=2, scope='DeConv2d_17')
        print('module 17 shape:', [dim.value for dim in net.shape])

        net = tf.concat([net, net1], 3)
        net = slim.conv2d(net, 12, [3, 3], stride=1, padding='SAME', scope='Conv18', activation_fn=None)
        print('output shape:', [dim.value for dim in net.shape])

        # net = tf.nn.sigmoid(net)

        return net