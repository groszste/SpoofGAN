import tensorflow as tf
import tensorflow.contrib.slim as slim
##################################################################################
# UTILS (taken from utils.py)
##################################################################################
#batch_norm_params = {
#    'decay': 0.995,
#    'epsilon': 0.001,
#    'updates_collections': None,
#    'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
#}


def orthogonal_regularizer(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w) :
        """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
        _, _, _, c = w.get_shape().as_list()

        w = tf.reshape(w, [-1, c])

        """ Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)

        """ Regularizer Wt*W - I """
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """Calculating the Loss Obtained"""
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg

def orthogonal_regularizer_fully(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Fully Connected Layer """

    def ortho_reg_fully(w) :
        """ Reshaping the matrix in to 2D tensor for enforcing orthogonality"""
        _, c = w.get_shape().as_list()

        """Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """ Calculating the Loss """
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg_fully

##################################################################################
# OPS (taken from ops.py)
##################################################################################

##################################################################################
# Initialization
##################################################################################

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# Truncated_normal : tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
# Orthogonal : tf.orthogonal_initializer(1.0) / relu = sqrt(2), the others = 1.0

##################################################################################
# Regularization
##################################################################################

# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)
# orthogonal_regularizer : orthogonal_regularizer(0.0001) / orthogonal_regularizer_fully(0.0001)

weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = orthogonal_regularizer(0.0001)
weight_regularizer_fully = orthogonal_regularizer_fully(0.0001)

# Regularization only G in BigGAN

##################################################################################
# Layer
##################################################################################

# pad = ceil[ (kernel - stride) / 2 ]

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero' :
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect' :
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn :
            if scope.__contains__('generator') :
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                    regularizer=weight_regularizer)
            else :
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                    regularizer=None)

            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            if scope.__contains__('generator'):
                x = tf.layers.conv2d(inputs=x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=weight_regularizer,
                                     strides=stride, use_bias=use_bias)
            else :
                x = tf.layers.conv2d(inputs=x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=None,
                                     strides=stride, use_bias=use_bias)


        return x


def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [tf.shape(x)[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        else:
            output_shape =[tf.shape(x)[0], x_shape[1] * stride + max(kernel - stride, 0), x_shape[2] * stride + max(kernel - stride, 0), channels]
        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x

def fully_conneted(x, units, use_bias=True, sn=False, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn :
            if scope.__contains__('generator'):
                w = tf.get_variable("kernel", [channels, units], tf.float32, initializer=weight_init, regularizer=weight_regularizer_fully)
            else :
                w = tf.get_variable("kernel", [channels, units], tf.float32, initializer=weight_init, regularizer=None)

            if use_bias :
                bias = tf.get_variable("bias", [units], initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else :
                x = tf.matmul(x, spectral_norm(w))

        else :
            if scope.__contains__('generator'):
                x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                    kernel_regularizer=weight_regularizer_fully, use_bias=use_bias)
            else :
                x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                    kernel_regularizer=None, use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

def hw_flatten(x) :
    h, w = x.get_shape().as_list()[1:3]
    flat_dim = h*w
    return tf.reshape(x, shape=[-1, flat_dim, x.shape[-1]])

##################################################################################
# Residual-block, Self-Attention-block
##################################################################################

def resblock(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)

        return x + x_init

def resblock_up(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock_up'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = batch_norm(x_init, is_training)
            x = relu(x)
            x = deconv(x, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2') :
            x = batch_norm(x, is_training)
            x = relu(x)
            x = deconv(x, channels, kernel=3, stride=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('skip') :
            x_init = deconv(x_init, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)


    return x + x_init

def resblock_up_condition(x_init, z, channels, use_bias=True, is_training=True, sn=False, scope='resblock_up'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = condition_batch_norm(x_init, z, is_training)
            x = relu(x)
            x = deconv(x, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2') :
            x = condition_batch_norm(x, z, is_training)
            x = relu(x)
            x = deconv(x, channels, kernel=3, stride=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('skip') :
            x_init = deconv(x_init, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)


    return x + x_init


def resblock_down(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock_down'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = batch_norm(x_init, is_training)
            x = relu(x)
            x = conv(x, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2') :
            x = batch_norm(x, is_training)
            x = relu(x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('skip') :
            x_init = conv(x_init, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, sn=sn)


    return x + x_init

def self_attention(x, channels, sn=False, scope='self_attention'):
    with tf.variable_scope(scope):
        f = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
        g = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']
        h = conv(x, channels, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
        x = gamma * o + x

    return x

def self_attention_2(x, channels, sn=False, scope='self_attention'):
    with tf.variable_scope(scope):
        f = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
        f = max_pooling(f)

        g = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']

        h = conv(x, channels // 2, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]
        h = max_pooling(h)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=[-1, x.shape[1], x.shape[2], channels // 2])  # [bs, h, w, C]
        o = conv(o, channels, kernel=1, stride=1, sn=sn, scope='attn_conv')
        x = gamma * o + x

    return x

##################################################################################
# Sampling
##################################################################################

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])

    return gap

def global_sum_pooling(x) :
    gsp = tf.reduce_sum(x, axis=[1, 2])

    return gsp

def max_pooling(x) :
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')
    return x

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf.layers.batch_normalization(x,
                                         momentum=0.9,
                                         epsilon=1e-05,
                                         training=is_training,
                                         name=scope)

def condition_batch_norm(x, z, is_training=True, scope='batch_norm', is_training_bool=True):
    """
    is_training = is_training_bool # quick hack
    with tf.variable_scope(scope) :
        _, _, _, c = x.get_shape().as_list()
        decay = 0.9
        epsilon = 1e-05

        test_mean = tf.get_variable("pop_mean", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
        test_var = tf.get_variable("pop_var", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=False)

        beta = fully_conneted(z, units=c, scope='beta')
        gamma = fully_conneted(z, units=c, scope='gamma')

        beta = tf.reshape(beta, shape=[-1, 1, 1, c])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, c])

        if is_training:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            ema_mean = tf.assign(test_mean, test_mean * decay + batch_mean * (1 - decay))
            ema_var = tf.assign(test_var, test_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([ema_mean, ema_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)
        else:
            return tf.nn.batch_normalization(x, test_mean, test_var, beta, gamma, epsilon)
    # quick HACK
    #return batch_norm(x, is_training, scope)
    """
    with tf.variable_scope(scope):
        _, _, _, c = x.get_shape().as_list()
        gamma = slim.fully_connected(z, c, activation_fn=None, normalizer_fn=None, scope='gamma_linear_layer')
        gamma = tf.reshape(gamma, [-1, 1, 1, c], name='gamma')
        beta = slim.fully_connected(z, c, activation_fn=None, normalizer_fn=None, scope='beta_linear_layer')
        beta = tf.reshape(beta, [-1, 1, 1, c], name='beta')
        return gamma * tf.contrib.layers.instance_norm(x, center=False, scale=False) + beta
    

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

##################################################################################
# Style Encoder
##################################################################################
def style_encoder(random_z, is_training, reuse=False):
    net = None
    with slim.arg_scope([slim.fully_connected],
             activation_fn=tf.nn.relu,
             weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
             weights_regularizer=slim.l2_regularizer(4e-5)):
        with tf.variable_scope('style_encoder', reuse=reuse):
            net = slim.fully_connected(random_z, 128, scope='fc_0')
            for i in range(3):
                net = slim.fully_connected(net, 128, scope='fc_{}'.format(i+1))
    return net
        

##################################################################################
# Generator
##################################################################################
def generator(minutiae, random_z, is_training=True, reuse=False):
        z_dim=512
        ch = 48 # 96
        sn = True
        c_dim = 1
        with tf.variable_scope("generator", reuse=reuse):
            # get style vector
            style = style_encoder(random_z, is_training, reuse)

            # compress minutiae map to 128
            #z = relu(batch_norm(conv(minutiae, channels=16, kernel=3, stride=2, pad=1, pad_type='zero', use_bias=True, sn=sn, scope='conv_256'), is_training, scope='conv_256_batch_norm'))
            #print("z shape: {}".format(z.shape))
            z = relu(batch_norm(conv(minutiae, channels=32, kernel=3, stride=2, pad=1, pad_type='zero', use_bias=True, sn=sn, scope='conv_128'), is_training, scope='conv_128_batch_norm'))
            print("z shape: {}".format(z.shape))
            z = relu(batch_norm(conv(z, channels=64, kernel=3, stride=2, pad=1, pad_type='zero', use_bias=True, sn=sn, scope='conv_0a'), is_training, scope='conv_0a_batch_norm'))
            print("z shape: {}".format(z.shape))
            z = relu(batch_norm(conv(z, channels=128, kernel=3, stride=2, pad=1, pad_type='zero', use_bias=True, sn=sn, scope='conv_1a'), is_training, scope='conv_1a_batch_norm'))
            print("z shape: {}".format(z.shape))
            z = relu(batch_norm(conv(z, channels=256, kernel=3, stride=2, pad=1, pad_type='zero', use_bias=True, sn=sn, scope='conv_2a'), is_training, scope='conv_2a_batch_norm'))
            print("z shape: {}".format(z.shape))
            zfinal = relu(batch_norm(conv(z, channels=512, kernel=3, stride=2, pad=1, pad_type='zero', use_bias=True, sn=sn, scope='conv_3a'), is_training, scope='conv_3a_batch_norm'))
            print("zfinal shape: {}".format(zfinal.shape))
            #z = fully_conneted(zfinal, z_dim, use_bias=True, sn=sn, scope='fully_0a')
            #z = tf.concat([z, style_vec], axis=-1)
            z = fully_conneted(zfinal, z_dim, use_bias=True, sn=sn, scope='fully_1a')
            z = tf.expand_dims(z, axis=1)
            z = tf.expand_dims(z, axis=2)
            if z_dim == 128 :
                split_dim = 16
                split_dim_remainder = z_dim - (split_dim * 7)

                z_split = tf.split(z, num_or_size_splits=[split_dim] * 7 + [split_dim_remainder], axis=-1)

            else :
                split_dim = z_dim // 8
                split_dim_remainder = z_dim - (split_dim * 8)

                if split_dim_remainder == 0 :
                    z_split = tf.split(z, num_or_size_splits=[split_dim] * 8, axis=-1)
                else :
                    z_split = tf.split(z, num_or_size_splits=[split_dim] * 7 + [split_dim_remainder], axis=-1)


            ch = 16 * ch
            #x = fully_conneted(z_split[0], units=4 * 4 * ch, sn=sn, scope='dense')
            x = fully_conneted(z, units=4*4*ch, sn=sn, scope='dense')
            x = tf.reshape(x, shape=[-1, 4, 4, ch])
            print("Layer 1: {}".format(x.shape))

            #x = resblock_up_condition(x, z_split[1], channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_up_16')
            x = resblock_up_condition(x, style, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_up_16')
            x = tf.concat([x, zfinal], axis=-1)
            ch = ch // 2
            print("Layer 2: {}".format(x.shape))

            x = resblock_up_condition(x, style, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_up_8_0')
            print("Layer 3: {}".format(x.shape))
            x = resblock_up_condition(x, style, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_up_8_1')
            print("Layer 4: {}".format(x.shape))
            ch = ch // 2

            x = resblock_up_condition(x, style, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_up_4')
            print("Layer 5: {}".format(x.shape))
            # Non-Local Block
            x = self_attention_2(x, channels=ch, sn=sn, scope='self_attention')
            print("Layer 6: {}".format(x.shape))
            ch = ch // 2

            x = resblock_up_condition(x, style, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_up_2')
            print("Layer 7: {}".format(x.shape))
            ch = ch // 2

            x = resblock_up_condition(x, style, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_up_1_0')
            print("Layer 8: {}".format(x.shape))
            x = resblock_up_condition(x, style, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_up_1_1')
            print("Layer 9: {}".format(x.shape))

            x = batch_norm(x, is_training)
            x = relu(x)
            x = conv(x, channels=c_dim, kernel=3, stride=1, pad=1, use_bias=False, sn=sn, scope='G_logit')
            print("Layer 10: {}".format(x.shape))

            x = tanh(x)

            return x

##################################################################################
# Discriminator
##################################################################################
def discriminator(x, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            ch = 48
            sn = True
            c_dim = 1

            x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_down_1_0')
            x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_down_1_1')
            ch = ch * 2

            x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_down_2')

            # Non-Local Block
            x = self_attention_2(x, channels=ch, sn=sn, scope='self_attention')
            ch = ch * 2

            x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_down_4')
            ch = ch * 2

            x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_down_8_0')
            x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_down_8_1')
            ch = ch * 2

            x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock_down_16')

            x = resblock(x, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope='resblock')
            x = relu(x)

            x = global_sum_pooling(x)

            x = fully_conneted(x, units=1, sn=sn, scope='D_logit')

            return x
