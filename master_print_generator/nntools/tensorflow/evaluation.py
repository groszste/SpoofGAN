import tensorflow as tf

def rank_accuracy(logits, label, k=1):
	batch_size = tf.shape(logits)[0]
    _, arg_top = tf.nn.top_k(logits, k)
    label = tf.cast(label, tf.int32)
    label = tf.reshape(label, [batch_size, 1])
    label = tf.tile(label, [1, k])
    correct = tf.reduce_any(tf.equal(label, arg_top), axis=1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy