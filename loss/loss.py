import tensorflow as tf


def sparse_cross_entropy(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true, logits=y_pred)
    loss_mean = tf.reduce_mean(loss)

    return loss_mean
