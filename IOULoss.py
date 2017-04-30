import tensorflow as tf
import keras

_EPSILON = 10e-8


def IOULoss(input, label):
    """
    :param input: the estimate position
    :param label: the ground truth position
    :return: the IoU loss
    """
    # the estimate position
    xt, xb, xl, xr = tf.split(input, num_or_size_splits=4, axis=3)

    # the ground truth position
    gt, gb, gl, gr = tf.split(label, num_or_size_splits=4, axis=3)

    # compute the bounding box size
    X = (xt + xb) + (xl + xr)
    G = (gt + gb) + (gl + gr)

    # compute the IOU
    Ih = tf.minimum(xt, gt) + tf.minimum(xb, gb)
    Iw = tf.minimum(xl, gl) + tf.minimum(xr, gr)

    I = tf.multiply(Ih, Iw, name="intersection")
    U = X + G - I + _EPSILON

    IoU = tf.divide(I, U)
    L = -tf.log(IoU + _EPSILON)

    L = tf.where(tf.equal(gt, tf.constant(0.01, dtype=tf.float32)), tf.zeros_like(L, tf.float32), L)

    return tf.reduce_sum(L)
