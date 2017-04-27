from __future__ import absolute_import
import tensorflow as tf
import tensorlayer as tl
from scipy.misc import imread, imresize
import numpy as np
from imagenet_classes import *


def vgg_16(net_in, include_top=True):
    with tf.name_scope('preprocess') as scope:
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in.outputs = net_in.outputs - mean

    """ conv1 """
    network = tl.layers.Conv2dLayer(net_in,
                    act = tf.nn.relu,
                    shape = [3, 3, 3, 64],  # 64 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name='conv1_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 64, 64],  # 64 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name='conv1_2')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool=tf.nn.max_pool,
                    name='pool1')
    """ conv2 """
    network = tl.layers.Conv2dLayer(network,
                    act=tf.nn.relu,
                    shape=[3, 3, 64, 128],  # 128 features for each 3x3 patch
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name='conv2_1')
    network = tl.layers.Conv2dLayer(network,
                    act=tf.nn.relu,
                    shape=[3, 3, 128, 128],  # 128 features for each 3x3 patch
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name='conv2_2')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool=tf.nn.max_pool,
                    name='pool2')
    """ conv3 """
    network = tl.layers.Conv2dLayer(network,
                    act=tf.nn.relu,
                    shape=[3, 3, 128, 256],  # 256 features for each 3x3 patch
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name='conv3_1')
    network = tl.layers.Conv2dLayer(network,
                    act= tf.nn.relu,
                    shape=[3, 3, 256, 256],  # 256 features for each 3x3 patch
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name='conv3_2')
    network = tl.layers.Conv2dLayer(network,
                    act=tf.nn.relu,
                    shape=[3, 3, 256, 256],  # 256 features for each 3x3 patch
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name='conv3_3')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool=tf.nn.max_pool,
                    name='pool3')
    """ conv4 """
    network = tl.layers.Conv2dLayer(network,
                    act=tf.nn.relu,
                    shape=[3, 3, 256, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name='conv4_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4_2')
    network = tl.layers.Conv2dLayer(network,
                    act=tf.nn.relu,
                    shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name='conv4_3')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool=tf.nn.max_pool,
                    name='pool4')
    """ conv5 """
    network = tl.layers.Conv2dLayer(network,
                    act=tf.nn.relu,
                    shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name='conv5_1')
    network = tl.layers.Conv2dLayer(network,
                    act=tf.nn.relu,
                    shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name='conv5_2')
    network = tl.layers.Conv2dLayer(network,
                    act=tf.nn.relu,
                    shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name='conv5_3')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool=tf.nn.max_pool,
                    name='pool5')

    if include_top:
        network = tl.layers.FlattenLayer(network, name='flatten')
        network = tl.layers.DenseLayer(network, n_units=4096,
                                       act=tf.nn.relu,
                                       name='fc6')
        network = tl.layers.DenseLayer(network, n_units=4096,
                                       act=tf.nn.relu,
                                       name='fc7')
        network = tl.layers.DenseLayer(network, n_units=1000,
                                       act=tf.identity,
                                       name='fc8')
    return network


def vgg16_init_weights(sess, include_top=True):
    conv_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                   'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
    fc_layers = ['fc6', 'fc7', 'fc8']

    val_list = []
    for layer in conv_layers:
        vals = tl.layers.get_variables_with_name(layer)
        val_list += vals

    if include_top:
        for layer in fc_layers:
            vals = tl.layers.get_variables_with_name(layer)
            val_list += vals

    npz = np.load('vgg16_weights.npz')
    params = []
    for val in sorted(npz.items()):
        # print(" Loading %s, %s" % (val[0], str(val[1].shape)))
        params.append(val[1])

    for idx, var in enumerate(val_list):
        assign_placeholder = tf.placeholder(tf.float32, shape=params[idx].shape)
        assign_op = var.assign(assign_placeholder)
        sess.run(assign_op, feed_dict={assign_placeholder: params[idx]})


if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    net_in = tl.layers.InputLayer(x, name='input_layer')
    network = vgg_16(net_in, True)
    y = network.outputs
    probs = tf.nn.softmax(y)
    y_op = tf.argmax(tf.nn.softmax(y), 1)
    cost = tl.cost.cross_entropy(y, y_, name='cost')

    correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.float32), tf.cast(y_, tf.float32))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    pool3 = tl.layers.get_layers_with_name(network, 'pool3')
    print(type(pool3))

    sess = tf.InteractiveSession()

    vgg16_init_weights(sess)

    img = imread('laska.png', mode='RGB')
    img = imresize(img, (224, 224))

    prob = sess.run(probs, feed_dict={x: [img]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])

    print(tf.shape(net_in.outputs)[0])
    print(tf.shape(net_in.outputs)[1])
    print(tf.shape(net_in.outputs)[2])
    print(tf.shape(net_in.outputs)[3])

    sess.close()
