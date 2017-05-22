import tensorflow as tf
import tensorlayer as tl
from VGG16 import vgg_16, vgg16_init_weights
import numpy as np
import cv2
from IOULoss import IOULoss
from loadmat import get_train_list
import os

_EPSILON = 10e-8


def Model(net_in):
    in_shape = tf.shape(net_in.outputs)

    vgg16 = vgg_16(net_in, include_top=False)

    pool4 = tl.layers.get_layers_with_name(vgg16, 'pool4')[0]
    pool4 = tl.layers.InputLayer(pool4, name='pool4_1')

    pool5 = tl.layers.get_layers_with_name(vgg16, 'pool5')[0]
    pool5 = tl.layers.InputLayer(pool5, name='pool5_1')

    # score
    score_conv = tl.layers.Conv2dLayer(pool4,
                                       shape=[3, 3, 512, 1],
                                       act=tl.act.identity,
                                       padding='SAME',
                                       name='score_conv')
    score = tl.layers.DeConv2dLayer(score_conv,
                                    shape=[32, 32, 1, 1],
                                    strides=[1, 16, 16, 1],
                                    output_shape=[in_shape[0], in_shape[1], in_shape[2], 1],
                                    act=tl.act.identity,
                                    padding='SAME',
                                    W_init=tl.layers.deconv2d_bilinear_upsampling_initializer([32, 32, 1, 1]),
                                    b_init=None,
                                    name='score')

    prob = tf.nn.sigmoid(score.outputs)

    # bounding box
    bbox_conv = tl.layers.Conv2dLayer(pool5,
                                      shape=[3, 3, 512, 4],
                                      act=tl.act.identity,
                                      padding='SAME',
                                      name='bbox_conv')
    bbox = tl.layers.DeConv2dLayer(bbox_conv,
                                   shape=[64, 64, 4, 4],
                                   strides=[1, 32, 32, 1],
                                   output_shape=[in_shape[0], in_shape[1], in_shape[2], 4],
                                   act=tf.nn.relu,
                                   padding='SAME',
                                   W_init=tl.layers.deconv2d_bilinear_upsampling_initializer([64, 64, 4, 4]),
                                   b_init=None,
                                   name='bbox')

    model = {}
    model['score'] = score.outputs
    model['prob'] = prob
    model['bbox'] = bbox.outputs

    return model


def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x, dtype)
    return x


def loss_function(sc_pred, sc_true, bbox_pred, bbox_true):

    score_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=sc_pred,
                                                labels=sc_true))

    bbox_loss = IOULoss(bbox_pred, bbox_true)

    l2 = 0.

    for w in tl.layers.get_variables_with_name('W_conv2d', train_only=True, printable=False):
        l2 += tf.contrib.layers.l2_regularizer(0.0005)(w)

    for w in tl.layers.get_variables_with_name('W_deconv2d', train_only=True, printable=False):
        l2 += tf.contrib.layers.l2_regularizer(0.0005)(w)

    return 0.01*score_loss + bbox_loss + l2


if __name__ == "__main__":

    x = tf.placeholder(tf.float32, [None, None, None, 3])
    sc_ = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='sc_')
    bbox_ = tf.placeholder(tf.float32, shape=[None, None, None, 4], name='bbox_')

    net_in = tl.layers.InputLayer(x, name='input_layer')

    model = Model(net_in)

    sess = tf.InteractiveSession()
    tl.layers.print_all_variables()

    loss = loss_function(model['score'], sc_, model['bbox'], bbox_)

    #train_op = tf.train.MomentumOptimizer(learning_rate=1e-7,
    #                                      momentum=0.9).minimize(loss)

    train_op = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(loss)

    tl.layers.initialize_global_variables(sess)
    vgg16_init_weights(sess, include_top=False)

    train_list = get_train_list()
    print(len(train_list.keys()))

    image_path = '../WiderFace'

    saver = tf.train.Saver()

    for epoch in range(20):
        total_loss = 0.
        count = 0
        for img_name, bboxes in train_list.items():
            path = os.path.join(image_path, img_name)
            im = cv2.imread(path)

            ratio = 1024. / np.maximum(im.shape[0], im.shape[1])
            nw = int(im.shape[1] * ratio)
            nh = int(im.shape[0] * ratio)

            im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_NEAREST)
            l_score = np.zeros((nh, nw), np.float32)
            l_bbox = np.zeros((nh, nw, 4), np.float32)

            for bbox in bboxes:
                bbox = np.array(bbox) * ratio

                x1, y1, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                x1 = np.maximum(x1, 0)
                y1 = np.maximum(y1, 0)
                x2 = np.minimum(x1+w, nw)
                y2 = np.minimum(y1+h, nh)

                l_score[y1:y2, x1:x2] = 1.

                for yy in range(y1, y2):
                    l_bbox[yy, x1:x2, 0] = yy - y1
                    l_bbox[yy, x1:x2, 1] = y2 - yy

                for xx in range(x1, x2):
                    l_bbox[y1:y2, xx, 2] = xx - x1
                    l_bbox[y1:y2, xx, 3] = x2 - xx

            im = np.expand_dims(im, axis=0)
            l_score = np.expand_dims(l_score, axis=0)
            l_score = np.expand_dims(l_score, axis=3)
            l_bbox = np.expand_dims(l_bbox, axis=0)

            l_bbox /= 64.

            _, loss_val = sess.run([train_op, loss], feed_dict={x: im,
                                                                sc_: l_score,
                                                                bbox_: l_bbox})

            # tl.files.save_npz(model['bbox'].all_params, name='model.npz')

            total_loss += loss_val
            count += 1

            if count % 20 == 0:
                print("Epoch [%d] average Loss [%d] : %f" % (epoch, count, total_loss / count))

        saver.save(sess, 'model/model_f.ckpt', global_step=epoch)

    sess.close()
