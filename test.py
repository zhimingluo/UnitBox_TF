import tensorflow as tf
import tensorlayer as tl
from UnitBox import Model, loss_function
import numpy as np
import cv2


if __name__ == "__main__":

    im = cv2.imread("22_Picnic_Picnic_22_2.jpg")

    x = tf.placeholder(tf.float32, [None, None, None, 3])
    sc_ = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='sc_')
    net_in = tl.layers.InputLayer(x, name='input_layer')

    model = Model(net_in)

    sess = tf.InteractiveSession()

    tl.layers.initialize_global_variables(sess)

    ckpt = tf.train.get_checkpoint_state('model')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

    im = np.expand_dims(im, axis=0)

    score = sess.run(model['prob'], feed_dict={x: im})
    print(score.shape)

    cv2.imshow("img", np.squeeze(im))
    cv2.imshow("prob", np.squeeze(score))
    cv2.waitKey()

    sess.close()
