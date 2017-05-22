import tensorflow as tf
import tensorlayer as tl
from UnitBox import Model
import numpy as np
import cv2


if __name__ == "__main__":

    im = cv2.imread("test_im/img_1996.jpg")

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

    score, bbox = sess.run([model['prob'], model['bbox']], feed_dict={x: im})

    cv2.imshow("img", np.squeeze(im))
    cv2.imshow("prob", np.squeeze(score))

    print(bbox.shape)
    print(np.min(bbox[:,:,0]), np.max(bbox[:,:,0]))

    score = np.squeeze(score)
    label = (score>0.9999).astype(np.float32)

    bbox = np.squeeze(bbox) * 64

    im = np.squeeze(im)

    for r in range(im.shape[0]):
        for c in range(im.shape[1]):
            if score[r,c] > 0.9:
                if np.random.rand(1) < 0.005:
                    cv2.rectangle(im, (int(c-bbox[r,c,2]), int(r-bbox[r,c,0])),
                                  (int(c+bbox[r,c,3]), int(r+bbox[r,c,1])), (0, 255, 0), thickness=2)

    cv2.imshow("img", im)


    cv2.waitKey()

    sess.close()
