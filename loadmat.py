import scipy.io
import numpy as np
import pickle
import cv2
import os


def get_train_list():
    train_file = open("train.pkl", 'r')

    train_list = pickle.load(train_file)

    train_dic = {}

    for img in train_list:
        img_name = img.keys()[0]
        bbox = img[img_name]

        train_dic[img_name] = bbox

    return train_dic
