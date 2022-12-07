import sys
from os.path import dirname, join, abspath

from load_data.loader.image.butterfly_segment import LoadButterflySegment
import utils.image.resize_image as resize
import preprocessing.image2D.rgb_ypbpr as ypbpr
import preprocessing.image2D.convolution as conv
import numpy as np
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append(abspath(join(dirname(__file__), '..')))


def save_as_pickle(new_size:int = 100, to_gray: bool = True):
    lb = LoadButterflySegment()
    # new_size = 100
    x_gray = []
    x_seg = []
    y = []
    i = 0
    for im, seg, btype in lb.get_all_yielded():
        print(i)
        i = i+1
        im2 = resize.with_image_module(im, new_size)
        if to_gray:
            im2 = im2.convert("L")  # to gray
        im2 = np.asarray(im2)
        x_gray.append(im2)
        seg2 = resize.with_image_module(seg, new_size)
        if to_gray:
            seg2 = seg2.convert("L")
        x_seg.append(np.asarray(seg2))
        y.append(btype)
    with open("created_models/butterfly.pkl", "wb") as file_handle:
        print("writing pickle xgray, type")
        pickle.dump({"x":x_gray,"y":y}, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("created_models/butterfly_segment.pkl", "wb") as file_handle:
        print("writing pickle xgray, xseg")
        pickle.dump({"x":x_gray,"y":x_seg}, file_handle, protocol=pickle.HIGHEST_PROTOCOL)


def open_pickle(which_pkl:str) -> dict:
    """Set docstring here.

    Parameters
    ----------
    which_pkl:str: 'butterfly' to open the pickle with "y" = types of butterfly or
    'segment' to open the pickle file with "y" = the segmentation of file

    Returns
    -------
    Dict: A Dictionary with two keys: "x" (an array of images) 
    and "y" (an array of int if 'butterfly' or images if 'segment')
    """
    path_to_pickle = "created_models/butterfly_segment.pkl" \
        if which_pkl == "segment" else "created_models/butterfly.pkl"
    with open(path_to_pickle, "rb") as file_handler:
        return pickle.load(file_handler)


def simple_classifier(x, y):
    model = keras.Sequential()
    model.add(layers.Conv2D(32, 3, activation="relu"))


if __name__ == "__main__":
    print("running as main")
    import os
    print(os.getcwd())
    # save_as_pickle()
    x = []
    y = []
    with open_pickle("butterfly") as pklfile:
        x = pklfile["x"]
        y = pklfile["y"]