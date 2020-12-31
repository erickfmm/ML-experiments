# -*- coding: utf-8 -*-

from load_data.ILoadUnsupervised import ILoadUnsupervised
from os.path import join, splitext
from os import listdir
import pims

class LoadTwitchEmotes(ILoadUnsupervised):

    def __init__(self, \
    datapath="train_data/Folder_Images_Unsupervised/Emotes/Emotes"):
        self.datapath = datapath
        
    def get_headers(self):
        return ["image"]

    def get_all_yielded(self, image_shape=(28,28,4)):
        for filename in listdir(self.datapath):
            if splitext(filename)[1].lower() == ".jpg":
                im = pims.ImageReader(join(self.datapath, filename))
                frame = im.get_frame(0) #28x28x4
                if image_shape is None:
                    yield frame
                elif frame.shape == image_shape:
                    yield frame

    def get_all(self):
        data = []
        for im in self.get_all_yielded():
            data.append(im)
        return data

