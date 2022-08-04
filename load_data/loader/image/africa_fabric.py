# -*- coding: utf-8 -*-

from load_data.ILoadUnsupervised import ILoadUnsupervised
from os.path import join, splitext
from os import listdir
from PIL import Image
#import pims

class LoadAfricaFabric(ILoadUnsupervised):

    def __init__(self, \
    datapath="train_data/Folder_Images_Unsupervised/African Fabric Images/africa_fabric"):
        self.datapath = datapath
        
    def get_headers(self):
        return ["image"]

    def get_all_yielded(self):
        for filename in listdir(self.datapath):
            if splitext(filename)[1].lower() == ".jpg":
                im = Image.open(join(self.datapath, filename))
                yield im #64x64x3

    def get_all(self):
        data = []
        for im in self.get_all_yielded():
            data.append(im)
        return data

