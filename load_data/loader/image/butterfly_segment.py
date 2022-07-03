# -*- coding: utf-8 -*-

from load_data.ILoadSupervised import ILoadSupervised
from os.path import join, splitext
from os import listdir
from PIL import Image

class LoadButterflySegment(ILoadSupervised):
    def __init__(self, \
    datapath="train_data/Folder_Images_Supervised/leedsbutterfly"):
        self.datapath = datapath
        pass

    def get_headers(self):
        return ["image"]
    
    def get_classes(self):
        return ["segment"]

    def get_all_yielded(self):
        """Set docstring here.

        Parameters
        ----------
        self: this object

        Returns yielded
        -------
        A tuple with:
            im:Image
            seg:Image
            butterfly_type:int 
        """
        for filename in listdir(join(self.datapath, "images")):
            if splitext(filename)[1].lower() == ".png":
                im = Image.open(join(self.datapath, "images", filename))
                seg = Image.open(join(self.datapath, "segmentations", \
                     splitext(filename)[0]+"_seg0"+splitext(filename)[1]))
                butterfly_type = int(filename[0:3])
                yield im, seg, butterfly_type


    def get_all(self):
        data = []
        for im, seg, btype in self.get_all_yielded():
            data.append((im, seg, btype))
        return data