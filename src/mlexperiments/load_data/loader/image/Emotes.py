from mlexperiments.load_data.ILoadUnsupervised import ILoadUnsupervised
from os.path import join, splitext
from os import listdir
from PIL import Image
import numpy as np
import opendatasets as od

__all__ = ["LoadTwitchEmotes"]


class LoadTwitchEmotes(ILoadUnsupervised):

    def __init__(self,
                 folder_path="data/train_data/Images_Unsupervised/twitch-emotes-images-dataset/Emotes"):
        self.folder_path = folder_path
        
    def get_headers(self):
        return ["image"]

    def get_data_yielded(self):  # image_shape=(28,28,4)
        list_files = listdir(self.folder_path)
        print("Cantidad de archivos: ",len(list_files))
        i_file = 0
        i_files_yielded = 0
        for filename in list_files:
            if i_file % 10000 == 0:
                print(i_file, ", ", i_files_yielded)
            i_file += 1
            if splitext(filename)[1].lower() == ".png":
                try:
                    im = Image.open(join(self.folder_path, filename))
                    im.load()
                    im = np.array(im)
                except:
                    continue
                if im.shape == (28,28, 4):
                    i_files_yielded += 1
                    yield im

    def get_data(self):
        data = []
        for im in self.get_data_yielded():
            data.append(im)
        return data

    def download(self):
        od.download("https://www.kaggle.com/datasets/quantum360/twitch-emotes-images-dataset", "data/train_data/Images_Unsupervised")