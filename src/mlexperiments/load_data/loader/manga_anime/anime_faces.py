from mlexperiments.load_data.ILoadUnsupervised import ILoadUnsupervised
from os import listdir
from os.path import join, splitext
from PIL import Image
import opendatasets as od

__all__ = ["LoadAnimeFaces"]


class LoadAnimeFaces(ILoadUnsupervised):
    def __init__(self, folder_path="data/train_data/Manga_Anime/animefacedataset/images"):
        self.folder_path = folder_path

    def get_headers(self):
        return ["Image"]

    def get_data(self):
        all_data = []
        for data in self.get_data_yield():
            all_data.append(data)
        return all_data

    def get_data_yield(self):
        for filename in listdir(self.folder_path):
            if splitext(filename)[1].lower() == ".png":
                fullname = join(self.folder_path, filename)
                im = Image.open(fullname)  # TODO: 64x64x4 (to see if alpha channel is real)
                yield im

    def download(self):
        od.download("https://www.kaggle.com/datasets/splcher/animefacedataset", "data/train_data/Manga_Anime")
