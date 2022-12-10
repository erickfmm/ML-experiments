from load_data.ILoadUnsupervised import ILoadUnsupervised
from os import listdir
from os.path import join, splitext
from PIL import Image

__all__ = ["LoadAnimeFaces"]


class LoadAnimeFaces(ILoadUnsupervised):
    def __init__(self, folder_path="train_data/Folder_Manga_Anime/anime-faces"):
        self.folder_path = folder_path

    def get_headers(self):
        return ["Image"]

    def get_all(self):
        all_data = []
        for data in self.get_all_yield():
            all_data.append(data)
        return all_data

    def get_all_yield(self):
        for filename in listdir(self.folder_path):
            if splitext(filename)[1].lower() == ".png":
                fullname = join(self.folder_path, filename)
                im = Image.open(fullname)  # TODO: 64x64x4 (to see if alpha channel is real)
                yield im
