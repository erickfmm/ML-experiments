from load_data.ILoadUnsupervised import ILoadUnsupervised
from os import listdir
from os.path import join, splitext
from pims import ImageReader

class LoadAnimeFaces(ILoadUnsupervised):
    def __init__(self, folderpath="train_data\\not_shared\\Folder_Manga_Anime\\anime-faces"):
        self.folderpath = folderpath

    def get_headers(self):
        return ["Image"]

    def get_all(self):
        all_data = []
        for data in self.get_all_yield():
            all_data.append(data)
        return all_data

    def get_all_yield(self):
        for filename in listdir(self.folderpath):
            if splitext(filename)[1].lower() == ".png":
                fullname = join(self.folderpath, filename)
                data = ImageReader(fullname)
                yield data.get_frame(0)