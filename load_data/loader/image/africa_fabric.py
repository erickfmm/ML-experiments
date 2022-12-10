from load_data.ILoadUnsupervised import ILoadUnsupervised
from os.path import join, splitext
from os import listdir
from PIL import Image

__all__ = ["LoadAfricaFabric"]


class LoadAfricaFabric(ILoadUnsupervised):

    def __init__(self,
                 folder_path="train_data/Folder_Images_Unsupervised/African Fabric Images/africa_fabric"):
        self.folder_path = folder_path
        
    def get_headers(self):
        return ["image"]

    def get_all_yielded(self):
        for filename in listdir(self.folder_path):
            if splitext(filename)[1].lower() == ".jpg":
                im = Image.open(join(self.folder_path, filename))
                yield im  # 64x64x3

    def get_all(self):
        data = []
        for im in self.get_all_yielded():
            data.append(im)
        return data
