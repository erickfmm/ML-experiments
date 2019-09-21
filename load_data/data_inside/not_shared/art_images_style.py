from load_data.ILoadSupervised import ILoadSupervised
import pims
from os.path import join, splitext
from os import listdir

__all__ = ["LoadArtImageStyle",]

class LoadArtImageStyle(ILoadSupervised):
    def __init__(self, folderpath="train_data\\not_shared\\Folder_FromKaggle\\art-images-drawings-painting-sculpture-engraving",\
        dataset="dataset"):
        """
        dataset
        musemart
        """
        self.folderpath = folderpath
        self.datasetpath = "musemart" if dataset == "musemart" else "dataset"
        self.classes = [
            "drawings",
            "engraving",
            "iconography",
            "painting",
            "sculpture"
        ]

    def get_default(self):
        return None

    def get_splited(self):
        self.XTrain = []
        self.XTest = []
        self.YTrain = []
        self.YTest = []
        train_folder = join(self.folderpath, self.datasetpath, "training_set")
        validation_folder = join(self.folderpath, self.datasetpath, "validation_set")
        for cl in self.classes:
            for filename in listdir(join(train_folder, cl)):
                if splitext(filename)[1].lower() == ".jpg":
                    fullname = join(train_folder, cl, filename)
                    try:
                        im = pims.ImageReader(fullname)
                        self.XTrain.append(im.get_frame(0))
                        self.YTrain.append(cl)
                    except:
                        print("Error in: ", fullname)
            for filename in listdir(join(validation_folder, cl)):
                if splitext(filename)[1].lower() == ".jpg":
                    fullname = join(validation_folder, cl, filename)
                    try:
                        im = pims.ImageReader(fullname)
                        self.XTest.append(im.get_frame(0))
                        self.YTest.append(cl)
                    except:
                        print("Error in: ", fullname)
        return (self.XTrain, self.YTrain), (self.XTest, self.YTest)
    
    def get_all(self):
        self.X = []
        self.Y = []
        train_folder = join(self.folderpath, self.datasetpath, "training_set")
        validation_folder = join(self.folderpath, self.datasetpath, "validation_set")
        for cl in self.classes:
            for filename in listdir(join(train_folder, cl)):
                if splitext(filename)[1].lower() == ".jpg":
                    fullname = join(train_folder, cl, filename)
                    try:
                        im = pims.ImageReader(fullname)
                        self.X.append(im.get_frame(0))
                        self.Y.append(cl)
                    except:
                        print("Error in: ", fullname)
            for filename in listdir(join(validation_folder, cl)):
                if splitext(filename)[1].lower() == ".jpg":
                    fullname = join(validation_folder, cl, filename)
                    try:
                        im = pims.ImageReader(fullname)
                        self.X.append(im.get_frame(0))
                        self.Y.append(cl)
                    except:
                        print("Error in: ", fullname)
        return self.X, self.Y
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return ["image"]# None #self.headers


