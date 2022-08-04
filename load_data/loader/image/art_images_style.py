from load_data.ILoadSupervised import ILoadSupervised
from PIL import Image
from os.path import join, splitext
from os import listdir

__all__ = ["LoadArtImageStyle",]

class LoadArtImageStyle(ILoadSupervised):
    def __init__(self, folderpath="train_data/Folder_Images_Supervised/art-images-drawings-painting-sculpture-engraving",\
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
        XTrain = []
        XTest = []
        YTrain = []
        YTest = []
        train_folder = join(self.folderpath, self.datasetpath, "training_set")
        validation_folder = join(self.folderpath, self.datasetpath, "validation_set")
        for cl in self.classes:
            for filename in listdir(join(train_folder, cl)):
                if splitext(filename)[1].lower() == ".jpg":
                    fullname = join(train_folder, cl, filename)
                    try:
                        im = Image.open(fullname)
                        XTrain.append(im)
                        YTrain.append(cl)
                    except:
                        print("Error in: ", fullname)
            for filename in listdir(join(validation_folder, cl)):
                if splitext(filename)[1].lower() == ".jpg":
                    fullname = join(validation_folder, cl, filename)
                    try:
                        im = Image.open(fullname)
                        XTest.append(im)
                        YTest.append(cl)
                    except:
                        print("Error in: ", fullname)
        return (XTrain, YTrain), (XTest, YTest)
    
    def get_all(self):
        X = []
        Y = []
        train_folder = join(self.folderpath, self.datasetpath, "training_set")
        validation_folder = join(self.folderpath, self.datasetpath, "validation_set")
        for cl in self.classes:
            for filename in listdir(join(train_folder, cl)):
                if splitext(filename)[1].lower() == ".jpg":
                    fullname = join(train_folder, cl, filename)
                    try:
                        im = Image.open(fullname)
                        X.append(im)
                        Y.append(cl)
                    except:
                        print("Error in: ", fullname)
            for filename in listdir(join(validation_folder, cl)):
                if splitext(filename)[1].lower() == ".jpg":
                    fullname = join(validation_folder, cl, filename)
                    try:
                        im = Image.open(fullname)
                        X.append(im)
                        Y.append(cl)
                    except:
                        print("Error in: ", fullname)
        return X, Y
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return ["image"]# None #self.headers


