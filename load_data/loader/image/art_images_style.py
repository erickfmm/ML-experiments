from load_data.ILoadSupervised import ILoadSupervised
from PIL import Image
from os.path import join, splitext
from os import listdir

__all__ = ["LoadArtImageStyle"]


class LoadArtImageStyle(ILoadSupervised):
    def __init__(self,
                 folder_path="train_data/Folder_Images_Supervised/art-images-drawings-painting-sculpture-engraving",
                 dataset="dataset"):
        """
        dataset
        musemart
        """
        self.folder_path = folder_path
        self.dataset_path = "musemart" if dataset == "musemart" else "dataset"
        self.classes = [
            "drawings",
            "engraving",
            "iconography",
            "painting",
            "sculpture"
        ]

    def get_default(self):
        return self.get_all()

    def get_splited(self):
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        train_folder = join(self.folder_path, self.dataset_path, "training_set")
        validation_folder = join(self.folder_path, self.dataset_path, "validation_set")
        for cl in self.classes:
            for filename in listdir(join(train_folder, cl)):
                if splitext(filename)[1].lower() == ".jpg":
                    fullname = join(train_folder, cl, filename)
                    try:
                        im = Image.open(fullname)
                        x_train.append(im)
                        y_train.append(cl)
                    except Exception as e:
                        print("Error in: ", fullname, " error: ", e)
            for filename in listdir(join(validation_folder, cl)):
                if splitext(filename)[1].lower() == ".jpg":
                    fullname = join(validation_folder, cl, filename)
                    try:
                        im = Image.open(fullname)
                        x_test.append(im)
                        y_test.append(cl)
                    except Exception as e:
                        print("Error in: ", fullname, " error: ", e)
        return (x_train, y_train), (x_test, y_test)
    
    def get_all(self):
        xs = []
        ys = []
        train_folder = join(self.folder_path, self.dataset_path, "training_set")
        validation_folder = join(self.folder_path, self.dataset_path, "validation_set")
        for cl in self.classes:
            for filename in listdir(join(train_folder, cl)):
                if splitext(filename)[1].lower() == ".jpg":
                    fullname = join(train_folder, cl, filename)
                    try:
                        im = Image.open(fullname)
                        xs.append(im)
                        ys.append(cl)
                    except Exception as e:
                        print("Error in: ", fullname, " error: ", e)
            for filename in listdir(join(validation_folder, cl)):
                if splitext(filename)[1].lower() == ".jpg":
                    fullname = join(validation_folder, cl, filename)
                    try:
                        im = Image.open(fullname)
                        xs.append(im)
                        ys.append(cl)
                    except Exception as e:
                        print("Error in: ", fullname, " error: ", e)
        return xs, ys
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return ["image"]
