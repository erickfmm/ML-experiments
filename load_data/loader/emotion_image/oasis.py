#train_data\\Folder_ImageEmotion\\oasis\images
#train_data\\Folder_ImageEmotion\\oasis\\OASIS_bygender_CORRECTED_092617.csv
#train_data\\Folder_ImageEmotion\\oasis\\OASIS.csv

from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import pims
from os.path import join, splitext
from os import listdir
import csv

__all__ = ["LoadOASISImageEmotion",]

class LoadOASISImageEmotion(ILoadSupervised):
    def __init__(self, folderpath="train_data/Folder_ImageEmotion/oasis"):
        self.folderpath = folderpath
        self.TYPE = SupervisedType.Regression

    def get_default(self):
        return None

    def get_splited(self):
        return None
    
    def get_all(self):
        return self.read_files()
    
    def get_classes(self):
        return ["Valence mean", "Arousal mean"] #self.classes
    
    def get_headers(self):
        return ["image"] #self.headers

    def read_files(self):
        self.X = []
        self.Y = []
        self.Metadata = []
        self.MetadataHeaders = ["Category", "Source", "Valence_SD", "Valence_N", "Arousal_SD", "Arousal_N"]
        csv_path = join(self.folderpath, "oasis.csv")
        with open(csv_path, "r") as csv_obj:
            csv_reader = csv.DictReader(csv_obj)
            for row in csv_reader:
                image_path = join(self.folderpath, "images", row["Theme"].strip()+".jpg")
                im = pims.ImageReader(image_path)
                self.X.append(im.get_frame(0)) #500x400
                self.Y.append([float(row["Valence_mean"]), float(row["Arousal_mean"])])
                self.Metadata.append([
                    row["Category"],
                    row["Source"],
                    float(row["Valence_SD"]),
                    int(row["Valence_N"]),
                    float(row["Arousal_SD"]),
                    int(row["Arousal_N"])
                ])
        return self.X, self.Y
