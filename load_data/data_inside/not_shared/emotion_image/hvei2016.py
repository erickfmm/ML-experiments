#train_data\\not_shared\\Folder_ImageEmotion\\ImageDataset
#ground_truth_each_image.csv

from load_data.ILoadSupervised import ILoadSupervised
import pims
from os.path import join, splitext
from os import listdir
import csv

__all__ = ["LoadHVEI2016ImageEmotion",]

class LoadHVEI2016ImageEmotion(ILoadSupervised):
    def __init__(self, folderpath="train_data\\not_shared\\Folder_ImageEmotion\\ImageDataset",\
        target_type="discrete"): #"discrete" or "circumplex"
        self.folderpath = folderpath
        self.target_type = "circumplex" if target_type == "circumplex" else "discrete"

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
        if self.target_type == "circumplex":
            self.MetadataHeaders = [
                "Face",
                "Color",
                "Scene",
                "Object",
                "Text",
                "Emoji",
                "Halo",
                "joy",
                "sadness",
                "fear",
                "disgust",
                "anger",
                "surprise",
                "neutral"
            ]
        else: #discrete
            self.MetadataHeaders = [
                "Face",
                "Color",
                "Scene",
                "Object",
                "Text",
                "Emoji",
                "Halo",
                "v_score_mean",
                "a_score_mean"
            ]
        csv_path = join(self.folderpath, "ground_truth_each_image.csv")
        with open(csv_path, "r") as csv_obj:
            csv_reader = csv.DictReader(csv_obj)
            for row in csv_reader:
                image_path = join(self.folderpath, row["image"]+".jpg")
                im = pims.ImageReader(image_path)
                self.X.append(im.get_frame(0)) #640x640
                if self.target_type == "circumplex":
                    self.Y.append([float(row["v_score_mean"]), float(row["a_score_mean"])])
                    self.Metadata.append([
                        float(row["Face"]),
                        float(row["Color"]),
                        float(row["Scene"]),
                        float(row["Object"]),
                        float(row["Text"]),
                        float(row["Emoji"]),
                        float(row["Halo"]),
                        float(row["joy"]),
                        float(row["sadness"]),
                        float(row["fear"]),
                        float(row["disgust"]),
                        float(row["anger"]),
                        float(row["surprise"]),
                        float(row["neutral"])
                    ])
                else: #discrete
                    self.Y.append([
                        float(row["joy"]),
                        float(row["sadness"]),
                        float(row["fear"]),
                        float(row["disgust"]),
                        float(row["anger"]),
                        float(row["surprise"]),
                        float(row["neutral"])
                    ])
                    self.Metadata.append([
                        float(row["Face"]),
                        float(row["Color"]),
                        float(row["Scene"]),
                        float(row["Object"]),
                        float(row["Text"]),
                        float(row["Emoji"]),
                        float(row["Halo"]),
                        float(row["v_score_mean"]),
                        float(row["a_score_mean"])
                    ])
        return self.X, self.Y
