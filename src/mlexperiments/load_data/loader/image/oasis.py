from mlexperiments.load_data.ILoadSupervised import ILoadSupervised, SupervisedType
from PIL import Image
from os.path import join
import csv

__all__ = ["LoadOASISImageEmotion",]


class LoadOASISImageEmotion(ILoadSupervised):
    def __init__(self, folder_path="data/train_data/Emotions_Images/OASIS"):
        self.TYPE = SupervisedType.Regression
        self.folder_path = folder_path
        self.Metadata = []
        self.MetadataHeaders = ["Category", "Source", "Valence_SD", "Valence_N", "Arousal_SD", "Arousal_N"]
    
    def get_X_Y(self):
        return self.read_files()
    
    def get_classes(self):
        return ["Valence mean", "Arousal mean"]  # self.classes
    
    def get_headers(self):
        return ["image"]  # self.headers

    def read_files(self):
        xs = []
        ys = []
        csv_path = join(self.folder_path, "oasis.csv")
        with open(csv_path, "r") as csv_obj:
            csv_reader = csv.DictReader(csv_obj)
            for row in csv_reader:
                image_path = join(self.folder_path, "images", row["Theme"].strip()+".jpg")
                im: Image = Image.open(image_path)
                xs.append(im)  # 500x400
                ys.append([float(row["Valence_mean"]), float(row["Arousal_mean"])])
                self.Metadata.append([
                    row["Category"],
                    row["Source"],
                    float(row["Valence_SD"]),
                    int(row["Valence_N"]),
                    float(row["Arousal_SD"]),
                    int(row["Arousal_N"])
                ])
        return xs, ys
