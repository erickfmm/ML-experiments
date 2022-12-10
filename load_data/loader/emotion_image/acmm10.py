from load_data.ILoadSupervised import ILoadSupervised
from load_data.loader.util_emotions import DiscreteEmotion
from PIL import Image
from os.path import join, splitext
from os import listdir
import csv

__all__ = ["LoadACMM10ImageEmotion"]


class LoadACMM10ImageEmotion(ILoadSupervised):
    def __init__(self, folder_path="train_data/Folder_ImageEmotion/"
                                   "Affective Image Classification Using Features inspired by Psychology and Art Theory",
                 dataset="art", use_probability=True):
        """
        testImages_artphoto
        testImages_abstract
        """
        self.folder_path = folder_path
        self.dataset = "testImages_abstract" if dataset == "abstract" else "testImages_artphoto"
        self.use_probability = use_probability
        # cl = "'Amusement','Anger','Awe','Content','Disgust','Excitement','Fear','Sad'"
        self.classes = []
        if self.dataset == "testImages_abstract":
            self.classes = [
                DiscreteEmotion.Amusement,
                DiscreteEmotion.Angry,
                DiscreteEmotion.Surprise,  # Awe (?)
                DiscreteEmotion.Happy,  # Content(?)
                DiscreteEmotion.Disgust,
                DiscreteEmotion.Amusement,  # Excitement(?)
                DiscreteEmotion.Fear,
                DiscreteEmotion.Sad
            ]

    def get_default(self):
        return self.get_all()

    @staticmethod
    def get_splited():
        return None
    
    def get_all(self):
        if self.dataset == "testImages_abstract":
            return self.read_abstract()
        else:  # art
            return self.read_art()
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return ["image"]  # self.headers
    
    def read_art(self):
        xs = []
        ys = []
        self.classes = []
        for filename in listdir(join(self.folder_path, self.dataset)):
            if splitext(filename)[1].lower() == ".jpg":
                segments_name = filename.split("_")
                fullname = join(self.folder_path, self.dataset, filename)
                im = Image.open(fullname)
                xs.append(im)
                ys.append(segments_name[0])
                if segments_name[0] not in self.classes:
                    self.classes.append(segments_name[0])
        return xs, ys

    def read_abstract(self):
        truth_file_obj = open(join(self.folder_path, self.dataset, "ABSTRACT_groundTruth.csv"))
        truth_csv_reader = csv.reader(truth_file_obj, quotechar="'")
        first_row = True
        xs = []
        ys = []
        for row in truth_csv_reader:
            if first_row:
                first_row = False
                continue
            filename = row[0]
            fullname = join(self.folder_path, self.dataset, filename)
            values = [int(e) for e in row[1:]]
            s = sum(values)
            if self.use_probability:
                values = [e/float(s) for e in values]
            im = Image.open(fullname)
            xs.append(im)
            ys.append(values)
        return xs, ys
