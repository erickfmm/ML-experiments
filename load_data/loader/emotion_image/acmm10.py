#train_data\Folder_ImageEmotion\Affective Image Classification Using Features inspired by Psychology and Art Theory\testImages_artphoto
#emo_numero

#train_data\Folder_ImageEmotion\Affective Image Classification Using Features inspired by Psychology and Art Theory\testImages_abstract
#ABSTRACT_groundTruth.csv
from load_data.ILoadSupervised import ILoadSupervised
from load_data.data_inside.not_shared.util_emotions import DiscreteEmotion
import pims
from os.path import join, splitext
from os import listdir
import csv

__all__ = ["LoadACMM10ImageEmotion",]

class LoadACMM10ImageEmotion(ILoadSupervised):
    def __init__(self, folderpath="train_data/Folder_ImageEmotion/Affective Image Classification Using Features inspired by Psychology and Art Theory"\
        , dataset="art",
        use_probability=True):
        """
        testImages_artphoto
        testImages_abstract
        """
        self.folderpath = folderpath
        self.dataset = "testImages_abstract" if dataset=="abstract" else "testImages_artphoto"
        self.use_probability = use_probability
        #cl = "'Amusement','Anger','Awe','Content','Disgust','Excitement','Fear','Sad'"
        self.classes = []
        if self.dataset == "testImages_abstract":
            self.classes = [
                DiscreteEmotion.Amusement,
                DiscreteEmotion.Angry,
                DiscreteEmotion.Surprise,#Awe (?)
                DiscreteEmotion.Happy,#Content(?)
                DiscreteEmotion.Disgust,
                DiscreteEmotion.Amusement,#Excitement(?)
                DiscreteEmotion.Fear,
                DiscreteEmotion.Sad
            ]
        #classes = [DiscreteEmotion.Amusement,DiscreteEmotion.Angry,DiscreteEmotion.Surprise,DiscreteEmotion.Happy,DiscreteEmotion.Disgust,DiscreteEmotion.Amusement,DiscreteEmotion.Fear,DiscreteEmotion.Sad]

    def get_default(self):
        return None

    def get_splited(self):
        return None
    
    def get_all(self):
        if self.dataset == "testImages_abstract":
            return self.read_abstract()
        else: #art
            return self.read_art()
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return ["image"] #self.headers
    
    def read_art(self):
        self.X = []
        self.Y = []
        self.classes = []
        for filename in listdir(join(self.folderpath, self.dataset)):
            if splitext(filename)[1].lower() == ".jpg":
                segments_name = filename.split("_")
                fullname = join(self.folderpath, self.dataset, filename)
                im = pims.ImageReader(fullname)
                self.X.append(im.get_frame(0))
                self.Y.append(segments_name[0])
                if segments_name[0] not in self.classes:
                    self.classes.append(segments_name[0])
        return self.X, self.Y

    def read_abstract(self):
        truth_file_obj = open(join(self.folderpath, self.dataset, "ABSTRACT_groundTruth.csv"))
        truth_csv_reader = csv.reader(truth_file_obj, quotechar="'")
        first_row = True
        self.X = []
        self.Y = []
        self.Metadata = []
        self.MetadataHeaders = ["Sum of values"]
        for row in truth_csv_reader:
            if first_row:
                first_row = False
                continue
            filename = row[0]
            fullname = join(self.folderpath, self.dataset, filename)
            values = [int(e) for e in row[1:]]
            s = sum(values)
            if self.use_probability:
                values = [e/float(s) for e in values]
            im = pims.ImageReader(fullname)
            self.X.append(im.get_frame(0))
            self.Y.append(values)
            self.Metadata.append(s)
        return self.X, self.Y
            

