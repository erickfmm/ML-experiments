import sqlite3
from enum import Enum
from load_data.ILoadSupervised import ILoadSupervised
import os

#this code: import load_data.data_inside.not_shared.emotion_eeg.extracted_sqlite as extr

__all__ = ["LoadEEGEmotionExtracted", "EEGEmotionSegmentions", "EEGEmotionDataset"]

class EEGEmotionSegmentions(Enum):
    Paper = 1
    Yo = 2
    Otro5Classes = 3

class EEGEmotionDataset(Enum):
    Deap = 1
    Enterface = 2
    Mahnob = 3

class LoadEEGEmotionExtracted(ILoadSupervised):
    def __init__(self, segtype=EEGEmotionSegmentions.Paper, dataset=EEGEmotionDataset.Deap):
        self.fs = -1
        self.segmentation_type = segtype if isinstance(segtype, EEGEmotionSegmentions) else EEGEmotionSegmentions.Paper
        self.dataset_type = dataset if isinstance(dataset, EEGEmotionDataset) else EEGEmotionDataset.Deap
        self.basepath = "train_data/not_shared/Folder_EEGEmotion/datasetseegemo (extraido)/"
        self.specific_path = ""
        if self.segmentation_type == EEGEmotionSegmentions.Paper:
            if self.dataset_type == EEGEmotionDataset.Deap:
                self.specific_path = "SegPaper-DEAP"
            elif self.dataset_type == EEGEmotionDataset.Enterface:
                self.specific_path = "SegPaper-ENTERFACE"
            elif self.dataset_type == EEGEmotionDataset.Mahnob:
                self.specific_path = "SegPaperMod-MAHNOB"
        elif self.segmentation_type == EEGEmotionSegmentions.Yo:
            if self.dataset_type == EEGEmotionDataset.Deap:
                self.specific_path = "SegYo-DEAP"
            elif self.dataset_type == EEGEmotionDataset.Enterface:
                self.specific_path = "SegYo-ENTERFACE"
            elif self.dataset_type == EEGEmotionDataset.Mahnob:
                self.specific_path = "SegYo-MAHNOB"
        elif self.segmentation_type == EEGEmotionSegmentions.Otro5Classes:
            if self.dataset_type == EEGEmotionDataset.Deap:
                self.specific_path = "SegOtro5Cl-DEAP"
            elif self.dataset_type == EEGEmotionDataset.Enterface:
                self.specific_path = "SegOtro5Cl-ENTERFACE"
            elif self.dataset_type == EEGEmotionDataset.Mahnob:
                self.specific_path = "SegOtro5Cl-MAHNOB"
        self.path = os.path.join(self.basepath, self.specific_path)
        self.classes = [tag.replace(".db", "") for tag in os.listdir(self.path)]
        self.headers = ["F3", "C4"]


    def get_default(self):
        return self.get_all()

    def get_splited(self):
        return None
    
    def get_all(self):
        self.X = []
        #self.Y = []
        self.YSession = []
        isession = -1
        max_fs = 0
        for filename in os.listdir(self.path):
            conn = sqlite3.connect(os.path.join(self.path, filename))
            last_session = None
            for row in conn.execute("SELECT * from data"):
                if row[5] != last_session:
                    isession += 1
                    self.X.append([ [], [] ])
                    #self.Y.append([])
                    self.YSession.append(filename.replace(".db", ""))
                if row[2] > max_fs: #row[2] it the iteration in sec
                    max_fs = row[2]
                elif 9*(max_fs+1) == len(self.X[isession][0]):
                    continue
                #if row[1] <= 8:
                #self.X[isession].append([row[3], row[4]])
                self.X[isession][0].append(row[3])
                self.X[isession][1].append(row[4])
                #self.Y[isession].append(filename)
                last_session = row[5]
        self.fs = max_fs+1
        return (self.X, self.YSession)
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers
