from load_data.ILoadSupervised import ILoadSupervised
from load_data.data_inside.not_shared.util_emotions import DiscreteEmotion
import os

#this code: import load_data.data_inside.not_shared.emotion_eeg.individual_clasiff as indiv

__all__ = ["LoadEEGIndividualEmotions",]

class LoadEEGIndividualEmotions(ILoadSupervised):
    def __init__(self):
        self.fs = 1024
        self._path = "train_data/not_shared/Folder_EEGEmotion/Dataset Individual Classification of Emotions Using EEG/"
        self.tags = {
            '01': DiscreteEmotion.Sad.name,
            '05': DiscreteEmotion.Disgust.name,
            '09': DiscreteEmotion.Neutral.name,
            '13': DiscreteEmotion.Amusement.name}
        self.classes = [self.tags[key] for key in self.tags]

    def get_default(self):
        return self.get_all()

    def get_splited(self):
        return None
    
    def get_all(self):
        self.X = []
        self.Y = []
        for filename in os.listdir(self._path):
            tag = self.tags[filename[2:4]]
            file_obj = open(os.path.join(self._path, filename), "r")
            data_of_file = []
            for line in file_obj:
                data_line = line.split("\t")
                data_line = [float(element) for element in data_line]
                data_of_file.append(data_line)
            self.X.append(data_of_file)
            self.Y.append(tag)
        return (self.X, self.Y)
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return None #self.headers
