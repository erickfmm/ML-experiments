from load_data.ILoadSupervised import ILoadSupervised
from load_data.loader.util_emotions import DiscreteEmotion
import pickle
from os.path import join
from os import listdir

#from load_data.loader.gsr_mahnob_emotion import LoadGsrMahnobEmotion

__all__ = ["LoadGsrMahnobEmotion",]

class LoadGsrMahnobEmotion(ILoadSupervised):
    def __init__(self, data_folder="train_data/Folder_Biosignal/gsr_mahnob_data"):
        self.data_folder = data_folder
        self.classes = [
            DiscreteEmotion.Amusement.name,
            DiscreteEmotion.Angry.name,
            DiscreteEmotion.Anxiety.name,
            DiscreteEmotion.Disgust.name,
            DiscreteEmotion.Fear.name,
            DiscreteEmotion.Happy.name,
            DiscreteEmotion.Neutral.name,
            DiscreteEmotion.Sad.name,
            DiscreteEmotion.Surprise.name]
        self.headers = ["Amplitude"]
        self.folders_of_data = [
            'Amusement',
            'Anger',
            'Anxiety',
            'Disgust',
            'Fear',
            'Joy, Happiness',
            'Neutral',
            'Sadness',
            'Surprise']

    def get_default(self):
        return self.get_all()

    def get_splited(self):
        return None
    
    def get_all(self):
        self.X = []
        self.Y = []
        ifolder = 0
        for folder_name in self.folders_of_data:
            files = listdir(join(self.data_folder, folder_name))
            for filename in files:
                with open(join(self.data_folder, folder_name, filename), "rb") as fobj:
                    self.X.append(pickle.load(fobj))
                    self.Y.append(self.classes[ifolder])
            ifolder += 1
        return (self.X, self.Y)
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers
