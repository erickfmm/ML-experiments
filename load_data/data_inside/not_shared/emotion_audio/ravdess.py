import scipy.io.wavfile as wav
from load_data.ILoadSupervised import ILoadSupervised
from load_data.data_inside.not_shared.util_emotions import DiscreteEmotion
import os
from os.path import join, splitext

__all__ = ["LoadRavdess",]

class LoadRavdess(ILoadSupervised):
    def __init__(self, modalities=["speech", "song"], folderpath="train_data\\not_shared\\Folder_AudioEmotion\\RAVDESS"):
        self.folderpath = folderpath
        #1 = neutral, 2 = calm, 3 = happy, 4 = sad, 5 = angry, 6 = fearful, 7 = disgust, 8 = surprised
        self.modalities = modalities
        if "speech" in modalities:
            self.classes = [
                DiscreteEmotion.Neutral,
                DiscreteEmotion.Calm,
                DiscreteEmotion.Happy,
                DiscreteEmotion.Sad,
                DiscreteEmotion.Angry,
                DiscreteEmotion.Fear,
                DiscreteEmotion.Disgust,
                DiscreteEmotion.Surprise
            ]
        else:
            self.classes = [
                DiscreteEmotion.Neutral,
                DiscreteEmotion.Calm,
                DiscreteEmotion.Happy,
                DiscreteEmotion.Sad,
                DiscreteEmotion.Angry,
                DiscreteEmotion.Fear
            ]

    def get_all(self):
        X = []
        Y = []
        for x_, y_ in self.get_all_yielded():
            X.append(x_)
            Y.append(y_)
        return (X, Y)
    
    def get_all_yielded(self):
        self.Metadata = []
        for audioname in os.listdir(self.folderpath):
            if splitext(audioname)[1].lower() == ".wav":
                fullname = join(self.folderpath, audioname)
                try:
                    rate, signal = wav.read(fullname)
                    meta = splitext(audioname)[0]
                    meta = meta.split("-")
                    meta = [int(e) for e in meta]
                    Y = self.classes[meta[2]-1]
                    self.Metadata.append([
                        rate,
                        meta[1],#1=speech, 2=song
                        meta[3], #intensity 1=normal, 2=strong. no strong in neutral
                        meta[4], #statement
                        meta[5], #repetition 1 or 2
                        meta[6], #actor
                        meta[6]%2]) #gender, 0 = female, 1 = male
                    yield (signal, Y)
                except:
                    print("error in reading ", audioname)
    
    def get_classes(self):
        return []
    
    def get_headers(self):
        return None #self.headers