import scipy.io.wavfile as wav
from load_data.ILoadSupervised import ILoadSupervised
from load_data.data_inside.not_shared.util_emotions import DiscreteEmotion
import os
from os.path import join, splitext

__all__ = ["LoadSavee",]

class LoadSavee(ILoadSupervised):
    def __init__(self, folderpath="train_data\\not_shared\\Folder_AudioEmotion\\SAVEE"):
        self.folderpath = folderpath
        self.classes_dict = {
            "a": DiscreteEmotion.Angry,
            "d": DiscreteEmotion.Disgust,
            "f": DiscreteEmotion.Fear,
            "h": DiscreteEmotion.Happy,
            "n": DiscreteEmotion.Neutral,
            "sa": DiscreteEmotion.Sad,
            "su": DiscreteEmotion.Surprise
        }

    def get_all(self):
        X = []
        Y = []
        for x_, y_ in self.get_all_yielded():
            X.append(x_)
            Y.append(y_)
        return (X, Y)
    
    def get_all_yielded(self):
        audiofolder = join(self.folderpath, "AudioData")
        folder_speakers = ["DC", "JE", "JK", "KL"]
        self.Metadata = []
        for speaker in folder_speakers:
            for audioname in os.listdir(join(audiofolder, speaker)):
                if splitext(audioname)[1].lower() == ".wav":
                    fullname = join(audiofolder, speaker, audioname)
                    rate, signal = wav.read(fullname)
                    self.Metadata.append([speaker, rate])
                    y_ = audioname[0]
                    if audioname[0].lower() == "s":
                        y_ = audioname[:2]
                    Y = self.classes_dict[y_]
                    yield (signal, Y)
    
    def get_classes(self):
        return [DiscreteEmotion.Angry,
                DiscreteEmotion.Disgust,
                DiscreteEmotion.Fear,
                DiscreteEmotion.Happy,
                DiscreteEmotion.Neutral,
                DiscreteEmotion.Sad,
                DiscreteEmotion.Surprise]
    
    def get_headers(self):
        return None #self.headers