import scipy.io.wavfile as wav
from load_data.ILoadSupervised import ILoadSupervised
from load_data.data_inside.not_shared.util_emotions import DiscreteEmotion
import os

__all__ = ["LoadAditya",]

class LoadAditya(ILoadSupervised):
    def __init__(self, classesBinaryArray=[1,1,1,1,1], foldername="train_data\\not_shared\\Folder_AudioEmotion\\Aditya_Recordings"):
        self.foldername = foldername
        allClasses = [
            DiscreteEmotion.Angry.name,
            DiscreteEmotion.Fear.name,
            DiscreteEmotion.Happy.name,
            DiscreteEmotion.Sad.name,
            DiscreteEmotion.Surprise.name]
        self.classesBinaryArray = classesBinaryArray
        self.classes = []
        for i in range(len(allClasses)):
            if classesBinaryArray[i] == 1:
                self.classes.append(allClasses[i])

    def get_default(self):
        return self.get_all()

    def get_splited(self):
        return None
    
    def get_all(self):
        X = []
        Y = []
        self.Metadata = []
        folders = ["angry", "fear", "happy", "sad", "surprised"]
        icl = 0
        for ifolder in range(len(folders)):
            if self.classesBinaryArray[ifolder] == 1:
                emo_folder = os.path.join(self.foldername, folders[ifolder])
                audio_files = os.listdir(emo_folder)
                for fname in audio_files:
                    rate, signal = wav.read(os.path.join(emo_folder, fname))
                    self.Metadata.append(rate)
                    X.append(signal)
                    Y.append(self.classes[icl])
                icl += 1
        return (X, Y)
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return None #self.headers

