import scipy.io.wavfile as wav
from load_data.ILoadSupervised import ILoadSupervised
from load_data.loader.util_emotions import DiscreteEmotion
import os

__all__ = ["LoadAditya"]


class LoadAditya(ILoadSupervised):
    def __init__(self, classes_binary_array=None,
                 folder_name="train_data/Folder_AudioEmotion/Aditya_Recordings"):
        if classes_binary_array is None:
            classes_binary_array = [1, 1, 1, 1, 1]  # using all classes
        self.folder_name = folder_name
        self.rate = []
        all_classes = [
            DiscreteEmotion.Angry.name,
            DiscreteEmotion.Fear.name,
            DiscreteEmotion.Happy.name,
            DiscreteEmotion.Sad.name,
            DiscreteEmotion.Surprise.name]
        self.classes_binary_array = classes_binary_array
        self.classes = []
        for i in range(len(all_classes)):
            if classes_binary_array[i] == 1:
                self.classes.append(all_classes[i])

    def get_default(self):
        return self.get_all()

    @staticmethod
    def get_splited():
        return None
    
    def get_all(self):
        xs = []
        ys = []
        folders = ["angry", "fear", "happy", "sad", "surprised"]
        icl = 0
        for ifolder in range(len(folders)):
            if self.classes_binary_array[ifolder] == 1:
                emo_folder = os.path.join(self.folder_name, folders[ifolder])
                audio_files = os.listdir(emo_folder)
                for file_name in audio_files:
                    rate, signal = wav.read(os.path.join(emo_folder, file_name))
                    self.rate.append(rate)
                    xs.append(signal)
                    ys.append(self.classes[icl])
                icl += 1
        return xs, ys
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return ["audio"]
