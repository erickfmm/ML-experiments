import scipy.io.wavfile as wav
from load_data.ILoadSupervised import ILoadSupervised
from load_data.loader.util_emotions import DiscreteEmotion
import os
from os.path import join, splitext

__all__ = ["LoadSavee"]


class LoadSavee(ILoadSupervised):
    def __init__(self, folder_path="train_data/Folder_AudioEmotion/SAVEE"):
        self.folder_path = folder_path
        self.classes_dict = {
            "a": DiscreteEmotion.Angry,
            "d": DiscreteEmotion.Disgust,
            "f": DiscreteEmotion.Fear,
            "h": DiscreteEmotion.Happy,
            "n": DiscreteEmotion.Neutral,
            "sa": DiscreteEmotion.Sad,
            "su": DiscreteEmotion.Surprise
        }
        self.Metadata = []

    def get_all(self):
        xs = []
        ys = []
        for x_, y_ in self.get_all_yielded():
            xs.append(x_)
            ys.append(y_)
        return xs, ys
    
    def get_all_yielded(self):
        audio_folder = join(self.folder_path, "AudioData")
        folder_speakers = ["DC", "JE", "JK", "KL"]
        for speaker in folder_speakers:
            for audio_name in os.listdir(join(audio_folder, speaker)):
                if splitext(audio_name)[1].lower() == ".wav":
                    fullname = join(audio_folder, speaker, audio_name)
                    rate, signal = wav.read(fullname)
                    self.Metadata.append([speaker, rate])
                    y_ = audio_name[0]
                    if audio_name[0].lower() == "s":
                        y_ = audio_name[:2]
                    y = self.classes_dict[y_]
                    yield signal, y
    
    def get_classes(self):
        return [DiscreteEmotion.Angry,
                DiscreteEmotion.Disgust,
                DiscreteEmotion.Fear,
                DiscreteEmotion.Happy,
                DiscreteEmotion.Neutral,
                DiscreteEmotion.Sad,
                DiscreteEmotion.Surprise]
    
    def get_headers(self):
        return None  # self.headers
