import scipy.io.wavfile as wav
from load_data.ILoadSupervised import ILoadSupervised
from load_data.loader.util_emotions import DiscreteEmotion
import os
from os.path import join, splitext

__all__ = ["LoadRavdess"]


class LoadRavdess(ILoadSupervised):
    def __init__(self, modalities=None,
                 folder_path="train_data/Folder_AudioEmotion/RAVDESS"):
        if modalities is None:
            modalities = ["speech", "song"]
        self.folder_path = folder_path
        # 1 = neutral, 2 = calm, 3 = happy, 4 = sad, 5 = angry, 6 = fearful, 7 = disgust, 8 = surprised
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
        self.Metadata = []

    def get_all(self):
        xs = []
        ys = []
        for x_, y_ in self.get_all_yielded():
            xs.append(x_)
            ys.append(y_)
        return xs, ys
    
    def get_all_yielded(self):
        for audio_name in os.listdir(self.folder_path):
            if splitext(audio_name)[1].lower() == ".wav":
                fullname = join(self.folder_path, audio_name)
                try:
                    rate, signal = wav.read(fullname)
                    meta = splitext(audio_name)[0]
                    meta = meta.split("-")
                    meta = [int(e) for e in meta]
                    y = self.classes[meta[2]-1]
                    self.Metadata.append([
                        rate,
                        meta[1],  # 1=speech, 2=song
                        meta[3],  # intensity 1=normal, 2=strong. no strong in neutral
                        meta[4],  # statement
                        meta[5],  # repetition 1 or 2
                        meta[6],  # actor
                        meta[6] % 2])  # gender, 0 = female, 1 = male
                    yield signal, y
                except Exception as e:
                    print("error in reading ", audio_name, " error:", e)
    
    def get_classes(self):
        return []
    
    def get_headers(self):
        return None  # self.headers
