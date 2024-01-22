from mlexperiments.load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import librosa
import os
import opendatasets as od

__all__ = ["LoadBirdSongs"]


class LoadBirdSongs(ILoadSupervised):
    def __init__(self, path="data/train_data/Audio/birdsongs-from-europe"):
        self.TYPE = SupervisedType.Classification
        self.path_files = path
        self.metadata = []

    def get_X_Y(self):
        xs = []
        ys = []
        for x_, y_ in self.get_X_Y_yielded():
            xs.append(x_)
            ys.append(y_)
        return xs, ys
    
    def get_classes(self):
        return None 
    
    def get_headers(self):
        return None

    def get_X_Y_yielded(self):
        wav_filenames = os.listdir(self.path_files, "mp3")
        for filename in wav_filenames:
            full_filename = os.path.join(self.path_files, "mp3", filename)
            name_segments = filename.replace(".mp3", "").split("-")
            signal, rate = librosa.load(full_filename)
            self.metadata.append([name_segments[0], name_segments[1], name_segments[2], rate])
            yield signal, name_segments[0]+name_segments[1]
    
    def get_classes(self):
        return []
    
    def get_headers(self):
        return ["Audio"]  # self.headers

    def download(self, folder_path="data/train_data/Audio/"):
        od.download("https://www.kaggle.com/datasets/monogenea/birdsongs-from-europe", folder_path)
