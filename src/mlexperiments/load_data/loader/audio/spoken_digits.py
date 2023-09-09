from mlexperiments.load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import scipy.io.wavfile as wav
import os

__all__ = ["LoadSpokenDigits"]


class LoadSpokenDigits(ILoadSupervised):
    def __init__(self, digits=None, path="data/train_data/Audio/free-spoken-digit-dataset/recordings"):
        self.TYPE = SupervisedType.Classification
        if digits is None:
            digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.digits = digits
        self.path_files = path
        self.metadata = []
        self.metadata_headers = ["PersonName", "iteration", "rate"]

    def get_X_Y(self):
        xs = []
        ys = []
        for x_, y_ in self.get_X_Y_yielded():
            xs.append(x_)
            ys.append(y_)
        return xs, ys
    
    def get_classes(self):
        return self.digits 
    
    def get_headers(self):
        return ["Audio"]

    def get_X_Y_yielded(self):
        wav_filenames = os.listdir(self.path_files)
        for filename in wav_filenames:
            full_filename = os.path.join(self.path_files, filename)
            name_segments = filename.replace(".wav", "").split("_")
            if int(name_segments[0]) not in self.digits:
                continue
            rate, signal = wav.read(full_filename)
            self.metadata.append([name_segments[1], int(name_segments[2]), rate])
            yield signal, int(name_segments[0])
