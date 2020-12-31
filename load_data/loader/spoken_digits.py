from load_data.ILoadSupervised import ILoadSupervised
import scipy.io.wavfile as wav
import os

__all__ = ["LoadSpokenDigits",]

class LoadSpokenDigits(ILoadSupervised):
    def __init__(self, digits=[0,1,2,3,4,5,6,7,8,9], \
        path="train_data/free-spoken-digit/recordings"):
        self.digits = digits
        self.path_files = path

    def get_default(self):
        return self.load_files()

    def get_splited(self):
        return None
    
    def get_all(self):
        X = []
        Y = []
        for x_, y_ in self.get_all_yielded():
            X.append(x_)
            Y.append(y_)
        return X, Y
    
    def get_classes(self):
        return self.digits #self.classes
    
    def get_headers(self):
        return None #self.headers

    def get_all_yielded(self):
        self.Metadata = []
        self.MetadataHeaders = ["PersonName", "iteration", "rate"]
        wav_filenames = os.listdir(self.path_files)
        for filename in wav_filenames:
            full_filename = os.path.join(self.path_files, filename)
            name_segments = filename.replace(".wav", "").split("_")
            if int(name_segments[0]) not in self.digits:
                continue
            rate, signal = wav.read(full_filename)
            self.Metadata.append([name_segments[1], int(name_segments[2]), rate])
            yield signal, int(name_segments[0])
