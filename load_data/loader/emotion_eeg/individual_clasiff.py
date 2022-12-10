from load_data.ILoadSupervised import ILoadSupervised
from load_data.loader.util_emotions import DiscreteEmotion
import os

__all__ = ["LoadEEGIndividualEmotions"]


class LoadEEGIndividualEmotions(ILoadSupervised):
    def __init__(self, channels=None):
        self.fs = 1024
        self.channels = channels if isinstance(channels, list) else None
        self._path = "train_data/Folder_EEGEmotion/Dataset Individual Classification of Emotions Using EEG/"
        self.tags = {
            '01': DiscreteEmotion.Sad.name,
            '05': DiscreteEmotion.Disgust.name,
            '09': DiscreteEmotion.Neutral.name,
            '13': DiscreteEmotion.Amusement.name}
        self.classes = [self.tags[key] for key in self.tags]

    def get_default(self):
        return self.get_all()

    @staticmethod
    def get_splited():
        return None
    
    def get_all(self, verbose=False):
        xs = []
        ys = []
        files = os.listdir(self._path)
        i_file = 0
        for filename in files:
            if len(filename) != 8:
                continue
            i_file += 1
            if verbose:
                print("to process ", i_file, " of ", len(files), ": ", round(i_file/float(len(files))*100, 2),
                      "%, file: ", filename)
            tag = self.tags[filename[2:4]]
            file_obj = open(os.path.join(self._path, filename), "r")
            data_of_file = []
            i_channel = 0
            for line in file_obj:
                if self.channels is not None and i_channel in self.channels:
                    data_line = line.split("\t")
                    data_line = [float(element) for element in data_line]
                    data_of_file.append(data_line)
                i_channel += 1
            xs.append(data_of_file)
            ys.append(tag)
        return xs, ys
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return None  # self.headers
