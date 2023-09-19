import scipy.io.wavfile as wav
from mlexperiments.load_data.ILoadSupervised import ILoadSupervised
from mlexperiments.load_data.loader.util_emotions import DiscreteEmotion
import os
import opendatasets as od


__all__ = ["LoadTESS"]


class LoadTESS(ILoadSupervised):
    def __init__(self, classes_binary_array=None,
                 folder_name="data/train_data/Emotions_Voice/toronto-emotional-speech-set-tess/TESS Toronto emotional speech set data/TESS Toronto emotional speech set data"):
        if classes_binary_array is None:
            classes_binary_array = [1, 1, 1, 1, 1, 1, 1]
        self.folder_name = folder_name
        all_classes = [
            DiscreteEmotion.Neutral.name,
            DiscreteEmotion.Angry.name,
            DiscreteEmotion.Disgust.name,
            DiscreteEmotion.Fear.name,
            DiscreteEmotion.Happy.name,
            DiscreteEmotion.Sad.name,
            DiscreteEmotion.Surprise.name]
        self.classes_binary_array = classes_binary_array
        self.classes = []
        for i in range(len(all_classes)):
            if classes_binary_array[i] == 1:
                self.classes.append(all_classes[i])
        self.Metadata = []
        self.MetadataHeaders = ["rate", "sex", "age", "text"]
    
    def get_X_Y(self):
        xs = []
        ys = []
        for folder_name in os.listdir(self.folder_name):
            for audio_filename in os.listdir(os.path.join(self.folder_name, folder_name)):
                if audio_filename.find(".wav") == -1:
                    print("its not a wav file: ", audio_filename)
                    continue
                segments = audio_filename.replace(".wav", "").split("_")
                emo_file = segments[2].capitalize() if segments[2] != "ps" else DiscreteEmotion.Surprise.name
                if emo_file in self.classes:
                    sex_speaker = "female"
                    age_speaker = 26 if segments[0] == "YAF" else 64
                    text = segments[1]
                    try:
                        rate, signal = wav.read(os.path.join(self.folder_name, folder_name, audio_filename))
                    except Exception as e:
                        print("error reading file: ", audio_filename, " error: ", e)
                        continue
                    xs.append(signal)
                    ys.append(emo_file)
                    self.Metadata.append([rate, sex_speaker, age_speaker, text])
        return xs, ys
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return None  # self.headers

    def download(self):
        od.download("https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess", "data/train_data/Emotions_Voice")
