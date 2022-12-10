import scipy.io.wavfile as wav
from load_data.ILoadSupervised import ILoadSupervised
from load_data.loader.util_emotions import DiscreteEmotion
import os

__all__ = ["LoadTESS"]


class LoadTESS(ILoadSupervised):
    def __init__(self, classes_binary_array=None,
                 folder_name="train_data/Folder_AudioEmotion/TESS"):
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

    def get_default(self):
        return self.get_all()

    @staticmethod
    def get_splited():
        return None
    
    def get_all(self):
        xs = []
        ys = []
        for audio_filename in os.listdir(self.folder_name):
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
                    rate, signal = wav.read(os.path.join(self.folder_name, audio_filename))
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
