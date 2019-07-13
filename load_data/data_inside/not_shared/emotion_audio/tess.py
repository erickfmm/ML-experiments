import scipy.io.wavfile as wav
from load_data.ILoadSupervised import ILoadSupervised
from load_data.data_inside.not_shared.util_emotions import DiscreteEmotion
import os

__all__ = ["LoadTESS",]

class LoadTESS(ILoadSupervised):
    def __init__(self, classesBinaryArray=[1,1,1,1,1,1,1], foldername="train_data\\not_shared\\Folder_AudioEmotion\\TESS"):
        self.foldername = foldername
        allClasses = [
            DiscreteEmotion.Neutral.name,
            DiscreteEmotion.Angry.name,
            DiscreteEmotion.Disgust.name,
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
        self.MetadataHeaders = ["rate", "sex", "age", "text"]
        for audio_filename in os.listdir(self.foldername):
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
                    rate, signal = wav.read(os.path.join(self.foldername, audio_filename))
                except:
                    print("error reading file: ", audio_filename)
                    continue
                X.append(signal)
                Y.append(emo_file)
                self.Metadata.append([rate, sex_speaker, age_speaker, text])
        return (X, Y)
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return None #self.headers

