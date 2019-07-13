import scipy.io.wavfile as wav
from load_data.ILoadSupervised import ILoadSupervised
from load_data.data_inside.not_shared.util_emotions import DiscreteEmotion
import os

__all__ = ["LoadBerlin",]

class LoadBerlin(ILoadSupervised):
    def __init__(self, classesBinaryArray=[1,1,1,1,1,1,1], foldername="train_data\\not_shared\\Folder_AudioEmotion\\Berlin\\wav"):
        self.foldername = foldername
        allClasses = [
            DiscreteEmotion.Angry.name,
            DiscreteEmotion.Boredom.name,
            DiscreteEmotion.Disgust.name,
            DiscreteEmotion.Fear.name,
            DiscreteEmotion.Happy.name,
            DiscreteEmotion.Sad.name,
            DiscreteEmotion.Neutral.name]
        self.classesBinaryArray = classesBinaryArray
        self.classes = []
        for i in range(len(allClasses)):
            if classesBinaryArray[i] == 1:
                self.classes.append(allClasses[i])
        self.speakers_info = {
            "03": ["male", 31],
            "08": ["female", 34],
            "09": ["female", 21],
            "10": ["male", 32],
            "11": ["male", 26],
            "12": ["male", 30],
            "13": ["female", 32],
            "14": ["female", 35],
            "15": ["male", 25],
            "16": ["female", 31],
        }
        self.texts = {
            "a01": ["Der Lappen liegt auf dem Eisschrank.", "The tablecloth is lying on the frigde."],
            "a02": ["Das will sie am Mittwoch abgeben.", "She will hand it in on Wednesday."],
            "a04": ["Heute abend könnte ich es ihm sagen.", "Tonight I could tell him."],
            "a05": ["Das schwarze Stück Papier befindet sich da oben neben dem Holzstück.", "The black sheet of paper is located up there besides the piece of timber."],
            "a07": ["In sieben Stunden wird es soweit sein.", "In seven hours it will be."],
            "b01": ["Was sind denn das für Tüten, die da unter dem Tisch stehen?", "What about the bags standing there under the table?"],
            "b02": ["Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter.", "They just carried it upstairs and now they are going down again."],
            "b03": ["An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.", "Currently at the weekends I always went home and saw Agnes."],
            "b09": ["Ich will das eben wegbringen und dann mit Karl was trinken gehen.", "I will just discard this and then go for a drink with Karl."],
            "b10": ["Die wird auf dem Platz sein, wo wir sie immer hinlegen.", "It will be in the place where we always store it."],
        }
        self.emotion_codes = {
            "W": DiscreteEmotion.Angry.name,
            "L": DiscreteEmotion.Boredom.name,
            "E": DiscreteEmotion.Disgust.name,
            "A": DiscreteEmotion.Fear.name,
            "F": DiscreteEmotion.Happy.name,
            "T": DiscreteEmotion.Sad.name,
            "N": DiscreteEmotion.Neutral.name
        }

    def get_default(self):
        return self.get_all()

    def get_splited(self):
        return None
    
    def get_all(self):
        X = []
        Y = []
        self.Metadata = []
        self.MetadataHeaders = [
            "rate",
            "sex",
            "age",
            "text german",
            "text translated",
            "version"
            ]
        for audio_filename in os.listdir(self.foldername):
            speaker_code = audio_filename[0:2]
            text_code = audio_filename[2:5]
            emotion_code = audio_filename[5]
            version_code = audio_filename[6]
            if self.emotion_codes[emotion_code] in self.classes:
                try:
                    rate, signal = wav.read(os.path.join(self.foldername, audio_filename))
                except:
                    print("error reading file: ", audio_filename)
                    continue
                X.append(signal)
                Y.append(self.emotion_codes[emotion_code])
                self.Metadata.append([
                    rate,
                    self.speakers_info[speaker_code][0],
                    self.speakers_info[speaker_code][1],
                    self.texts[text_code][0],
                    self.texts[text_code][1],
                    version_code])
        return (X, Y)
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return None #self.headers

