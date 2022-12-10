import scipy.io.wavfile as wav
from load_data.ILoadSupervised import ILoadSupervised
from load_data.loader.util_emotions import DiscreteEmotion
import os

__all__ = ["LoadBerlin"]


class LoadBerlin(ILoadSupervised):
    def __init__(self, classes_binary_array=None,
                 folder_name="train_data/Folder_AudioEmotion/Berlin/wav"):
        if classes_binary_array is None:
            classes_binary_array = [1, 1, 1, 1, 1, 1, 1]
        self.folder_name = folder_name
        all_classes = [
            DiscreteEmotion.Angry.name,
            DiscreteEmotion.Boredom.name,
            DiscreteEmotion.Disgust.name,
            DiscreteEmotion.Fear.name,
            DiscreteEmotion.Happy.name,
            DiscreteEmotion.Sad.name,
            DiscreteEmotion.Neutral.name]
        self.classes_binary_array = classes_binary_array
        self.classes = []
        for i in range(len(all_classes)):
            if classes_binary_array[i] == 1:
                self.classes.append(all_classes[i])
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
            "a01": ["Der Lappen liegt auf dem Eisschrank.",
                    "The tablecloth is lying on the frigde."],
            "a02": ["Das will sie am Mittwoch abgeben.",
                    "She will hand it in on Wednesday."],
            "a04": ["Heute abend könnte ich es ihm sagen.",
                    "Tonight I could tell him."],
            "a05": ["Das schwarze Stück Papier befindet sich da oben neben dem Holzstück.",
                    "The black sheet of paper is located up there besides the piece of timber."],
            "a07": ["In sieben Stunden wird es soweit sein.",
                    "In seven hours it will be."],
            "b01": ["Was sind denn das für Tüten, die da unter dem Tisch stehen?",
                    "What about the bags standing there under the table?"],
            "b02": ["Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter.",
                    "They just carried it upstairs and now they are going down again."],
            "b03": ["An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.",
                    "Currently at the weekends I always went home and saw Agnes."],
            "b09": ["Ich will das eben wegbringen und dann mit Karl was trinken gehen.",
                    "I will just discard this and then go for a drink with Karl."],
            "b10": ["Die wird auf dem Platz sein, wo wir sie immer hinlegen.",
                    "It will be in the place where we always store it."],
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
        self.Metadata = []
        self.MetadataHeaders = [
            "rate",
            "sex",
            "age",
            "text german",
            "text translated",
            "version"
            ]

    def get_default(self):
        return self.get_all()

    @staticmethod
    def get_splited():
        return None
    
    def get_all(self):
        xs = []
        ys = []
        for audio_filename in os.listdir(self.folder_name):
            speaker_code = audio_filename[0:2]
            text_code = audio_filename[2:5]
            emotion_code = audio_filename[5]
            version_code = audio_filename[6]
            if self.emotion_codes[emotion_code] in self.classes:
                try:
                    rate, signal = wav.read(os.path.join(self.folder_name, audio_filename))
                except Exception as e:
                    print("error reading file: ", audio_filename, " error", e)
                    continue
                xs.append(signal)
                ys.append(self.emotion_codes[emotion_code])
                self.Metadata.append([
                    rate,
                    self.speakers_info[speaker_code][0],
                    self.speakers_info[speaker_code][1],
                    self.texts[text_code][0],
                    self.texts[text_code][1],
                    version_code])
        return xs, ys
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return ["audio"]
