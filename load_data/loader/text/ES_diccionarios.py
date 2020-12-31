from load_data.ILoadUnsupervised import ILoadUnsupervised
from os.path import join
#import csv

class LoadDiccionario_Abstract(ILoadUnsupervised):
    def __init__(self):
        self.filepath = None
        self.filename = None

    def get_headers(self):
        return None

    def get_all(self):
        words = []
        for word in self.get_all_yielded():
            words.append(word)
        return words
    
    def get_all_yielded(self):
        fobj = open(self.fullpath, "r", encoding="utf-8")
        for word in fobj:
            yield word.strip().lower()

class LoadDiccionario_DuenasLerin(LoadDiccionario_Abstract):
    def __init__(self, 
        filepath="train_data/Folder_NLPEspañol/diccionario-espanol-txt-master",
        letter=None):
        self.filepath = filepath
        self.filename = "palabras_"+str(letter)+".txt" if letter is not None else "palabras_todas.txt"
        self.fullpath = join(self.filepath, self.filename)



class LoadDiccionario_Hunspell(LoadDiccionario_Abstract):
    def __init__(self, 
        filepath="train_data/Folder_NLPEspañol/dictionaries-master/dictionaries",
        lang="es"):
        self.filepath = filepath
        self.fullpath = join(self.filepath, str(lang), "index.dic")

    def get_all_yielded(self):
        fobj = open(self.fullpath, "r", encoding="utf-8")
        is_first = True
        for word in fobj:
            if is_first:
                is_first = False
                continue
            s = word.strip().lower()
            s = s[:s.find("/")]
            if len(s) > 0:
                yield s


class LoadDiccionario_Rae_completo(LoadDiccionario_Abstract):
    def __init__(self, 
        filepath="train_data/Folder_NLPEspañol/palabras-diccionario-rae-completo-master",
        letter=None,
        version="completo"):#"completo","hunspell","full","letter"
        self.filepath = filepath
        self.filename = "diccionario-rae-completo.txt"
        self.version = version
        if version == "hunspell":
            self.filename = join("fuentes", "diccionario-hunspell.txt")
        elif version == "full":
            self.filename = join("fuentes", "dict.txt")
        elif version == "letter":
            self.filename = join("fuentes", str(letter)+".txt")
        self.fullpath = join(self.filepath, self.filename)

    def get_all_yielded(self):
        fobj = open(self.fullpath, "r", encoding="utf-8")
        for word in fobj:
            if self.version == "hunspell":
                s = word.strip().lower()
                s = s[:s.find("/")]
                if len(s) > 0:
                    yield s

def get_words_from_all_dictionaries():
    ws = set(LoadDiccionario_DuenasLerin().get_all())
    ws = ws.union(LoadDiccionario_Hunspell().get_all())
    ws = ws.union(LoadDiccionario_Rae_completo(version="completo").get_all())
    ws = ws.union(LoadDiccionario_Rae_completo(version="hunspell").get_all())
    ws = ws.union(LoadDiccionario_Rae_completo(version="full").get_all())
    return ws
