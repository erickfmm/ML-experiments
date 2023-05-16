from load_data.ILoadUnsupervised import ILoadUnsupervised
from os.path import join

__all__ = ["LoadDictionaryDuenasLerin", "LoadDictionaryHunspell", "LoadDictionaryRaeComplete"]


class LoadDictionaryAbstract(ILoadUnsupervised):
    def __init__(self):
        self.filepath = None
        self.filename = None
        self.fullpath = None

    def get_headers(self):
        return None

    def get_all(self):
        words = []
        for word in self.get_all_yielded():
            words.append(word)
        return words
    
    def get_all_yielded(self):
        file_obj = open(self.fullpath, "r", encoding="utf-8")
        for word in file_obj:
            yield word.strip().lower()


class LoadDictionaryDuenasLerin(LoadDictionaryAbstract):
    def __init__(self, filepath="train_data/Folder_NLPEspañol/diccionario-espanol-txt-master", letter=None):
        super().__init__()
        self.filepath = filepath
        self.filename = "palabras_"+str(letter)+".txt" if letter is not None else "palabras_todas.txt"
        self.fullpath = join(self.filepath, self.filename)


class LoadDictionaryHunspell(LoadDictionaryAbstract):
    def __init__(self, filepath="train_data/Folder_NLPEspañol/dictionaries-master/dictionaries", lang="es"):
        super().__init__()
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


class LoadDictionaryRaeComplete(LoadDictionaryAbstract):
    def __init__(self, filepath="train_data/Folder_NLPEspañol/palabras-diccionario-rae-completo-master", letter=None,
                 version="completo"):  # "completo","hunspell","full","letter"
        super().__init__()
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
        file_obj = open(self.fullpath, "r", encoding="utf-8")
        for word in file_obj:
            if self.version == "hunspell":
                s = word.strip().lower()
                s = s[:s.find("/")]
                if len(s) > 0:
                    yield s


def get_words_from_all_dictionaries():
    ws = set(LoadDictionaryAbstract().get_all())
    ws = ws.union(LoadDictionaryHunspell().get_all())
    ws = ws.union(LoadDictionaryRaeComplete(version="completo").get_all())
    ws = ws.union(LoadDictionaryRaeComplete(version="hunspell").get_all())
    ws = ws.union(LoadDictionaryRaeComplete(version="full").get_all())
    return ws
