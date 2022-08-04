from load_data.ILoadUnsupervised import ILoadUnsupervised
from os.path import join
#import sys

class LoadWMT11(ILoadUnsupervised):

    #datapath="train_data/Folder_NLPEspañol_Translation/WMT13 [ES-EN]/training"
    def __init__(self, datapath="train_data/Folder_NLPEspañol_Translation/WMT11_Translation/training", \
                 dataset="news-commentary-v6"):
        """
        nombre + ".es-en.en" y ".es-en.es"
        wmt11:
            - europarl-v6
            - news-commentary-v6
        wmt13:
            - commoncrawl.es-en
            - europarl-v7.es-en
            - news-commentary-v8.es-en
            - undoc.2000.es-en
        """
        self.datapath = datapath
        #self.filename_eng = "europarl-v6.es-en.en" if dataset == "europarl" else "news-commentary-v6.es-en.en"
        #self.filename_es = "europarl-v6.es-en.es" if dataset == "europarl" else "news-commentary-v6.es-en.es"
        self.filename_eng = dataset + ".es-en.en"
        self.filename_es = dataset + ".es-en.es"

    def get_headers(self):
        return None

    def get_all(self):
        data_eng = []
        data_es = []
        for eng, es in self.get_all_yielded():
            data_eng.append(eng)
            data_es.append(es)
        return data_eng, data_es
    
    def get_all_yielded(self):
        fullname_eng = join(self.datapath, self.filename_eng)
        eng_obj = open(fullname_eng, "r", encoding="utf-8")
        fullname_es = join(self.datapath, self.filename_es)
        es_obj = open(fullname_es, "r", encoding="utf-8")
        print(fullname_eng)
        has_lines = True
        while has_lines:
            try:
                eng_line = eng_obj.readline()
                es_line = es_obj.readline()
                if len(eng_line) == 0 and len(es_line) == 0:
                    has_lines = False
                else:
                    yield eng_line.strip(), es_line.strip()
            except:
                has_lines = False
                eng_obj.close()
                es_obj.close()
        eng_obj.close()
        es_obj.close()