from mlexperiments.load_data.ILoadUnsupervised import ILoadUnsupervised
import csv
import opendatasets as od

__all__ = ["LoadOnuDebates"]


class LoadOnuDebates(ILoadUnsupervised):
    def __init__(self,
                 path_to_csv="data/train_data/NLP_ENG_Dialogs/un-general-debates.csv",):
        self.path_to_csv = path_to_csv

    def get_headers(self):
        return ["year", "country", "session", "text"]

    def get_data(self):
        return self.read_file()


    def read_file(self):
        # country_names = self.read_country_codes()
        file_obj = open(self.path_to_csv, "r", encoding="utf-8-sig")
        file_reader = csv.DictReader(file_obj, delimiter=',')
        data = []
        for row in file_reader:
            # if row["country"] not in country_names:
            #    print("not found: ", row["country"])
            text = row["text"] if row["text"][0] != "\ufeff" else row["text"][1:]
            data.append([
                int(row["year"]),
                row["country"],
                int(row["session"]),
                text
            ])
        file_obj.close()
        return data

    def download(self, folder_path="data/train_data/NLP_ENG_Dialogs"):
        od.download("https://www.kaggle.com/datasets/unitednations/un-general-debates", folder_path)
