from load_data.ILoadUnsupervised import ILoadUnsupervised
import csv

__all__ = ["LoadOnuDebates"]


class LoadOnuDebates(ILoadUnsupervised):
    def __init__(self,
                 path_to_csv="train_data/Folder_NLPEnglish/ONU Debates de asambleas generales/un-general-debates.csv",
                 country_codes_csv="train_data/Folder_NLPEnglish/"
                                   "ONU Debates de asambleas generales/ISO-Country-Codes.csv"):
        self.path_to_csv = path_to_csv
        self.country_codes_csv = country_codes_csv

    def get_headers(self):
        return ["year", "country", "session", "text"]

    def get_default(self):
        return self.read_file()

    def get_all(self):
        return self.read_file()

    def read_country_codes(self):
        fobj = open(self.country_codes_csv, "r", encoding="utf-8")
        file_reader = csv.DictReader(fobj, delimiter=',')
        country_names = {}
        for row in file_reader:
            country_names[row["Alpha-3 code"]] = row["English short name"]
        fobj.close()
        return country_names

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
