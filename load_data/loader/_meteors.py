from load_data.ILoadUnsupervised import ILoadUnsupervised
import csv
import preprocessing.verify_types as verify_types

__all__ = ["LoadMeteorsUnsupervised"]


class LoadMeteorsUnsupervised(ILoadUnsupervised):

    def __init__(self, path_to_csv="train_data/Folder_TimeSeries/meteors.csv"):
        self.path_to_csv = path_to_csv

    def get_headers(self):
        return ["year", "mass", "latitude", "longitude", "fell_found", "type_of_meteorite", "place"]

    def get_default(self):
        return self.read_file()

    def get_all(self):
        return self.read_file()

    def read_file(self):
        file_handler = open(self.path_to_csv, "r", encoding="utf-8-sig")
        file_reader = csv.DictReader(file_handler, delimiter=';')
        data = []
        for row in file_reader:
            if row["year"] != "" and verify_types.is_integer(row["year"]) \
                    and row["mass_g"] != "0" and verify_types.is_float(row["mass_g"].replace(",", ".")) \
                    and verify_types.is_float(row["latitude"].replace(",", ".")) \
                    and verify_types.is_float(row["longitude"].replace(",", ".")):
                data.append([
                    int(row["year"]),
                    float(row["mass_g"].replace(",", ".")),
                    float(row["latitude"].replace(",", ".")),
                    float(row["longitude"].replace(",", ".")),
                    row["fell_found"],
                    row["type_of_meteorite"],
                    row["place"]
                ])
        file_handler.close()
        return data
