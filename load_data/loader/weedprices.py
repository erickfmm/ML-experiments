# -*- coding: utf-8 -*-

from load_data.ILoadUnsupervised import ILoadUnsupervised
from os.path import join, splitext
from os import listdir
import csv
from datetime import date

class LoadWeedPrices(ILoadUnsupervised):

    def __init__(self, \
    datapath="train_data/Folder_TimeSeries/price-of-weed-master/data"):
        self.datapath = datapath
        self.states = []
        
    def get_headers(self):
        return ["year", "country", "session", "text"]

    def get_all(self):
        data = []
        dates = []
        for filename in listdir(self.datapath):
            if splitext(filename)[1] == ".csv":
                readed = False
                with open(join(self.datapath, filename), "r") as fileobj:
                    reader = csv.DictReader(fileobj)
                    for row in reader:
                        if not readed:
                            readed = True
                        if row["State"] not in self.states:
                            self.states.append(row["State"])
                            data.append([[], [], [], [], [], []])
                        index = self.states.index(row["State"])
                        data[index][0].append(float(row["HighQ"].replace("$", "")))
                        data[index][1].append(float(row["MedQ"].replace("$", "")))
                        if row["LowQ"] == "I feel bad for these guys -->":
                            data[index][2].append(None)
                        else:
                            data[index][2].append(float(row["LowQ"].replace("$", "")))
                        data[index][3].append(int(row["HighQN"]))
                        data[index][4].append(int(row["MedQN"]))
                        data[index][5].append(int(row["LowQN"]))
                if readed:
                    date_name = filename.replace(".csv", "").replace("weedprices", "")
                    dates.append(date(year = int(date_name[4:8]), month = int(date_name[2:4]), day = int(date_name[0:2])))
        return data, dates


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    lwd = LoadWeedPrices()
    data, dates = lwd.get_all()
    plt.plot(dates, data[0][0], ".")
    plt.plot(dates, data[0][1], ".")
    plt.plot(dates, data[0][2], ".")
    plt.legend(["high", "med", "low"])
    plt.show()