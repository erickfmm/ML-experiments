# -*- coding: utf-8 -*-

from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
from csv import DictReader
from os.path import join

class LoadAnimeData(ILoadSupervised):
    def __init__(self, folderpath="train_data\\not_shared\\Folder_Manga_Anime\\anime-recommendations-database"):
        self.folderpath = folderpath
        self.TYPE = SupervisedType.Regression
        
    
    def get_all(self):
        xs = []
        ys = []
        for x, y in self.get_all_yielded():
            xs.append(x)
            ys.append(y)
        return xs, ys

    def get_all_yielded(self):
        with open(join(self.folderpath, "anime.csv"), "r", encoding="utf-8") as fileobj:
            filereader = DictReader(fileobj)
            i = 0
            for row in filereader:
                i+=1
                name = row["name"]
                name = name.replace("°", "")
                name = name.replace("&#039;", "")
                name = name.strip()
                genre = row["genre"].split(", ")
                episodes = None
                rating = None
                try:
                    episodes = int(row["episodes"])
                    rating = float(row["rating"])
                except:
                    error_msg = ""
                    if episodes is None:
                        error_msg += ". Episodes is not integer: "+row["episodes"]
                    if rating is None:
                        error_msg += ". Rating is not float: "+row["rating"]
                    print(i,": ", name, error_msg)
                if episodes is not None and rating is not None:
                    yield ([
                        name,
                        genre,
                        row["type"],
                        episodes,
                        int(row["members"])
                        ],rating )
                

    def get_classes(self):
        return []
    
    def get_headers(self):
        return ["name", "genres", "type", "episodes", "members"]