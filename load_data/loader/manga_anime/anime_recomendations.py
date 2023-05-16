from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
from csv import DictReader
from os.path import join

__all__ = ["LoadAnimeData"]


class LoadAnimeData(ILoadSupervised):
    def __init__(self, folder_path="data/train_data/Manga_Anime/anime-recommendations-database"):
        self.folder_path = folder_path
        self.TYPE = SupervisedType.Regression

    def get_all(self):
        xs = []
        ys = []
        for x, y in self.get_all_yielded():
            xs.append(x)
            ys.append(y)
        return xs, ys

    def get_all_yielded(self):
        with open(join(self.folder_path, "anime.csv"), "r", encoding="utf-8") as file_obj: #TODO: load rating.csv
            filereader = DictReader(file_obj)
            i = 0
            for row in filereader:
                i += 1
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
                except Exception as e:
                    error_msg = ""
                    if episodes is None:
                        error_msg += ". Episodes is not integer: "+row["episodes"]
                    if rating is None:
                        error_msg += ". Rating is not float: "+row["rating"]
                    print(i, ": ", name, error_msg, " error: ", e)
                if episodes is not None and rating is not None:
                    yield [
                        name,
                        genre,
                        row["type"],
                        episodes,
                        int(row["members"])
                        ], rating

    def get_classes(self):
        return []
    
    def get_headers(self):
        return ["name", "genres", "type", "episodes", "members"]
