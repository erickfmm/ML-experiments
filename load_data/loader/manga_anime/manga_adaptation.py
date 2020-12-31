# -*- coding: utf-8 -*-

from load_data.ILoadUnsupervised import ILoadUnsupervised
from csv import DictReader

class LoadMangaAdaptations(ILoadUnsupervised):
    def __init__(self, csvpath="train_data/Folder_Manga_Anime/MangaAdaptaitonScore.csv"):
        self.csvpath = csvpath

    def get_headers(self):
        return []

    def get_all(self):
        all_data = []
        for data in self.get_all_yield():
            all_data.append(data)
        return all_data

    def get_values(self):
        manga = []
        anime = []
        for row in self.get_all_yield():
            manga.append(row[2])
            anime.append(row[3])
        return manga, anime

    def get_all_yield(self):
        with open(self.csvpath, "r") as csvobj:
            csvreader = DictReader(csvobj)
            for row in csvreader:
                try:
                    manga_title = row["mangaTitle"]
                    anime_title = row["adaptationTitle"]
                    manga_rating = float(row["mangaRating"])
                    anime_rating = float(row["adaptationRating"])
                    yield [manga_title, anime_title, manga_rating, anime_rating]
                except:
                    pass