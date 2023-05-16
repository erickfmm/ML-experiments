# -*- coding: utf-8 -*-

# from pims import ImageReader
from PIL import Image
from load_data.ILoadSupervised import ILoadSupervised
from os.path import join, exists
import csv

__all__ = ["LoadPokemon"]


class LoadPokemon(ILoadSupervised):
    def __init__(self, path="train_data/Folder_Videojuegos/pokemon-images-and-types"):
        self.path = path
        self.classes = set()
        
    def get_all(self, sum1=False):
        xs = []
        ys_not_processed = []
        with open(join(self.path, "pokemon.csv"), "r") as csv_obj:
            csv_reader = csv.DictReader(csv_obj)
            for row in csv_reader:
                image_name = join(self.path, "images", row["Name"]+".png")
                if exists(image_name):
                    im = Image.open(image_name)
                    xs.append(im)
                    self.classes.add(row["Type1"])
                    actual_ys = [row["Type1"]]
                    if row["Type2"] is not None:
                        self.classes.add(row["Type2"])
                        actual_ys.append(row["Type2"])
                    ys_not_processed.append(actual_ys)
        ys = self.make_targets(ys_not_processed, sum1)
        return xs, ys
    
    def make_targets(self, not_processed, sum1=False):
        ys = []
        lcl = list(self.classes)
        for e in not_processed:
            target = [0 for _ in self.classes]
            for pktype in e:
                target[lcl.index(pktype)] = 1
            ys.append(target)
        if sum1:
            for i in range(len(ys)):
                sum_i = sum(ys[i])
                ys[i] = [e/float(sum_i) for e in ys[i]]
        return ys

    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return ["image"]  # None #self.headers
