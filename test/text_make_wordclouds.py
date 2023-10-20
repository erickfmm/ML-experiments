import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################





NERS_FOLDER = "data/created_models/pei/ners/"
FOLDER = "data/created_models/pei/imgs/"

import os

if not os.path.exists(FOLDER):
  os.mkdir(FOLDER)
ners = []
metadata = []
for filename in os.listdir(NERS_FOLDER):
  metadata.append(filename.replace(".txt", "").replace("ners_", ""))
  with open(os.path.join(NERS_FOLDER, filename), "r", encoding="utf8") as fobj:
    ners.append(fobj.read().split(" "))





###############################

wc = [x2 for x in ners for x2 in x]
d = {}
for x in wc:
    if x in d:
        d[x] += 1
    else:
        d[x] = 1

from wordcloud import WordCloud
import matplotlib.pyplot as plt
cloud = WordCloud(
                background_color="white",
                width=2500,
                height=2500,
                max_words=100)
cloud.generate_from_frequencies(d)
plt.imshow(cloud, interpolation="bilinear")
plt.axis("off")
plt.title("Entidades")
plt.savefig(f"{FOLDER}Named Entities.png")
#plt.show()


###################################
i_articulo = 0
for wc in ners:
    d = {}
    for x in wc:
        if x in d:
            d[x] += 1
        else:
            d[x] = 1


    cloud = WordCloud(
                    background_color="white",
                    width=2500,
                    height=2500,
                    max_words=100)
    cloud.generate_from_frequencies(d)
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Entidades#"+str(metadata[i_articulo]))
    plt.savefig(f"{FOLDER}Entidades_{metadata[i_articulo]}.png")
    #plt.show()
    print(f"Done {i_articulo} of {len(ners)}")
    i_articulo +=1