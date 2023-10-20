import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################




NERS_FOLDER = "data/created_models/pei/ners/"
FOLDER = "data/created_models/pei/"

import os

if not os.path.exists(FOLDER):
  os.mkdir(FOLDER)
ners = []
metadata = []
for filename in os.listdir(NERS_FOLDER):
  metadata.append(filename.replace(".txt", "").replace("ners_", ""))
  with open(os.path.join(NERS_FOLDER, filename), "r", encoding="utf8") as fobj:
    ners.append(fobj.read().split(" "))

##############################

import mlexperiments.preprocessing.text.tf_idf.tf_idf_document as tfidf

array_form = []
i_document = 0
data = {}
for document in ners:
  arr = []
  for word in document:
    arr.append(tfidf.tf_idf_document1(word, document, ners))
  array_form.append(arr)
  data[int(metadata[i_document])] = arr
  print(f"Done {i_document} of {len(ners)}")
  i_document += 1

import pickle
with open(f"{FOLDER}ners_tf_idf.pkl", "wb") as fobj:
    pickle.dump(array_form, fobj)
