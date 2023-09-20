import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', '..', 'src')))
######################################################
from mlexperiments.preprocessing.text.chunk_text import get_chunks
from mlexperiments.preprocessing.text.lemma_stem import LemmaStemmaText, Vectorize_Clustering


lemma_text = LemmaStemmaText(lang="es", blacklist_words=None, not_touch_words=None)

def process_doc(doc, lemmas_docs: list, ners: list, i_doc: int, rbd: int):
    FOLDER = "data/created_models/convivencia/"
    lemmed_doc = ""
    ners_doc = ""
    for chunk in get_chunks(doc, 100000, 50):
        #print(chunk)
        lemmas, _ners = lemma_text.lemmatize_docstring(chunk, to_get_ner=True)
        lemmed_doc += " ".join(lemmas)
        ners_doc += " ".join(_ners)
    print(len(lemmas))
    print(len(_ners))
    lemmas_docs.append(lemmed_doc)
    ners.append(ners_doc)
    with open(f"{FOLDER}lemmed/lemmed_{rbd}.txt", "w", encoding="utf-8") as fh:
        fh.write(lemmed_doc)
        fh.flush()
    with open(f"{FOLDER}ners/ners_{rbd}.txt", "w", encoding="utf-8") as fh:
        fh.write(ners_doc)
        fh.flush()
    print(f"finished {i_doc}")