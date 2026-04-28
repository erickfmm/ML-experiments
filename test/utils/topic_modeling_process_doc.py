import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', '..', 'src')))
######################################################
from mlexperiments.preprocessing.text.chunk_text import get_chunks
from mlexperiments.preprocessing.text.lemma_stem import LemmaStemmaText, Vectorize_Clustering


TOPIC_SPACY_MODEL = os.environ.get("ML_SPACY_MODEL", "es_core_news_sm").strip()
lemma_text = LemmaStemmaText(
    lang="es",
    blacklist_words=None,
    not_touch_words=None,
    spacy_model=TOPIC_SPACY_MODEL,
)

def process_doc(doc, lemmas_docs: list, ners: list, i_doc: int, rbd: int, folder: str):
    lemmed_chunks = []
    ners_chunks = []
    last_lemmas_count = 0
    last_ners_count = 0
    for chunk in get_chunks(doc, 100000, 50):
        #print(chunk)
        lemmas, _ners = lemma_text.lemmatize_docstring(chunk, to_get_ner=True)
        last_lemmas_count = len(lemmas)
        last_ners_count = len(_ners)
        if lemmas:
            lemmed_chunks.append(" ".join(lemmas))
        if _ners:
            ners_chunks.append(" ".join(_ners))
    lemmed_doc = " ".join(lemmed_chunks).strip()
    ners_doc = " ".join(ners_chunks).strip()
    print(last_lemmas_count)
    print(last_ners_count)
    lemmas_docs[i_doc] = lemmed_doc
    ners[i_doc] = ners_doc
    with open(f"{folder}lemmed/lemmed_{rbd}.txt", "w", encoding="utf-8") as fh:
        fh.write(lemmed_doc)
        fh.flush()
    with open(f"{folder}ners/ners_{rbd}.txt", "w", encoding="utf-8") as fh:
        fh.write(ners_doc)
        fh.flush()
    print(f"finished {i_doc}")