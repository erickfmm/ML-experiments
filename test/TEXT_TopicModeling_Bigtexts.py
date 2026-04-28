from utils.topic_modeling_process_doc import process_doc

import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################

from mlexperiments.load_data.loader.text.get_txt_docs import LoadTXTsFolder
from mlexperiments.load_data.loader.text.ES_wikihow import LoadWikihowSpanish
from mlexperiments.load_data.loader.text.ES_Spanish_news import LoadSpanishNews

from mlexperiments.unsupervised.clustering.lda import LDA_model

#from multiprocessing import Process, Manager, current_process
import threading
from time import sleep


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _resolve_dataset(raw_dataset: str) -> str:
    aliases = {
        "news": "spanish_news",
    }
    return aliases.get(raw_dataset, raw_dataset)


def _get_metadata(loader, index: int):
    metadata = getattr(loader, "metadata", None)
    if metadata is None or index >= len(metadata):
        return index
    return metadata[index]


def _create_wordcloud(output_path: str, frequencies: dict, title: str, max_words: int = 100, width: int = 2500, height: int = 2500):
    if not frequencies:
        print(f"Skipping word cloud for '{title}': no named entities were extracted.")
        return

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    cloud = WordCloud(
                    background_color="white",
                    width=width,
                    height=height,
                    max_words=max_words)
    cloud.generate_from_frequencies(frequencies)
    plt.figure(figsize=(10, 10))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.savefig(output_path)
    plt.close()


if __name__ ==  '__main__':
    dataset = _resolve_dataset(os.environ.get("ML_DATASET", "wikihow").strip().lower())
    n_topics = _env_int("ML_N_TOPICS", 20, minimum=2)
    maxthreads = _env_int("ML_MAX_THREADS", 16, minimum=1)
    ##########################################
    FOLDER = f"data/created_models/{dataset}/"
    os.makedirs(FOLDER, exist_ok=True)
    os.makedirs(FOLDER+"ners/", exist_ok=True)
    os.makedirs(FOLDER+"lemmed/", exist_ok=True)

    ####################
    if dataset == "evaluacion":
        l = LoadTXTsFolder("data/train_data/NLP_ESP/education-reglamento-evaluacion/txt_files_evaluacion", which="evaluacion", to_catch_rbds=True)
        l.download()
        docs = l.get_data()
    elif dataset == "wikihow":
        l = LoadWikihowSpanish()
        l.download()
        docs, _ = l.get_X_Y()
    elif dataset == "spanish_news":
        l = LoadSpanishNews()
        l.download()
        docs, _ = l.get_X_Y()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")



    lemmas_docs = [""] * len(docs)
    ners = [""] * len(docs)

    n_docs = len(docs)
    jobs = []
    i_doc = 0
    while i_doc < n_docs:
        jobs = [job for job in jobs if job.is_alive()]
        if len(jobs) >= maxthreads:
            sleep(0.05)
            continue

        doc = docs[i_doc]
        metadata_value = _get_metadata(l, i_doc)
        print(f"initiating {i_doc + 1} of {n_docs}")
        p : threading.Thread = threading.Thread(target=process_doc, name=process_doc.__name__+str(i_doc), args=(
                                doc, lemmas_docs, ners, i_doc, metadata_value, FOLDER,))
        p.start()
        jobs.append(p)
        i_doc += 1

    for job in jobs:
        job.join()

    lemmas_docs = [doc for doc in lemmas_docs if doc.strip()]
    ners = [doc for doc in ners if doc.strip()]

    if not lemmas_docs:
        raise RuntimeError("No lemmatized documents were produced.")

    print(f"Processed {len(lemmas_docs)} documents with up to {maxthreads} worker threads.")


    ###############################

    wc = [token for doc in ners for token in doc.split() if token]
    d = {}
    for x in wc:
        if x in d:
            d[x] += 1
        else:
            d[x] = 1

    _create_wordcloud(f"{FOLDER}Named Entities.png", d, "Entidades")



    ####################################
    i_articulo = 0
    for wc in ners:
        d = {}
        for x in wc.split():
            if x in d:
                d[x] += 1
            else:
                d[x] = 1

        metadata_value = _get_metadata(l, i_articulo)
        _create_wordcloud(f"{FOLDER}Entidades_{metadata_value}.png", d, "Entidades#"+str(metadata_value))
        i_articulo +=1


    ##########################################
    if False:
        from bertopic import BERTopic


        topic_model = BERTopic(language="spanish", min_topic_size=5)
        success_bertopic = True
        try:
            topics, probs = topic_model.fit_transform(lemmas_docs)
        except:
            success_bertopic = False
        if all([x==-1 for x in topics]):
            success_bertopic = False


    ###########################################
    lda = LDA_model()
    lda.fit([x.split() for x in lemmas_docs], n_topics)
    lda.save(f"{FOLDER}lda.model", f"{FOLDER}lda")
    lda.create_cloud_of_topics(f"{FOLDER}lda")
    probs_topic, words_with_id, fh_all = lda.get_word_topic_probability()
    assignments = lda.get_doc_assignment([x.split() for x in lemmas_docs])
    print(lda.get_self_metrics())