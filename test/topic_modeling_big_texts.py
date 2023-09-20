from utils.topic_modeling_process_doc import process_doc

import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################

from mlexperiments.load_data.loader.text.get_txt_docs import LoadTXTsFolder


from mlexperiments.unsupervised.clustering.lda import LDA_model

#from multiprocessing import Process, Manager, current_process
import threading
from time import sleep


if __name__ ==  '__main__':
    ##########################################
    FOLDER = "data/created_models/convivencia/"
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)

    if not os.path.exists(FOLDER+"ners/"):
        os.mkdir(FOLDER+"ners/")
    if not os.path.exists(FOLDER+"lemmed/"):
        os.mkdir(FOLDER+"lemmed/")

    ####################

    l = LoadTXTsFolder("data/train_data/NLP_ESP/txt_files_convivencia", to_catch_rbds=True)
    docs = l.get_data()





    lemmas_docs = []
    ners = []

    n_docs = len(docs)
    jobs = []
    #for doc in docs:
    #manager = Manager()
    #lemmas_docs = manager.list()
    #ners = manager.list()
    #print(f"processing {i_doc} of {n_docs}")
    
    #jobs.append(p)
    #process_doc(doc, lemmas_docs, ners)
    #i_doc += 1
    #for p in jobs:
    #    print(f"{p.name} iniciando...")
    #    p.start()
    ################################
    maxthreads = 16
    i_doc = 0
    while True:
        if threading.active_count() <= maxthreads:
            doc = docs[i_doc]
            print(f"initiating {i_doc} of {n_docs}")
            p : threading.Thread = threading.Thread(target=process_doc, name=process_doc.__name__+str(i_doc), args=(
                                doc, lemmas_docs, ners, i_doc, l.metadata[i_doc],))
            p.start()
            i_doc += 1
        #sleep(1)


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



    ####################################
    #wc = [x2 for x in ners for x2 in x]
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
        plt.title("Entidades#"+str(l.metadata[i_articulo]))
        plt.savefig(f"{FOLDER}Entidades_{l.metadata[i_articulo]}.png")
        #plt.show()
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
    lda.fit([x.split(" ") for x in lemmas_docs], 20)
    lda.save(f"{FOLDER}lda.model", f"{FOLDER}lda")
    lda.create_cloud_of_topics(f"{FOLDER}lda")
    probs_topic, words_with_id, fh_all = lda.get_word_topic_probability()
    assignments = lda.get_doc_assignment([x.split(" ") for x in lemmas_docs])
    print(lda.get_self_metrics())