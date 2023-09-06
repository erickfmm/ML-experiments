import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..')))
######################################################

from load_data.loader.text.get_txt_docs import LoadTXTsFolder
from preprocessing.text.lemma_stem import LemmaStemmaText, Vectorize_Clustering
from preprocessing.text.chunk_text import get_chunks
from unsupervised.clustering.lda import LDA_model
##########################################
FOLDER = "data/created_models/pei/"
if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)

####################

l = LoadTXTsFolder(to_catch_rbds=True)
docs = l.get_all()
lemma_text = LemmaStemmaText(lang="es", blacklist_words=None, not_touch_words=None)

lemmas_docs = []
ners = []
i_doc = 0
n_docs = len(docs)
for doc in docs:
    lemmed_doc = ""
    ners_doc = ""
    print(f"processing {i_doc} of {n_docs}")
    for chunk in get_chunks(doc, 100000, 50):
        #print(chunk)
        lemmas, _ners = lemma_text.lemmatize_docstring(chunk, to_get_ner=True)
        lemmed_doc += " ".join(lemmas)
        ners_doc += " ".join(_ners)
    print(len(lemmas))
    print(len(_ners))
    lemmas_docs.append(lemmed_doc)
    with open(f"{FOLDER}lemmed_{l.metadata[i_doc]}.txt", "w", encoding="utf-8") as fh:
        fh.write(lemmed_doc)
        fh.flush()
    with open(f"{FOLDER}ners_{l.metadata[i_doc]}.txt", "w", encoding="utf-8") as fh:
        fh.write(ners_doc)
        fh.flush()
    ners.append(ners_doc)
    i_doc += 1


################################

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
plt.show()



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
    plt.show()
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