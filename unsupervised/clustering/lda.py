
import gensim.corpora as corpora# Create Dictionary

#print(corpus[:1][0][:30])

import gensim
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pickle 
import pyLDAvis# Visualize the topics
from io import StringIO

from typing import List

import matplotlib.pyplot as plt

from wordcloud import WordCloud

class LDA_model:
    def __init__(self) -> None:
        pass

    def fit(self, docs: List[List[str]], num_topics: int):
        self.id2word = corpora.Dictionary(docs)# Create Corpus
        self.texts = docs# Term Document Frequency
        self.corpus = [self.id2word.doc2bow(text) for text in self.texts]
        self.lda_model = gensim.models.LdaModel(corpus=self.corpus,
                                        id2word=self.id2word,
                                        num_topics=num_topics)
        return self.lda_model
    
    def get_self_metrics(self):
        perplexity = self.lda_model.log_perplexity(self.corpus)
        coherence_model_lda = CoherenceModel(model=self.lda_model, texts=self.texts, dictionary=self.id2word, coherence='c_v', processes=1)
        return perplexity, coherence_model_lda

    def get_word_topic_probability(self):
        ids_ = []
        words = []
        words_with_id = []
        fh_all = StringIO("")
        topics = self.lda_model.get_topics()
        for id_ in self.lda_model.id2word:
            ids_.append(id_)
            words.append(self.lda_model.id2word[id_])
            words_with_id.append({
                "id": id_,
                "word": self.lda_model.id2word[id_]
            })
        for i in range(len(ids_)):
            fh_all.write(str(ids_[i])+";")
        fh_all.write("\n")
        fh_all.flush()
        for i in range(len(ids_)):
            fh_all.write(str(words[i])+";")
        fh_all.write("\n")
        fh_all.flush()
        probs_topic = []
        i_topic = 0
        for t in topics:
            probs_topic.append([])
            i_prob = 0
            for prob in t:
                fh_all.write(str(prob).replace(".",",")+";")
                probs_topic.append({
                    "word": words_with_id[i_prob],
                    "prob": prob
                })
                i_prob += 1
            fh_all.write("\n")
            fh_all.flush()
            i_topic += 1
        fh_all.flush()
        #return fh_all.read()
        return probs_topic, words_with_id

    def get_doc_assignment(self, docs: List[List[str]]):
        assignments = []
        for doc in docs:
            bow_text = self.lda_model.id2word.doc2bow(doc)
            doc_topics, word_topics, phi_values  = self.lda_model.get_document_topics(bow_text, per_word_topics=True)
            max_prob = 0
            top = 0
            all_probs = []
            for topi in doc_topics:
                all_probs.append({
                    "topic": topi[0],
                    "prob": topi[1]
                })
                if topi[1] > max_prob:
                    top = topi[0]
                    max_prob = topi[1]
            assignments.append({
                "document": " ".join(doc),
                "topic": top,
                "topic_probability": max_prob,
                "all_probabilities": all_probs})
        return assignments
    
    def create_cloud_of_topics(self, filename):
        i_topic = 0
        topics = self.lda_model.get_topics()
        words = []
        PROBABILITY_MULTIPLIER = 10000.0
        for id_ in self.lda_model.id2word:
            words.append(self.lda_model.id2word[id_])
        for t in topics:
            d = {}
            i = 0
            for prob in t:
                d[str(words[i])] = int(prob*PROBABILITY_MULTIPLIER)
                i +=1
            cloud = WordCloud(
                background_color="white",
                width=2500,
                height=2500,
                max_words=20)
            print("to freq")
            cloud.generate_from_frequencies(d)
            plt.imshow(cloud, interpolation="bilinear")
            plt.axis("off")
            plt.title("Topico "+str(i_topic+1))
            plt.savefig(f"{filename}--{i_topic}.png")
            i_topic += 1

    def save(self, LDAvis_data_filepath, LDAvis_html_filepath):
        LDAvis_prepared = gensimvis.prepare(self.lda_model, self.corpus, self.id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
        pyLDAvis.save_html(LDAvis_prepared, f'{LDAvis_html_filepath}'+'.html')
