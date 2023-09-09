from typing import List, Tuple
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
import numpy as np
import spacy
import matplotlib.pyplot as plt #optional

class LemmaStemmaText:
    def __init__(self, lang="es",
                 blacklist_words = ["ser"],
                 not_touch_words = ["sige", "pme", "deprov", "sigpa", "sac", "simce", "faep", "sep", "superintendencia"],
                 allow_postags = set(['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']),
                 excluded_terms_in_ner = None):
        if lang == "es":
            self.nlp = spacy.load('es_core_news_lg')
            self.stemmer = SnowballStemmer('spanish')
            nltk.download('stopwords')
            self.stw = stopwords.words('spanish')
        if lang == "en":
            self.nlp = spacy.load('en_core_news_lg')
            self.stemmer = SnowballStemmer('english')
            nltk.download('stopwords')
            self.stw = stopwords.words('english')
        self.blacklist = blacklist_words
        self.not_touch = not_touch_words
        self.allowed_postags = allow_postags
        self.excluded_terms = excluded_terms_in_ner

    def string_to_doc(self, docstring):
        pass

    def lemmatize_docstring(self, docstring: str, to_get_ner=False) -> Tuple[List[str],List[str]]:
        doc_nlp = self.nlp(docstring.strip())
        lemmas = []
        named_entities = []
        for tok in doc_nlp:
            if (self.stw is None or tok.text not in self.stw) and (self.blacklist is None or tok.text not in self.blacklist) and (self.allowed_postags is None or tok.pos_ in self.allowed_postags):
                if self.not_touch is not None and tok.text.lower() in self.not_touch:
                    lemmas.append(tok.text.upper())
                else:
                    lemmas.append(tok.lemma_.lower())
                if to_get_ner and tok.ent_type_ is not None and tok.ent_type_ != "" and tok.ent_type_ != "PER" and (self.excluded_terms is None or tok.text.lower() not in self.excluded_terms):
                    named_entities.append(tok.text.lower())
        return lemmas, named_entities
    
    def stem_doc(self, doc: List[str]):
        doc_final = []
        for word in doc:
            if word not in self.stw and word not in self.blacklist:
                if word in self.not_touch:
                    doc_final.append(word)
                    continue
                doc_final.append(self.stemmer.stem(word))
        return doc_final
    

    def get_distances_similarities(full_documents : List[List[str]], to_show : bool = False) -> Tuple[List[float], List[float]]:
        distances = []
        similarities = []
        i = 0
        for doc in full_documents:
            distances.append([])
            similarities.append([])
            for doc2 in full_documents:
                #if doc != doc2:
                    distance = np.linalg.norm(doc.vector - doc2.vector)
                    similarity = doc.similarity(doc2)
                    distances[i].append(distance)
                    similarities[i].append(similarity)
            i += 1
        if to_show:
            plt.imshow(distances)
            plt.show()
            plt.imshow(similarities)
            plt.show()
        return distances, similarities
    


def count_words(docs:  List[List[str]]):
    words = set()
    wc = dict()
    for doc in docs:
        for w in doc:
            words.add(w)
            if w in wc:
                wc[w] += 1
            else:
                wc[w] = 1
    return wc, words

class Vectorize_Clustering:
    def __init__(self) -> None:
        pass

    @staticmethod
    def concat_list_of_str(l : List[str]) -> str:
        l2 = ''
        for x in l:
            l2 += str(x)+" "
        return l2.strip()

    from sklearn.preprocessing import normalize
    @staticmethod
    def vectorize(text : str, nlp : spacy.language.Language) -> List[float]:
        # Get the SpaCy vector -- turning off other processing to speed things up
        return nlp(text, disable=['parser', 'tagger', 'ner']).vector
    #X = normalize(np.stack(vectorize(t.text) for t in full_documents))
    
    @staticmethod
    def get_X(docs : List[List[str]]):
        X = Vectorize_Clustering.normalize(np.stack(Vectorize_Clustering.vectorize(Vectorize_Clustering.concat_list_of_str(t)) for t in docs))
        return X

    @staticmethod
    def plot_groups(X, y, groups):
        for group in groups:
            plt.scatter(X[y == group, 0], X[y == group, 1], label=group, alpha=0.4)
        plt.legend()
        plt.show()

    
    
    @staticmethod
    def to_2d(X):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)
        #print("X2 shape is {}".format(X2.shape))
        return X2

    @staticmethod
    def cluster(X, n_clusters : int, random_seed=42):
        #CLUSTERS = 3
        # First we fit the model...
        from sklearn.cluster import KMeans
        k_means = KMeans(n_clusters=n_clusters, random_state=random_seed)
        k_means.fit(X)

        yhat = k_means.predict(X)

        # Let's take a look at the distribution across classes
        plt.hist(yhat, bins=range(n_clusters))
        plt.show()
        Vectorize_Clustering.plot_groups(Vectorize_Clustering.to_2d(X), yhat, np.arange(n_clusters))

        #for i in range(len(docs)):
        #    print(i, " - ", yhat[i], " --- ", docs[i])