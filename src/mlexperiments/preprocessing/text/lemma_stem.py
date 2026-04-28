from typing import List, Tuple
import warnings
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
import numpy as np
import spacy
import matplotlib.pyplot as plt #optional

class LemmaStemmaText:
    _LANGUAGE_CONFIG = {
        "es": {
            "default_spacy_model": "es_core_news_sm",
            "stemmer_language": "spanish",
            "stopwords_language": "spanish",
        },
        "en": {
            "default_spacy_model": "en_core_web_sm",
            "stemmer_language": "english",
            "stopwords_language": "english",
        },
    }

    def __init__(self, lang="es",
                 blacklist_words = ["ser"],
                 not_touch_words = ["sige", "pme", "deprov", "sigpa", "sac", "simce", "faep", "sep", "superintendencia"],
                 allow_postags = set(['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']),
                 excluded_terms_in_ner = None,
                 spacy_model: str | None = None):
        if lang not in self._LANGUAGE_CONFIG:
            raise ValueError(f"Unsupported language: {lang}")

        cfg = self._LANGUAGE_CONFIG[lang]
        if spacy_model is None:
            self.model_name = cfg["default_spacy_model"]
        else:
            self.model_name = spacy_model.strip()
            if not self.model_name:
                raise ValueError("spaCy model name cannot be empty.")
        self.nlp = self._load_spacy_pipeline(self.model_name)
        self.stemmer = SnowballStemmer(cfg["stemmer_language"])
        self.stw = set(self._load_stopwords(cfg["stopwords_language"]))
        self.blacklist = set(blacklist_words) if blacklist_words is not None else None
        self.not_touch = set(not_touch_words) if not_touch_words is not None else set()
        self.allowed_postags = allow_postags if self._supports_pos_tags() else None
        self.excluded_terms = set(excluded_terms_in_ner) if excluded_terms_in_ner is not None else None
        self.has_lemmatizer = self.nlp.has_pipe("lemmatizer")
        self.has_ner = self.nlp.has_pipe("ner")

    @staticmethod
    def _load_stopwords(language: str):
        try:
            return stopwords.words(language)
        except LookupError:
            try:
                nltk.download("stopwords", quiet=True)
                return stopwords.words(language)
            except LookupError:
                warnings.warn(
                    f"NLTK stopwords for '{language}' are unavailable; continuing without stopword filtering.",
                    RuntimeWarning,
                )
                return []

    @staticmethod
    def _load_spacy_pipeline(model_name: str):
        try:
            return spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed or cannot be loaded. "
                f"Install that exact model and rerun the pipeline. Original error: {exc}"
            ) from exc

    def _supports_pos_tags(self) -> bool:
        return any(self.nlp.has_pipe(pipe_name) for pipe_name in ("tagger", "morphologizer", "parser"))

    def string_to_doc(self, docstring):
        pass

    def lemmatize_docstring(self, docstring: str, to_get_ner=False) -> Tuple[List[str],List[str]]:
        doc_nlp = self.nlp(docstring.strip())
        lemmas = []
        named_entities = []
        for tok in doc_nlp:
            token_text = tok.text.lower()
            if tok.is_space or tok.is_punct:
                continue
            if (self.stw is None or token_text not in self.stw) and (self.blacklist is None or token_text not in self.blacklist) and (self.allowed_postags is None or tok.pos_ in self.allowed_postags):
                if token_text in self.not_touch:
                    lemmas.append(tok.text.upper())
                else:
                    lemma = tok.lemma_.lower() if self.has_lemmatizer and tok.lemma_ else token_text
                    lemmas.append(lemma)
                if to_get_ner and self.has_ner and tok.ent_type_ is not None and tok.ent_type_ != "" and tok.ent_type_ != "PER" and (self.excluded_terms is None or token_text not in self.excluded_terms):
                    named_entities.append(token_text)
        return lemmas, named_entities
    
    def stem_doc(self, doc: List[str]):
        doc_final = []
        for word in doc:
            if word not in self.stw and (self.blacklist is None or word not in self.blacklist):
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