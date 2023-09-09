# -*- coding: utf-8 -*-

import math
from mlexperiments.preprocessing.text.tf_idf.term_frequency import tf_binary

def idf_nt(term, documents):
    nt = 0
    for doc in documents:
        nt += tf_binary(term, doc)
    return nt

def idf(term, documents):
    N = len(documents)
    nt = idf_nt(term, documents)
    if nt != 0:
        return math.log(N / float(nt))
    else:
        return - math.log(nt / float(N))

def idf_unary(term, documents):
    return 1.0

def idf_smooth(term, documents):
    N = len(documents)
    return math.log( N / float(1.0 + idf_nt(term, documents)) )

def idf_max(term, documents):
    max_nt = 0
    all_words = set()
    for doc in documents:
        for w in doc:
            all_words.add(w)
    for word in all_words:
        word_nt = idf_nt(word, documents)
        if word_nt > max_nt:
            max_nt = word_nt
    return math.log( max_nt / float(1.0 + idf_nt(term, documents)))


def idf_probabilistic(term, documents):
    nt = idf_nt(term, documents)
    N = len(documents)
    return math.log( (N - nt) / float(nt))
