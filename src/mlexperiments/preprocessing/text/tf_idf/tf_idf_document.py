# -*- coding: utf-8 -*-

import math
from mlexperiments.preprocessing.text.tf_idf.term_frequency import tf_raw
from mlexperiments.preprocessing.text.tf_idf.inverse_document_frequency import idf_nt, idf


def tf_idf_document1(term, document, documents):
    N = len(documents)
    tf = tf_raw(term, document)
    idf_ = math.log(N / float(idf_nt(term, documents)))
    return tf * idf_

def tf_idf_document2(term, document, documents):
    return 1.0 + math.log(tf_raw(term, document))

def tf_idf_document3(term, document, documents):
    tf = 1.0 + math.log(tf_raw(term, document))
    idf_ = idf(term, documents)
    return tf * idf_
