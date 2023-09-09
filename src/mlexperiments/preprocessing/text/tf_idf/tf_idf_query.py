# -*- coding: utf-8 -*-

import math
from mlexperiments.preprocessing.text.tf_idf.term_frequency import tf_raw, tf_doublenormk
from mlexperiments.preprocessing.text.tf_idf.inverse_document_frequency import idf_nt, idf

def tf_idf_query1(term, document, documents):
    tf = tf_doublenormk(term, document)
    idf_ = idf(term, documents)
    return tf * idf_

def tf_idf_query2(term, document, documents):
    N = len(documents)
    return math.log( 1.0 + (N / float(idf_nt(term, documents))))

def tf_idf_query3(term, document, documents):
    tf = 1.0 + math.log(tf_raw(term, document))
    idf_ = idf(term, documents)
    return tf * idf_
