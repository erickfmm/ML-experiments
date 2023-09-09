# -*- coding: utf-8 -*-


def tf_binary(term, document):
    if term in document:
        return 1
    else:
        return 0

def tf_raw(term, document):
    count = 0
    for w in document:
        if w == term:
            count +=1
    return count

def tf_relative(term, document):
    return tf_raw(term, document)/float(len(document))

import math
def tf_lognorm(term, document):
    return math.log(1.0 + tf_raw(term, document))

def tf_doublenormk(term, document, k=0.5):
    words_in_doc = set(document)
    max_tf = 0
    for w in words_in_doc:
        count = tf_raw(w, document)
        if count > max_tf:
            max_tf = count
    return k + ( (1.0 - k) * (tf_raw(term, document)/float(max_tf)) )
