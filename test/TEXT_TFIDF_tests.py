import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################


import mlexperiments.preprocessing.text.tf_idf.term_frequency as tf
import mlexperiments.preprocessing.text.tf_idf.inverse_document_frequency as idf
import mlexperiments.preprocessing.text.tf_idf.tf_idf_document as tf_idf_doc_funcs
import mlexperiments.preprocessing.text.tf_idf.tf_idf_query as tf_idf_query_funcs

# text taken from sanish newsparlament dataset from wmt11, first 10 lines
ss = []
ss.append("Musharraf's Last Act?")
ss.append("Desperate to hold onto power, Pervez Musharraf has discarded Pakistan's constitutional framework and declared a state of emergency.")
ss.append("His goal?")
ss.append("To stifle the independent judiciary and free media.")
ss.append("Artfully, though shamelessly, he has tried to sell this action as an effort to bring about stability and help fight the war on terror more effectively.")
ss.append("Nothing could be further from the truth.")
ss.append("If Pakistan's history is any indicator, his decision to impose martial law may prove to be the proverbial straw that breaks the camel's back.")
ss.append('General Musharraf appeared on the national scene on October 12, 1999, when he ousted an elected government and announced an ambitious "nation-building" project.')
ss.append("Many Pakistanis, disillusioned with Pakistan's political class, remained mute, thinking that he might deliver.")
ss.append("The September 11, 2001, terrorist attacks on America brought Musharraf into the international limelight as he agreed to ditch the Taliban and support the United States-led war on terror.")

import re
docs = []
words = set()
for s in ss:
    doc = []
    for w in re.findall(r"\w+", s):
        if len(w) > 1:
            doc.append(w)
            words.add(w)
    docs.append(doc)

import matplotlib.pyplot as plt

tf_points = {
    "bin": [],
    "raw": [],
    "rel": [],
    "log": [],
    "double": []
}
idf_point = {
    "idf": [],
    "smooth": [],
    "max": [],
    "prob": []
}
tf_idf_doc = [[], [], []]
tf_idf_query = [[], [], []]

i = 0
for doc in docs:
    tf_points["bin"].append([])
    tf_points["raw"].append([])
    tf_points["rel"].append([])
    tf_points["log"].append([])
    tf_points["double"].append([]) 
    tf_idf_doc[0].append([])
    tf_idf_doc[1].append([])
    tf_idf_doc[2].append([])
    tf_idf_query[0].append([])
    tf_idf_query[1].append([])
    tf_idf_query[2].append([])
    for w in doc:
        tf_points["bin"][i].append(tf.tf_binary(w, doc))
        tf_points["raw"][i].append(tf.tf_raw(w, doc))
        tf_points["rel"][i].append(tf.tf_relative(w, doc))
        tf_points["log"][i].append(tf.tf_lognorm(w, doc))
        tf_points["double"][i].append(tf.tf_doublenormk(w, doc))
        tf_idf_doc[0][i].append(tf_idf_doc_funcs.tf_idf_document1(w, doc, docs))
        tf_idf_doc[1][i].append(tf_idf_doc_funcs.tf_idf_document2(w, doc, docs))
        tf_idf_doc[2][i].append(tf_idf_doc_funcs.tf_idf_document3(w, doc, docs))
        tf_idf_query[0][i].append(tf_idf_query_funcs.tf_idf_query1(w, doc, docs))
        tf_idf_query[1][i].append(tf_idf_query_funcs.tf_idf_query2(w, doc, docs))
        tf_idf_query[2][i].append(tf_idf_query_funcs.tf_idf_query3(w, doc, docs))
    i += 1


for w in words:
    idf_point["idf"].append(idf.idf(w, docs))
    idf_point["smooth"].append(idf.idf_smooth(w, docs))
    idf_point["max"].append(idf.idf_max(w, docs))
    idf_point["prob"].append(idf.idf_probabilistic(w, docs))

# import pprint
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(tf_points["bin"])


plot_tf = True  # False
plot_tf_idf_doc = True
plot_tf_idf_query = True
plot_idf = True  # True

mode_plot = "."  # "-"

if plot_tf:
    fig = plt.figure()
    plt.title("Term Frequency")
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # ax = fig.add_subplot(2, 2, 1)
    # for i in range(len(docs)):
    #    ax.plot(tf_points["bin"][i])

    ax = fig.add_subplot(2, 2, 1)
    ax.title.set_text("raw")
    for i in range(len(docs)):
        ax.plot(tf_points["raw"][i], mode_plot)

    ax = fig.add_subplot(2, 2, 2)
    ax.title.set_text("relative")
    for i in range(len(docs)):
        ax.plot(tf_points["rel"][i], mode_plot)

    ax = fig.add_subplot(2, 2, 3)
    ax.title.set_text("log norm")
    for i in range(len(docs)):
        ax.plot(tf_points["log"][i], mode_plot)
    # plt.show()

    ax = fig.add_subplot(2, 2, 4)
    ax.title.set_text("double norm")
    for i in range(len(docs)):
        ax.plot(tf_points["double"][i], mode_plot)
    plt.show()

if plot_idf:
    fig = plt.figure()
    plt.title("Inverse Document Frequency")
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(2, 2, 1)
    ax.title.set_text("idf")
    ax.plot(idf_point["idf"], mode_plot)

    ax = fig.add_subplot(2, 2, 2)
    ax.title.set_text("smooth")
    ax.plot(idf_point["smooth"], mode_plot)

    ax = fig.add_subplot(2, 2, 3)
    ax.title.set_text("max")
    ax.plot(idf_point["max"], mode_plot)
    
    ax = fig.add_subplot(2, 2, 4)
    ax.title.set_text("prob")
    ax.plot(idf_point["prob"], mode_plot)
    plt.show()


if plot_tf_idf_doc:
    fig = plt.figure()
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(2, 2, 1)
    ax.title.set_text("1")
    for i in range(len(docs)):
        ax.plot(tf_idf_doc[0][i], mode_plot)

    ax = fig.add_subplot(2, 2, 2)
    ax.title.set_text("2")
    for i in range(len(docs)):
        ax.plot(tf_idf_doc[1][i], mode_plot)

    ax = fig.add_subplot(2, 2, 3)
    ax.title.set_text("3")
    for i in range(len(docs)):
        ax.plot(tf_idf_doc[2][i], mode_plot)
    plt.show()

if plot_tf_idf_query:
    fig = plt.figure()
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(2, 2, 1)
    ax.title.set_text("1")
    for i in range(len(docs)):
        ax.plot(tf_idf_query[0][i], mode_plot)

    ax = fig.add_subplot(2, 2, 2)
    ax.title.set_text("2")
    for i in range(len(docs)):
        ax.plot(tf_idf_query[1][i], mode_plot)

    ax = fig.add_subplot(2, 2, 3)
    ax.title.set_text("3")
    for i in range(len(docs)):
        ax.plot(tf_idf_query[2][i], mode_plot)
    plt.show()
