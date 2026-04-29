import os
import re
import shutil
import sys
from os.path import dirname, join, abspath

sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################

import matplotlib

if not os.getenv("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

import mlexperiments.preprocessing.text.tf_idf.inverse_document_frequency as idf
import mlexperiments.preprocessing.text.tf_idf.term_frequency as tf
import mlexperiments.preprocessing.text.tf_idf.tf_idf_document as tf_idf_doc_funcs
import mlexperiments.preprocessing.text.tf_idf.tf_idf_query as tf_idf_query_funcs


OUTPUT_DIR = os.path.join("data", "created_models", "tfidf_tests")


def _get_env_bool(name, default=True):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _prepare_output_dir():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_figure(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


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

docs = []
words = set()
for sentence in ss:
    doc = []
    for word in re.findall(r"\w+", sentence):
        if len(word) > 1:
            doc.append(word)
            words.add(word)
    docs.append(doc)

tf_points = {
    "bin": [],
    "raw": [],
    "rel": [],
    "log": [],
    "double": [],
}
idf_point = {
    "idf": [],
    "smooth": [],
    "max": [],
    "prob": [],
}
tf_idf_doc = [[], [], []]
tf_idf_query = [[], [], []]

for i, doc in enumerate(docs):
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
    for word in doc:
        tf_points["bin"][i].append(tf.tf_binary(word, doc))
        tf_points["raw"][i].append(tf.tf_raw(word, doc))
        tf_points["rel"][i].append(tf.tf_relative(word, doc))
        tf_points["log"][i].append(tf.tf_lognorm(word, doc))
        tf_points["double"][i].append(tf.tf_doublenormk(word, doc))
        tf_idf_doc[0][i].append(tf_idf_doc_funcs.tf_idf_document1(word, doc, docs))
        tf_idf_doc[1][i].append(tf_idf_doc_funcs.tf_idf_document2(word, doc, docs))
        tf_idf_doc[2][i].append(tf_idf_doc_funcs.tf_idf_document3(word, doc, docs))
        tf_idf_query[0][i].append(tf_idf_query_funcs.tf_idf_query1(word, doc, docs))
        tf_idf_query[1][i].append(tf_idf_query_funcs.tf_idf_query2(word, doc, docs))
        tf_idf_query[2][i].append(tf_idf_query_funcs.tf_idf_query3(word, doc, docs))

for word in words:
    idf_point["idf"].append(idf.idf(word, docs))
    idf_point["smooth"].append(idf.idf_smooth(word, docs))
    idf_point["max"].append(idf.idf_max(word, docs))
    idf_point["prob"].append(idf.idf_probabilistic(word, docs))

plot_tf = _get_env_bool("ML_PLOT_TF", True)
plot_tf_idf_doc = _get_env_bool("ML_PLOT_TFIDF_DOC", True)
plot_tf_idf_query = _get_env_bool("ML_PLOT_TFIDF_QUERY", True)
plot_idf = _get_env_bool("ML_PLOT_IDF", True)

running_from_web = any(
    key in os.environ for key in ("ML_PLOT_TF", "ML_PLOT_IDF", "ML_PLOT_TFIDF_DOC", "ML_PLOT_TFIDF_QUERY")
)
show_plot = _get_env_bool("ML_SHOW_PLOT", default=(not running_from_web and bool(os.getenv("DISPLAY"))))
mode_plot = "."

_prepare_output_dir()

if plot_tf:
    fig = plt.figure()
    plt.title("Term Frequency")

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

    ax = fig.add_subplot(2, 2, 4)
    ax.title.set_text("double norm")
    for i in range(len(docs)):
        ax.plot(tf_points["double"][i], mode_plot)

    _save_figure(fig, "01_term_frequency.png")

if plot_idf:
    fig = plt.figure()
    plt.title("Inverse Document Frequency")

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

    _save_figure(fig, "02_inverse_document_frequency.png")

if plot_tf_idf_doc:
    fig = plt.figure()

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

    _save_figure(fig, "03_tfidf_document.png")

if plot_tf_idf_query:
    fig = plt.figure()

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

    _save_figure(fig, "04_tfidf_query.png")

if show_plot:
    plt.show()
else:
    plt.close("all")
