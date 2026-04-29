import math
import os
import random
import shutil
import sys
from os.path import dirname, join, abspath

sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################

import matplotlib

if not os.getenv("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mlexperiments.load_data.loader.basic.random_of_function import LoadRandom

import mlexperiments.supervised.learning_models_sklearn as learning_model
import mlexperiments.supervised.utils.confusion_matrix as confusion_matrix
import mlexperiments.unsupervised.correlation_matrix as correlation_matrix
import mlexperiments.unsupervised.distance as distance


OUTPUT_DIR = os.path.join("data", "created_models", "confusion_matrices")


def _get_env_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Invalid integer for {name}: {value!r}. Using default {default}.")
        return default


def _prepare_output_dir():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_matrix_plot(matrix, title, filename, value_format=None, cmap="viridis", x_label="Predicted", y_label="Actual"):
    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    image = ax.imshow(arr, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))

    threshold = (arr.max() + arr.min()) / 2.0 if arr.size else 0.0
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            value = arr[row, col]
            label = value_format(value) if value_format else f"{value:.3f}"
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color="#fff" if value >= threshold else "#111",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def print2d(data, max_rows=10, max_cols=10):
    for row_index, row in enumerate(data[:max_rows]):
        for value in row[:max_cols]:
            print(round(value, 3), end="\t")
        if len(row) > max_cols:
            print("...", end="")
        print()
    if len(data) > max_rows:
        print("...")
    print()


class RandomClass:
    def __init__(self, n_classes=2):
        self.n_classes = n_classes

    def __call__(self, x):
        return random.randint(0, self.n_classes - 1)


class TanhClass:
    def __init__(self, n_classes=2):
        self.n_classes = n_classes

    def __call__(self, x):
        tag = math.floor(math.tanh((math.fsum(x) / len(x))) * self.n_classes)
        return tag if tag >= 0 and tag < self.n_classes else self.n_classes - 1


if __name__ == "__main__":
    n_instances = _get_env_int("ML_INSTANCES", 1000)
    n_dimensions = _get_env_int("ML_DIMENSIONS", 5)
    n_classes = _get_env_int("ML_N_CLASSES", 3)
    cv_folds = _get_env_int("ML_CV_FOLDS", 10)

    tanh_classifier = TanhClass(n_classes)
    print("to load...")
    lrand = LoadRandom(
        num_instances=n_instances,
        num_dimensions=n_dimensions,
        min_value=0,
        max_value=1,
        fn=tanh_classifier,
    )
    x, y = lrand.get_X_Y()

    print("some calculations")
    rs = correlation_matrix.correlation_matrix(x)
    print("correlations")
    print2d(rs)

    dists = distance.distance_matrix(x)
    print("distances")
    print2d(dists)

    ms, bs = correlation_matrix.coefficients_matrix(x)
    print("ms")
    print2d(ms)
    print("bs")
    print2d(bs)

    print()
    print("to learn....")
    print()
    model, clf = learning_model.train_mlp(x, y)
    scores = learning_model.get_k_scores(clf, x, y, cv_folds)
    print("scores: ", scores)
    print("mean scores: ", sum(scores) / float(len(scores)))

    ypreds = model.predict(x)
    conf = confusion_matrix.confusion_matrix(y, ypreds)
    print("confusion matrix")
    print2d(conf)

    conf_norm = confusion_matrix.normalize(conf)
    print("normalized")
    print2d(conf_norm)

    preview_size = min(80, len(dists))
    dists_preview = [row[:preview_size] for row in dists[:preview_size]]

    _prepare_output_dir()
    _save_matrix_plot(rs, "Correlation Matrix", "01_correlation_matrix.png", value_format=lambda v: f"{v:.2f}", cmap="magma", x_label="Dimension", y_label="Dimension")
    _save_matrix_plot(dists_preview, f"Distance Matrix (first {preview_size} samples)", "02_distance_matrix.png", value_format=lambda v: f"{v:.2f}", cmap="plasma", x_label="Sample", y_label="Sample")
    _save_matrix_plot(ms, "Linear Coefficients (m)", "03_coefficients_m.png", value_format=lambda v: f"{v:.2f}", cmap="cividis", x_label="Dimension", y_label="Dimension")
    _save_matrix_plot(bs, "Linear Coefficients (b)", "04_coefficients_b.png", value_format=lambda v: f"{v:.2f}", cmap="viridis", x_label="Dimension", y_label="Dimension")
    _save_matrix_plot(conf, "Confusion Matrix", "05_confusion_matrix.png", value_format=lambda v: str(int(round(v))), cmap="Blues")
    _save_matrix_plot(conf_norm, "Normalized Confusion Matrix", "06_confusion_matrix_normalized.png", value_format=lambda v: f"{v:.2f}", cmap="Greens")