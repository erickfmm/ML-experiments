import os
import pprint
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

import mlexperiments.unsupervised.clustering.cluster_sklearn as clustering_sk
import mlexperiments.unsupervised.clustering.utils.initial_assignments as ias
import mlexperiments.unsupervised.clustering.utils.metrics as clustering_metrics
from mlexperiments.unsupervised.clustering.kmeans import KMeans


OUTPUT_DIR = os.path.join("data", "created_models", "clustering_benchmarks")


def _get_env_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Invalid integer for {name}: {value!r}. Using default {default}.")
        return default


def _get_env_bool(name, default=False):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _slugify(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def _prepare_output_dir():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _build_dataset(samples: int, seed: int = 844):
    samples = max(4, samples)
    rng = np.random.RandomState(seed)
    base = samples // 4
    extra = samples % 4
    sizes = [base + (1 if idx < extra else 0) for idx in range(4)]

    clusters = [
        rng.normal(5, 2, (sizes[0], 2)),
        rng.normal(15, 3, (sizes[1], 2)),
        rng.multivariate_normal([17, 3], [[1, 0], [0, 1]], sizes[2]),
        rng.multivariate_normal([2, 16], [[1, 0], [0, 1]], sizes[3]),
    ]
    labels = []
    for cluster_index, cluster_data in enumerate(clusters):
        labels.extend([cluster_index] * len(cluster_data))
    return np.concatenate(clusters), labels


def _save_cluster_plot(dataset, assignments, title, filename):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.scatter(dataset[:, 0], dataset[:, 1], s=8, lw=0, c=np.asarray(assignments), cmap="tab10")
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def evaluate_clustering(dataset1: list[list[float]], n_clusters: int = 4,
                        original_labels: list = None, to_plot: bool = False,
                        max_iter: int = 300):
    all_assignments = {}
    if original_labels is not None:
        all_assignments["original"] = original_labels, None
    all_assignments["furthest"] = ias.furthest_mine(dataset1, n_clusters), None
    all_assignments["random"] = ias.random_assignment(dataset1, n_clusters), None

    km_model = KMeans(dataset1)
    km_model.initial_assignment = ias.random_assignment
    all_assignments["kmeans random"] = km_model.cluster(n_clusters, max_iter), km_model.centroids

    km_model = KMeans(dataset1)
    km_model.initial_assignment = ias.furthest_mine
    all_assignments["kmeans furthest"] = km_model.cluster(n_clusters, max_iter), km_model.centroids

    all_assignments["affinity"] = clustering_sk.affinity_propagation(dataset1, False), None
    all_assignments["dbscan"] = clustering_sk.dbscan(dataset1), None
    all_assignments["dirichlet"] = clustering_sk.dirichlet(dataset1, n_clusters, max_iter=max_iter), None
    all_assignments["hierachical"] = clustering_sk.hierarchical(dataset1, n_clusters), None
    all_assignments["gaussian mixture"] = clustering_sk.gaussian_mixture(dataset1, n_clusters), None
    all_assignments["hierarchical connected"] = clustering_sk.hierarchical_connected(dataset1, n_clusters), None
    all_assignments["kmeans"] = clustering_sk.kmeans(dataset1, n_clusters, max_iter=max_iter), None
    all_assignments["mean shift"] = clustering_sk.mean_shift(dataset1), None

    pp = pprint.PrettyPrinter(indent=4)
    result_metrics = {}
    for assignment in all_assignments:
        print("calculating metrics...", assignment)
        result_metrics[assignment] = clustering_metrics.evaluate_all_metrics(
            dataset1,
            all_assignments[assignment][0],
            all_assignments[assignment][1],
        )

    pp.pprint(result_metrics)
    if to_plot:
        _prepare_output_dir()
        for assignment_name, (assignment_values, _) in all_assignments.items():
            _save_cluster_plot(
                np.asarray(dataset1),
                assignment_values,
                f"Clustering — {assignment_name.title()}",
                f"{_slugify(assignment_name)}.png",
            )
    return result_metrics, all_assignments


if __name__ == "__main__":
    n_clusters = _get_env_int("ML_N_CLUSTERS", 4)
    max_iter = _get_env_int("ML_MAX_ITER", 300)
    samples = _get_env_int("ML_SAMPLES", 4000)
    to_plot = _get_env_bool("ML_PLOT", True)

    dataset, labels = _build_dataset(samples)
    evaluate_clustering(dataset, n_clusters, labels, to_plot=to_plot, max_iter=max_iter)
