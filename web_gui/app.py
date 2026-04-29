import json
import os
import subprocess
import sys
import threading
import uuid

from flask import Flask, render_template, request, jsonify, send_from_directory

# Load .env at startup
try:
    from dotenv import load_dotenv
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    load_dotenv(os.path.join(_root, ".env"), override=False)
except ImportError:
    pass

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)

# ── Configuration ──────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DIR = os.path.join(BASE_DIR, "test")
RESULT_DIRS = {
    "tfidf": os.path.join(BASE_DIR, "data", "created_models", "tfidf_tests"),
    "clustering": os.path.join(BASE_DIR, "data", "created_models", "clustering_benchmarks"),
    "confusion": os.path.join(BASE_DIR, "data", "created_models", "confusion_matrices"),
    "gan": os.path.join(BASE_DIR, "data", "created_models", "gan"),
}

# In-memory store for running processes
_processes: dict[str, subprocess.Popen] = {}
_outputs: dict[str, list[str]] = {}


def _humanize_artifact_name(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    stem = stem.replace("_", " ").replace("-", " ").strip()
    parts = [part for part in stem.split() if not part.isdigit()]
    return " ".join(parts).title() if parts else filename


def _list_result_images(group: str):
    result_dir = RESULT_DIRS.get(group)
    if not result_dir:
        return None

    if not os.path.isdir(result_dir):
        return []

    items = []
    for name in os.listdir(result_dir):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            continue
        path = os.path.join(result_dir, name)
        if not os.path.isfile(path):
            continue
        updated = int(os.path.getmtime(path))
        items.append({
            "name": name,
            "title": _humanize_artifact_name(name),
            "updated": updated,
            "url": f"/api/visualizations/{group}/{name}?v={updated}",
        })

    return sorted(items, key=lambda item: (item["updated"], item["name"]), reverse=True)


# ══════════════════════════════════════════════════════════════════════
#  HTML Pages
# ══════════════════════════════════════════════════════════════════════


@app.route("/")
def index():
    """Test files browser / editor / runner."""
    return render_template("index.html")


@app.route("/psychoacoustic")
def psychoacoustic():
    """Psychoacoustic plots page."""
    return render_template("psychoacoustic.html")


@app.route("/butterfly")
def butterfly():
    """Butterfly segmentation page."""
    return render_template("butterfly.html")


@app.route("/settings")
def settings():
    """Settings page – Kaggle API token configuration."""
    return render_template("settings.html")


@app.route("/datasets")
def datasets():
    """Dataset download manager page."""
    return render_template("datasets.html")


@app.route("/autoencoder")
def autoencoder():
    """Autoencoder training page."""
    return render_template("autoencoder.html")


@app.route("/clustering")
def clustering():
    """Clustering benchmarks & Monte Carlo page."""
    return render_template("clustering.html")


@app.route("/confusion-matrices")
def confusion_matrices():
    """Confusion matrices page."""
    return render_template("confusion_matrices.html")


@app.route("/image-classifier")
def image_classifier():
    """CNN image classifier page."""
    return render_template("image_classifier.html")


@app.route("/sentiment")
def sentiment():
    """Sentiment analysis page."""
    return render_template("sentiment.html")


@app.route("/tfidf")
def tfidf():
    """TF-IDF analysis page."""
    return render_template("tfidf.html")


@app.route("/topic-modeling")
def topic_modeling():
    """Topic modeling & word clouds page."""
    return render_template("topic_modeling.html")


@app.route("/gan")
def gan():
    """GAN MNIST page."""
    return render_template("gan.html")


# ══════════════════════════════════════════════════════════════════════
#  Helper – run a test script with environment-variable params
# ══════════════════════════════════════════════════════════════════════


def _run_test_script(test_filename, extra_env=None):
    """Run a test/ file and return {run_id} or {error}."""
    filepath = os.path.join(TEST_DIR, test_filename)
    if not os.path.isfile(filepath):
        return None, f"File not found: {test_filename}"

    run_id = str(uuid.uuid4())[:8]
    _outputs[run_id] = []

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    proc = subprocess.Popen(
        [sys.executable, "-u", filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=BASE_DIR,
        env=env,
    )
    _processes[run_id] = proc

    def _reader(p, rid):
        for line in iter(p.stdout.readline, b""):
            _outputs[rid].append(line.decode("utf-8", errors="replace"))
        p.wait()
        _outputs[rid].append(f"\n[Process exited with code {p.returncode}]")

    threading.Thread(target=_reader, args=(proc, run_id), daemon=True).start()
    return run_id, None


# ══════════════════════════════════════════════════════════════════════
#  API – Test Files
# ══════════════════════════════════════════════════════════════════════


@app.route("/api/files", methods=["GET"])
def list_files():
    """Return sorted list of .py files in test/."""
    try:
        entries = sorted(os.listdir(TEST_DIR))
        py_files = [f for f in entries if f.endswith(".py")]
        return jsonify(py_files)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/file/<path:filename>", methods=["GET"])
def read_file(filename):
    """Return the content of a test file."""
    filepath = os.path.join(TEST_DIR, filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 404
    with open(filepath, "r", encoding="utf-8") as fh:
        content = fh.read()
    return jsonify({"filename": filename, "content": content})


@app.route("/api/file/<path:filename>", methods=["PUT"])
def save_file(filename):
    """Save content to a test file."""
    filepath = os.path.join(TEST_DIR, filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 404
    data = request.get_json(force=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(data.get("content", ""))
    return jsonify({"status": "saved"})


@app.route("/api/run/<path:filename>", methods=["POST"])
def run_file(filename):
    """Execute a test file and return a run-id for streaming output."""
    filepath = os.path.join(TEST_DIR, filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 404

    run_id = str(uuid.uuid4())[:8]
    _outputs[run_id] = []

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    def _reader(proc: subprocess.Popen, rid: str):
        for line in iter(proc.stdout.readline, b""):
            _outputs[rid].append(line.decode("utf-8", errors="replace"))
        proc.wait()
        _outputs[rid].append(f"\n[Process exited with code {proc.returncode}]")

    proc = subprocess.Popen(
        [sys.executable, "-u", filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=BASE_DIR,
        env=env,
    )
    _processes[run_id] = proc
    t = threading.Thread(target=_reader, args=(proc, run_id), daemon=True)
    t.start()

    return jsonify({"run_id": run_id})


@app.route("/api/output/<run_id>", methods=["GET"])
def get_output(run_id):
    """Return accumulated output for a run and clear it."""
    lines = _outputs.get(run_id, [])
    _outputs[run_id] = []
    return jsonify({"lines": lines})


@app.route("/api/visualizations/<group>", methods=["GET"])
def api_list_visualizations(group):
    """List saved visualization images for a known result group."""
    images = _list_result_images(group)
    if images is None:
        return jsonify({"error": f"Unknown visualization group: {group}"}), 404
    return jsonify({"exists": bool(images), "images": images})


@app.route("/api/visualizations/<group>/<path:filename>", methods=["GET"])
def api_get_visualization(group, filename):
    """Serve a saved visualization image for a known result group."""
    result_dir = RESULT_DIRS.get(group)
    if not result_dir:
        return jsonify({"error": f"Unknown visualization group: {group}"}), 404
    return send_from_directory(result_dir, filename)


@app.route("/api/kill/<run_id>", methods=["POST"])
def kill_run(run_id):
    """Kill a running process."""
    proc = _processes.get(run_id)
    if proc and proc.poll() is None:
        proc.terminate()
        return jsonify({"status": "terminated"})
    return jsonify({"status": "not running"})


# ══════════════════════════════════════════════════════════════════════
#  API – Psychoacoustic data
# ══════════════════════════════════════════════════════════════════════


@app.route("/api/psychoacoustic", methods=["GET"])
def psychoacoustic_data():
    """Compute and return psychoacoustic scale data as JSON."""
    import importlib

    min_val = int(request.args.get("min", 0))
    max_val = int(request.args.get("max", 20000))
    step = int(request.args.get("step", 10))
    max_sone = int(request.args.get("max_sone", 625))

    # Add src to path so the modules can be imported
    src_path = os.path.join(BASE_DIR, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    bark_mod = importlib.import_module("mlexperiments.preprocessing.psychoacoustics.bark")
    erb_mod = importlib.import_module("mlexperiments.preprocessing.psychoacoustics.erb")
    mel_mod = importlib.import_module("mlexperiments.preprocessing.psychoacoustics.mel")
    sone_mod = importlib.import_module("mlexperiments.preprocessing.psychoacoustics.sone")

    x = list(range(min_val, max_val, step))
    x_sone = list(range(1, max_sone, step))

    data = {
        "x": x,
        "bark": {
            "7000": [bark_mod.bark(i) for i in x],
            "1990Traunmuller": [bark_mod.bark_1990_traunmuller(i) for i in x],
            "1992Wang": [bark_mod.bark_1992_wang(i) for i in x],
            "7500": [bark_mod.bark2(i) for i in x],
        },
        "erb": {
            "linear": [erb_mod.erb_linear(i) for i in x],
            "poly2nd": [erb_mod.erb_2ndorder_poly(i) for i in x],
            "matlab": [erb_mod.erb_matlab_voicebox(i) for i in x],
        },
        "mel": {
            "700": [mel_mod.mel_700(i) for i in x],
            "1000": [mel_mod.mel_1000(i) for i in x],
            "625": [mel_mod.mel_625(i) for i in x],
        },
        "x_sone": x_sone,
        "sone": {
            "sone": [sone_mod.sone(i) for i in x_sone],
            "approximation": [sone_mod.sone_aproximation(i) for i in x_sone],
        },
    }
    return jsonify(data)


# ══════════════════════════════════════════════════════════════════════
#  API – Butterfly segmentation
# ══════════════════════════════════════════════════════════════════════


@app.route("/api/butterfly/status", methods=["GET"])
def butterfly_status():
    """Check if butterfly model and data exist."""
    base = os.path.join(BASE_DIR, "data")
    model_seg = os.path.join(base, "created_models", "butterfly_segmentation.pt")
    model_cls = os.path.join(base, "created_models", "butterfly_classifier.pt")
    data_seg = os.path.join(base, "created_models", "butterfly_segment.pkl")
    data_cls = os.path.join(base, "created_models", "butterfly.pkl")
    dataset_dir = os.path.join(base, "train_data", "Images_Supervised", "butterfly-dataset", "leedsbutterfly")
    return jsonify({
        "dataset_exists": os.path.isdir(dataset_dir),
        "data_exists": os.path.isfile(data_seg),
        "data_cls_exists": os.path.isfile(data_cls),
        "seg_model_exists": os.path.isfile(model_seg),
        "cls_model_exists": os.path.isfile(model_cls),
    })


def _run_butterfly_task(task, extra_env=None):
    """Helper: run the butterfly training script with a given task and env vars."""
    test_file = "IMAGE_segmentation_clasification_butterfly.py"
    filepath = os.path.join(TEST_DIR, test_file)
    if not os.path.isfile(filepath):
        return None, "Training script not found"

    run_id = str(uuid.uuid4())[:8]
    _outputs[run_id] = []

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["BUTTERFLY_TASK"] = task
    if extra_env:
        env.update(extra_env)

    proc = subprocess.Popen(
        [sys.executable, "-u", filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=BASE_DIR,
        env=env,
    )
    _processes[run_id] = proc

    def _reader(proc, rid):
        for line in iter(proc.stdout.readline, b""):
            _outputs[rid].append(line.decode("utf-8", errors="replace"))
        proc.wait()
        _outputs[rid].append(f"\n[Process exited with code {proc.returncode}]")

    threading.Thread(target=_reader, args=(proc, run_id), daemon=True).start()
    return run_id, None


@app.route("/api/butterfly/download", methods=["POST"])
def butterfly_download():
    """Download the butterfly dataset from Kaggle."""
    run_id, err = _run_butterfly_task("download")
    if err:
        return jsonify({"error": err}), 500
    return jsonify({"run_id": run_id})


@app.route("/api/butterfly/save_pickle", methods=["POST"])
def butterfly_save_pickle():
    """Convert dataset to pickle files."""
    params = request.get_json(force=True, silent=True) or {}
    image_size = params.get("image_size", 100)
    run_id, err = _run_butterfly_task("save_pickle", {"BUTTERFLY_IMAGE_SIZE": str(image_size)})
    if err:
        return jsonify({"error": err}), 500
    return jsonify({"run_id": run_id})


@app.route("/api/butterfly/train_classifier", methods=["POST"])
def butterfly_train_classifier():
    """Run the butterfly classifier training script."""
    params = request.get_json(force=True, silent=True) or {}
    extra = {
        "BUTTERFLY_EPOCHS": str(params.get("epochs", 100)),
        "BUTTERFLY_BATCH_SIZE": str(params.get("batch_size", 16)),
        "BUTTERFLY_LR": str(params.get("learning_rate", 0.001)),
        "BUTTERFLY_TEST_SPLIT": str(params.get("test_split", 0.33)),
    }
    run_id, err = _run_butterfly_task("train_classifier", extra)
    if err:
        return jsonify({"error": err}), 500
    return jsonify({"run_id": run_id})


@app.route("/api/butterfly/train_segmentation", methods=["POST"])
def butterfly_train_segmentation():
    """Run the butterfly segmentation training script."""
    params = request.get_json(force=True, silent=True) or {}
    extra = {
        "BUTTERFLY_EPOCHS": str(params.get("epochs", 100)),
        "BUTTERFLY_BATCH_SIZE": str(params.get("batch_size", 16)),
        "BUTTERFLY_LR": str(params.get("learning_rate", 0.001)),
        "BUTTERFLY_TEST_SPLIT": str(params.get("test_split", 0.33)),
    }
    run_id, err = _run_butterfly_task("train_segmentation", extra)
    if err:
        return jsonify({"error": err}), 500
    return jsonify({"run_id": run_id})


@app.route("/api/butterfly/predict_segmentation", methods=["POST"])
def butterfly_predict_segmentation():
    """Run segmentation prediction on an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    upload_dir = os.path.join(BASE_DIR, "data", "created_models")
    os.makedirs(upload_dir, exist_ok=True)

    img_file = request.files["image"]
    img_path = os.path.join(upload_dir, f"_upload_{img_file.filename}")
    img_file.save(img_path)

    # Run prediction script
    run_id = str(uuid.uuid4())[:8]
    _outputs[run_id] = []

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["BUTTERFLY_TASK"] = "predict_segmentation"
    env["BUTTERFLY_IMAGE"] = img_path

    filepath = os.path.join(TEST_DIR, "IMAGE_segmentation_clasification_butterfly.py")
    proc = subprocess.Popen(
        [sys.executable, "-u", filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=BASE_DIR,
        env=env,
    )
    _processes[run_id] = proc

    def _reader(proc, rid):
        for line in iter(proc.stdout.readline, b""):
            _outputs[rid].append(line.decode("utf-8", errors="replace"))
        proc.wait()
        _outputs[rid].append(f"\n[Process exited with code {proc.returncode}]")

    threading.Thread(target=_reader, args=(proc, run_id), daemon=True).start()
    return jsonify({"run_id": run_id})


@app.route("/api/butterfly/segmentation_results", methods=["GET"])
def butterfly_segmentation_results():
    """Return segmentation result images if they exist."""
    results_dir = os.path.join(BASE_DIR, "data", "created_models", "butterfly_results")
    if not os.path.isdir(results_dir):
        return jsonify({"exists": False})
    images = sorted(
        f for f in os.listdir(results_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    return jsonify({"exists": True, "images": images})


@app.route("/api/butterfly/results_image/<path:filename>")
def butterfly_results_image(filename):
    results_dir = os.path.join(BASE_DIR, "data", "created_models", "butterfly_results")
    return send_from_directory(results_dir, filename)


# ══════════════════════════════════════════════════════════════════════
#  API – Butterfly prediction helpers (inline)
# ══════════════════════════════════════════════════════════════════════


@app.route("/api/butterfly/predict", methods=["POST"])
def butterfly_predict():
    """Run butterfly segmentation prediction inline and return result image."""
    import numpy as np

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    model_path = os.path.join(BASE_DIR, "data", "created_models", "butterfly_segmentation.pt")
    if not os.path.isfile(model_path):
        return jsonify({"error": "Model not found. Train first."}), 404

    try:
        import torch
        from PIL import Image
        import io, base64

        # Load model
        # Import model class from test file
        src_path = os.path.join(BASE_DIR, "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        from test.IMAGE_segmentation_clasification_butterfly import SimpleSegmentation

        model = SimpleSegmentation()
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()

        # Process image
        img_file = request.files["image"]
        image = Image.open(img_file.stream).convert("RGB")
        image = image.resize((100, 100))
        img_array = np.array(image) / 255.0
        img_tensor = torch.tensor(
            np.expand_dims(img_array, axis=0), dtype=torch.float32
        ).permute(0, 3, 1, 2)

        with torch.no_grad():
            prediction = model(img_tensor).permute(0, 2, 3, 1).numpy()[0]

        # Clamp prediction
        prediction = np.clip(prediction, 0, 1)

        # Create cropped image
        cropped = img_array * prediction

        # Encode results as base64
        def _to_b64(arr):
            pil_img = Image.fromarray((arr * 255).astype(np.uint8))
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "predicted": _to_b64(prediction),
            "original": _to_b64(img_array),
            "cropped": _to_b64(cropped),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ══════════════════════════════════════════════════════════════════════
#  API – Autoencoder
# ══════════════════════════════════════════════════════════════════════


@app.route("/api/autoencoder/run", methods=["POST"])
def api_autoencoder_run():
    """Run Autoencoder_Iris_Mnist.py with env-var parameters."""
    data = request.get_json(force=True)
    run_id, err = _run_test_script("Autoencoder_Iris_Mnist.py", {
        "ML_DATASET": data.get("dataset", "mnist"),
        "ML_EPOCHS": str(data.get("epochs", 50)),
        "ML_CODING_FACTOR": str(data.get("coding_factor", 0.5)),
        "ML_TEST_SPLIT": str(data.get("test_split", 0.2)),
    })
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"run_id": run_id})


# ══════════════════════════════════════════════════════════════════════
#  API – Clustering
# ══════════════════════════════════════════════════════════════════════


@app.route("/api/clustering/benchmarks", methods=["POST"])
def api_clustering_benchmarks():
    """Run CLUSTERING_Benchmarks.py with env-var parameters."""
    data = request.get_json(force=True)
    run_id, err = _run_test_script("CLUSTERING_Benchmarks.py", {
        "ML_N_CLUSTERS": str(data.get("n_clusters", 3)),
        "ML_MAX_ITER": str(data.get("max_iter", 300)),
        "ML_SAMPLES": str(data.get("samples", 300)),
        "ML_PLOT": str(data.get("plot", True)),
    })
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"run_id": run_id})


@app.route("/api/clustering/montecarlo", methods=["POST"])
def api_clustering_montecarlo():
    """Run CLUSTERING_Montecarlo.py with env-var parameters."""
    data = request.get_json(force=True)
    run_id, err = _run_test_script("CLUSTERING_Montecarlo.py", {
        "ML_RUNS": str(data.get("runs", 100)),
        "ML_N_CLUSTERS": str(data.get("n_clusters", 3)),
        "ML_MAX_ITER": str(data.get("max_iter", 300)),
    })
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"run_id": run_id})


# ══════════════════════════════════════════════════════════════════════
#  API – Confusion Matrices
# ══════════════════════════════════════════════════════════════════════


@app.route("/api/confusion_matrices/run", methods=["POST"])
def api_confusion_matrices_run():
    """Run Confusion_Matrices.py with env-var parameters."""
    data = request.get_json(force=True)
    run_id, err = _run_test_script("Confusion_Matrices.py", {
        "ML_INSTANCES": str(data.get("instances", 1000)),
        "ML_DIMENSIONS": str(data.get("dimensions", 5)),
        "ML_N_CLASSES": str(data.get("n_classes", 3)),
        "ML_CV_FOLDS": str(data.get("cv_folds", 10)),
    })
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"run_id": run_id})


# ══════════════════════════════════════════════════════════════════════
#  API – Image Classifier (CNN)
# ══════════════════════════════════════════════════════════════════════


@app.route("/api/image_classifier/run", methods=["POST"])
def api_image_classifier_run():
    """Run IMAGE_convolutional_clasiffier_mnist.py with env-var parameters."""
    data = request.get_json(force=True)
    run_id, err = _run_test_script("IMAGE_convolutional_clasiffier_mnist.py", {
        "ML_DATASET": data.get("dataset", "mnist"),
        "ML_BATCH_SIZE": str(data.get("batch_size", 32)),
        "ML_EPOCHS": str(data.get("epochs", 10)),
        "ML_LEARNING_RATE": str(data.get("learning_rate", 0.001)),
    })
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"run_id": run_id})


@app.route("/api/image_classifier/status", methods=["GET"])
def api_image_classifier_status():
    """Check if a trained CNN model file exists."""
    model_dir = os.path.join(BASE_DIR, "data", "created_models")
    for name in ("cnn_model.pth", "cnn_model.pt", "mnist_cnn.pth"):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return jsonify({"exists": True, "path": name})
    return jsonify({"exists": False})


# ══════════════════════════════════════════════════════════════════════
#  API – Sentiment Analysis
# ══════════════════════════════════════════════════════════════════════


@app.route("/api/sentiment/train", methods=["POST"])
def api_sentiment_train():
    """Run TEXT_Sentiment_Analysis1.py with env-var parameters."""
    data = request.get_json(force=True)
    run_id, err = _run_test_script("TEXT_Sentiment_Analysis1.py", {
        "ML_EMB_DIM": str(data.get("emb_dim", 200)),
        "ML_NB_FILTERS": str(data.get("nb_filters", 100)),
        "ML_FFN_UNITS": str(data.get("ffn_units", 256)),
        "ML_BATCH_SIZE": str(data.get("batch_size", 256)),
        "ML_EPOCHS": str(data.get("epochs", 25)),
        "ML_DROPOUT": str(data.get("dropout", 0.2)),
        "ML_DOWNSAMPLE": str(data.get("downsample", 0.8)),
        "ML_TOKENIZER": data.get("tokenizer", "tfds"),
        "ML_TASK": "train",
    })
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"run_id": run_id})


@app.route("/api/sentiment/evaluate", methods=["POST"])
def api_sentiment_evaluate():
    """Evaluate sentiment of a text string using a trained model."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Run the sentiment script in evaluate mode
    run_id, err = _run_test_script("TEXT_Sentiment_Analysis1.py", {
        "ML_TASK": "evaluate",
        "ML_EVAL_TEXT": text,
    })
    if err:
        return jsonify({"error": err}), 400

    # Wait for process to finish (short scripts)
    proc = _processes.get(run_id)
    if proc:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Collect output and try to extract prediction
    lines = _outputs.get(run_id, [])
    output_text = "".join(lines)
    # Look for sentiment label in output
    sentiment = "Unknown"
    for line in lines:
        line_lower = line.lower().strip()
        if "positive" in line_lower:
            sentiment = "Positive 😊"
            break
        elif "negative" in line_lower:
            sentiment = "Negative 😞"
            break
        elif "neutral" in line_lower:
            sentiment = "Neutral 😐"
            break

    return jsonify({"sentiment": sentiment, "raw_output": output_text})


# ══════════════════════════════════════════════════════════════════════
#  API – TF-IDF
# ══════════════════════════════════════════════════════════════════════


@app.route("/api/tfidf/tests", methods=["POST"])
def api_tfidf_tests():
    """Run TEXT_TFIDF_tests.py with env-var parameters."""
    data = request.get_json(force=True)
    run_id, err = _run_test_script("TEXT_TFIDF_tests.py", {
        "ML_PLOT_TF": str(data.get("plot_tf", True)),
        "ML_PLOT_IDF": str(data.get("plot_idf", True)),
        "ML_PLOT_TFIDF_DOC": str(data.get("plot_tfidf_doc", True)),
        "ML_PLOT_TFIDF_QUERY": str(data.get("plot_tfidf_query", True)),
    })
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"run_id": run_id})


@app.route("/api/tfidf/bigtext", methods=["POST"])
def api_tfidf_bigtext():
    """Run TEXT_TFIDF_Bigtexts.py."""
    run_id, err = _run_test_script("TEXT_TFIDF_Bigtexts.py")
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"run_id": run_id})


# ══════════════════════════════════════════════════════════════════════
#  API – Topic Modeling & Word Clouds
# ══════════════════════════════════════════════════════════════════════


@app.route("/api/topic_modeling/run", methods=["POST"])
def api_topic_modeling_run():
    """Run TEXT_TopicModeling_Bigtexts.py with env-var parameters."""
    data = request.get_json(force=True)
    spacy_model = str(data.get("spacy_model", "es_core_news_sm")).strip()
    if not spacy_model:
        return jsonify({"error": "spaCy model is required."}), 400
    run_id, err = _run_test_script("TEXT_TopicModeling_Bigtexts.py", {
        "ML_DATASET": data.get("dataset", "wikihow"),
        "ML_N_TOPICS": str(data.get("n_topics", 20)),
        "ML_MAX_THREADS": str(data.get("maxthreads", 16)),
        "ML_SPACY_MODEL": spacy_model,
    })
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"run_id": run_id})


@app.route("/api/wordclouds/run", methods=["POST"])
def api_wordclouds_run():
    """Run TEXT_Wordclouds.py with env-var parameters."""
    data = request.get_json(force=True)
    run_id, err = _run_test_script("TEXT_Wordclouds.py", {
        "ML_MAX_WORDS": str(data.get("max_words", 100)),
        "ML_WIDTH": str(data.get("width", 2500)),
        "ML_HEIGHT": str(data.get("height", 2500)),
    })
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"run_id": run_id})


# ══════════════════════════════════════════════════════════════════════
#  API – GAN MNIST
# ══════════════════════════════════════════════════════════════════════


@app.route("/api/gan/train", methods=["POST"])
def api_gan_train():
    """Run IMAGE_GAN_Mnist.py with env-var parameters."""
    data = request.get_json(force=True)
    run_id, err = _run_test_script("IMAGE_GAN_Mnist.py", {
        "ML_RANDOM_DIM": str(data.get("random_dim", 100)),
        "ML_EPOCHS": str(data.get("epochs", 100)),
        "ML_BATCH_SIZE": str(data.get("batch_size", 128)),
        "ML_SEED": str(data.get("seed", 3335)),
    })
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"run_id": run_id})


@app.route("/api/gan/status", methods=["GET"])
def api_gan_status():
    """Check if a trained GAN model file exists."""
    model_dir = os.path.join(BASE_DIR, "data", "created_models")
    for name in (
        "gan_generator.pth",
        "gan_generator.pt",
        "gan_discriminator.pth",
        os.path.join("gan_mnist_v1", "generator.pt"),
        os.path.join("gan_mnist_v1", "discriminator.pt"),
    ):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return jsonify({"exists": True})
    return jsonify({"exists": False})


@app.route("/api/gan/generate", methods=["POST"])
def api_gan_generate():
    """Generate images from a trained GAN model."""
    data = request.get_json(force=True)
    n_images = data.get("n", 12)

    # Check if model exists
    model_dir = os.path.join(BASE_DIR, "data", "created_models")
    model_path = None
    for name in (
        "gan_generator.pth",
        "gan_generator.pt",
        os.path.join("gan_mnist_v1", "generator.pt"),
    ):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            model_path = p
            break

    if not model_path:
        return jsonify({"error": "No trained GAN model found. Train the model first."}), 400

    # Generate images using an inline Python script
    code = f"""
import sys, os, json, base64, io
sys.path.insert(0, {repr(BASE_DIR)})
os.chdir({repr(BASE_DIR)})

import torch
import numpy as np
from PIL import Image

model_path = {repr(model_path)}
n_images = {n_images}

# Try to load the generator
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

# Determine architecture from checkpoint
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    random_dim = checkpoint.get('random_dim', 100)
elif isinstance(checkpoint, dict):
    state_dict = checkpoint
    random_dim = state_dict.get('model.0.weight').shape[1] if 'model.0.weight' in state_dict else 100
else:
    state_dict = None
    random_dim = 100

# Try importing the model definition from the test script
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("gan_test", {repr(os.path.join(TEST_DIR, "IMAGE_GAN_Mnist.py"))})
    gan_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gan_mod)
    generator = gan_mod.Generator(random_dim)
    if state_dict:
        generator.load_state_dict(state_dict if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint else state_dict)
    else:
        generator = checkpoint
except Exception as e:
    print(f"Load error: {{e}}", file=sys.stderr)
    # Fallback: try using the whole checkpoint as a model
    generator = checkpoint

generator.eval()
images = []
with torch.no_grad():
    for i in range(n_images):
        z = torch.randn(1, random_dim)
        img_tensor = generator(z)
        # Reshape to image
        arr = img_tensor.squeeze().numpy()
        # Normalize to 0-255
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        if arr.ndim == 1:
            side = int(np.sqrt(arr.shape[0]))
            arr = arr.reshape(side, side)
        pil_img = Image.fromarray(arr, mode='L')
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()
        images.append(f"data:image/png;base64,{{b64}}")

print(json.dumps({{"images": images}}))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
            cwd=BASE_DIR,
        )
        # Find JSON in stdout
        for line in result.stdout.strip().split("\n"):
            if line.startswith("{"):
                parsed = json.loads(line)
                return jsonify(parsed)
        return jsonify({"error": f"Generation failed: {result.stderr[:500]}"}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Generation timed out"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════
#  API – Dataset Manager
# ══════════════════════════════════════════════════════════════════════


def _discover_datasets():
    """Scan data/train_data/ and return structured dataset info."""
    train_dir = os.path.join(BASE_DIR, "data", "train_data")
    if not os.path.isdir(train_dir):
        return []

    datasets = []
    for cat_name in sorted(os.listdir(train_dir)):
        cat_path = os.path.join(train_dir, cat_name)
        if not os.path.isdir(cat_path):
            continue

        dl_script = os.path.join(cat_path, "download.sh")
        if not os.path.isfile(dl_script):
            continue

        # Parse download.sh to extract individual datasets
        sub_datasets = []
        with open(dl_script, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            # Match kaggle datasets download -d owner/dataset
            if "kaggle datasets download -d " in line:
                handle = line.split("-d ", 1)[-1].strip()
                # Derive a display name from handle
                parts = handle.split("/")
                owner = parts[0] if len(parts) > 1 else ""
                ds_name = parts[-1] if len(parts) > 1 else handle
                sub_datasets.append({
                    "handle": handle,
                    "name": ds_name.replace("-", " ").replace("_", " ").title(),
                    "owner": owner,
                    "type": "kaggle",
                    "line": line,
                })
            # Match unzip to find the folder name
            elif "unzip " in line and ".zip" in line:
                zip_name = line.split("unzip ", 1)[-1].split(".zip")[0].strip()
                folder_name = line.split("-d ", 1)[-1].strip() if "-d " in line else zip_name
                # Attach folder to last sub_dataset
                if sub_datasets:
                    # Find which sub_dataset this unzip belongs to
                    for sd in reversed(sub_datasets):
                        if "folder" not in sd:
                            sd["folder"] = folder_name
                            break
            # Match curl/wget downloads
            elif line.startswith("curl ") or line.startswith("wget "):
                url = ""
                if "--output " in line:
                    url = line.split("--output ", 1)[-1].strip()
                elif line.startswith("wget "):
                    url = line.split()[1] if len(line.split()) > 1 else ""
                elif line.startswith("curl "):
                    parts = line.split()
                    url = parts[-1] if parts else ""

                # Get the folder from the following unzip line
                folder = ""
                source_url = ""
                if line.startswith("wget "):
                    source_url = line.split()[1] if len(line.split()) > 1 else ""
                elif line.startswith("curl "):
                    for p in line.split():
                        if p.startswith("http"):
                            source_url = p

                sub_datasets.append({
                    "handle": "",
                    "name": url.replace(".zip", "").replace("-", " ").replace("_", " ").title() if url else "Download",
                    "owner": "",
                    "type": "url",
                    "line": line,
                    "source_url": source_url,
                })

        # Check which sub-datasets are already downloaded
        for sd in sub_datasets:
            folder = sd.get("folder", sd.get("handle", "").split("/")[-1] if sd.get("handle") else "")
            sd_path = os.path.join(cat_path, folder) if folder else ""
            sd["downloaded"] = os.path.isdir(sd_path) and len(os.listdir(sd_path)) > 0 if sd_path else False

        # Determine category-level status
        total = len(sub_datasets)
        downloaded = sum(1 for sd in sub_datasets if sd.get("downloaded"))

        datasets.append({
            "category": cat_name,
            "total": total,
            "downloaded": downloaded,
            "sub_datasets": sub_datasets,
        })

    return datasets


@app.route("/api/datasets", methods=["GET"])
def api_list_datasets():
    """List all dataset categories with download status."""
    datasets = _discover_datasets()
    return jsonify(datasets)


@app.route("/api/datasets/download", methods=["POST"])
def api_download_datasets():
    """Download selected datasets by running download.sh scripts.

    Body: { "categories": ["Emotions_Voice", "NLP_ENG"], "sub_datasets": ["owner/dataset", ...] }
    If sub_datasets is provided, only those specific datasets are downloaded.
    If only categories is provided, all datasets in those categories are downloaded.
    """
    data = request.get_json(force=True)
    categories = data.get("categories", [])
    sub_datasets_filter = data.get("sub_datasets", [])

    if not categories:
        return jsonify({"error": "No categories selected"}), 400

    train_dir = os.path.join(BASE_DIR, "data", "train_data")

    # Build the download commands
    commands = []
    for cat_name in categories:
        cat_path = os.path.join(train_dir, cat_name)
        dl_script = os.path.join(cat_path, "download.sh")
        if not os.path.isfile(dl_script):
            continue

        with open(dl_script, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#") or not line:
                i += 1
                continue

            # For kaggle datasets: download + unzip + rm
            if "kaggle datasets download -d " in line:
                handle = line.split("-d ", 1)[-1].strip()
                # Check if we should filter this specific dataset
                if sub_datasets_filter and handle not in sub_datasets_filter:
                    i += 1
                    # Skip associated unzip/rm lines
                    while i < len(lines):
                        nxt = lines[i].strip()
                        if nxt.startswith("#") or not nxt:
                            i += 1
                            continue
                        if nxt.startswith("unzip ") or nxt.startswith("rm "):
                            i += 1
                            continue
                        break
                    continue

                zip_name = line.split()[-1].replace(".zip", "") if ".zip" in line else handle.split("/")[-1]
                # Find the unzip line for folder name
                folder = zip_name
                if i + 1 < len(lines) and "unzip " in lines[i + 1] and "-d " in lines[i + 1]:
                    folder = lines[i + 1].split("-d ", 1)[-1].strip()

                commands.append({
                    "type": "kaggle",
                    "category": cat_name,
                    "handle": handle,
                    "folder": folder,
                    "download_cmd": line,
                })
                i += 1
                continue

            # For curl/wget downloads
            if line.startswith("curl ") or line.startswith("wget "):
                commands.append({
                    "type": "url",
                    "category": cat_name,
                    "line": line,
                    "full_script_lines": [],
                })
                # Collect associated unzip/rm lines
                cmd_idx = len(commands) - 1
                j = i
                while j < len(lines):
                    nxt = lines[j].strip()
                    if nxt.startswith("#") or not nxt:
                        j += 1
                        continue
                    if nxt.startswith("curl ") or nxt.startswith("wget ") or nxt.startswith("kaggle "):
                        break
                    commands[cmd_idx]["full_script_lines"].append(nxt)
                    j += 1
                i = j
                continue

            i += 1

    if not commands:
        return jsonify({"error": "No downloadable datasets found for the selection"}), 400

    # Run all download commands in a single subprocess
    run_id = str(uuid.uuid4())[:8]
    _outputs[run_id] = []

    # Build a temporary bash script
    import tempfile
    script_lines = ["#!/bin/bash", "set -e", ""]
    for cmd in commands:
        if cmd["type"] == "kaggle":
            folder = cmd["folder"]
            cat = cmd["category"]
            dest = os.path.join(train_dir, cat)
            script_lines.append(f'echo "⬇ Downloading {cmd["handle"]}..."')
            script_lines.append(f'cd "{dest}"')
            script_lines.append(cmd["download_cmd"])
            script_lines.append(f'unzip -o *.zip -d "{folder}" 2>/dev/null || true')
            script_lines.append(f'rm -f *.zip')
            script_lines.append("")
        elif cmd["type"] == "url":
            cat = cmd["category"]
            dest = os.path.join(train_dir, cat)
            script_lines.append(f'echo "⬇ Downloading {cmd["line"][:60]}..."')
            script_lines.append(f'cd "{dest}"')
            script_lines.append(cmd["line"])
            for extra_line in cmd.get("full_script_lines", []):
                script_lines.append(extra_line)
            script_lines.append("")

    script_content = "\n".join(script_lines)
    script_file = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, prefix="ml_download_")
    script_file.write(script_content)
    script_file.close()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        ["bash", script_file.name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=BASE_DIR,
        env=env,
    )
    _processes[run_id] = proc

    def _reader(proc, rid, script_path):
        for line in iter(proc.stdout.readline, b""):
            _outputs[rid].append(line.decode("utf-8", errors="replace"))
        proc.wait()
        _outputs[rid].append(f"\n[Process exited with code {proc.returncode}]")
        # Cleanup temp script
        try:
            os.unlink(script_path)
        except OSError:
            pass

    threading.Thread(target=_reader, args=(proc, run_id, script_file.name), daemon=True).start()
    return jsonify({"run_id": run_id, "total_commands": len(commands)})


@app.route("/api/datasets/download_sh/<category>", methods=["POST"])
def api_download_sh(category):
    """Run the full download.sh for a specific category."""
    train_dir = os.path.join(BASE_DIR, "data", "train_data")
    dl_script = os.path.join(train_dir, category, "download.sh")
    if not os.path.isfile(dl_script):
        return jsonify({"error": f"No download.sh found for {category}"}), 404

    run_id = str(uuid.uuid4())[:8]
    _outputs[run_id] = []

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        ["bash", dl_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.join(train_dir, category),
        env=env,
    )
    _processes[run_id] = proc

    def _reader(proc, rid):
        for line in iter(proc.stdout.readline, b""):
            _outputs[rid].append(line.decode("utf-8", errors="replace"))
        proc.wait()
        _outputs[rid].append(f"\n[Process exited with code {proc.returncode}]")

    threading.Thread(target=_reader, args=(proc, run_id), daemon=True).start()
    return jsonify({"run_id": run_id})


# ══════════════════════════════════════════════════════════════════════
#  API – Settings / Kaggle Token
# ══════════════════════════════════════════════════════════════════════


@app.route("/api/kaggle-token", methods=["GET"])
def get_kaggle_token():
    """Return whether a Kaggle API token is configured (never expose the value)."""
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    return jsonify({
        "configured": bool(token),
        "masked": f"{'*' * (len(token) - 4)}{token[-4:]}" if len(token) > 4 else ("****" if token else ""),
    })


@app.route("/api/kaggle-token", methods=["POST"])
def save_kaggle_token():
    """Save the Kaggle API token to the .env file."""
    data = request.get_json(force=True)
    token = data.get("token", "").strip()
    if not token:
        return jsonify({"error": "Token cannot be empty"}), 400

    env_path = os.path.join(BASE_DIR, ".env")

    # Read existing .env content
    lines: list[str] = []
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

    # Update or append KAGGLE_API_TOKEN
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith("KAGGLE_API_TOKEN="):
            lines[i] = f"KAGGLE_API_TOKEN={token}\n"
            found = True
            break
    if not found:
        # Ensure there's a newline before appending
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"KAGGLE_API_TOKEN={token}\n")

    with open(env_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)

    # Update the current process environment
    os.environ["KAGGLE_API_TOKEN"] = token

    return jsonify({"status": "saved"})


@app.route("/api/kaggle-token", methods=["DELETE"])
def delete_kaggle_token():
    """Remove the Kaggle API token from the .env file."""
    env_path = os.path.join(BASE_DIR, ".env")

    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        lines = [l for l in lines if not l.strip().startswith("KAGGLE_API_TOKEN=")]
        with open(env_path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)

    os.environ.pop("KAGGLE_API_TOKEN", None)
    return jsonify({"status": "deleted"})


# ══════════════════════════════════════════════════════════════════════
#  Server helper
# ══════════════════════════════════════════════════════════════════════


def create_app():
    """Factory for the Flask app (used by pywebview)."""
    return app
