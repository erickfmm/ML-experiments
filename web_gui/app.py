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

# In-memory store for running processes
_processes: dict[str, subprocess.Popen] = {}
_outputs: dict[str, list[str]] = {}


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
        for line in iter(proc.stdout.readline, ""):
            _outputs[rid].append(line)
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
    model_path = os.path.join(BASE_DIR, "data", "created_models", "butterfly_segmentation.pt")
    data_path = os.path.join(BASE_DIR, "data", "created_models", "butterfly_segment.pkl")
    return jsonify({
        "model_exists": os.path.isfile(model_path),
        "data_exists": os.path.isfile(data_path),
        "model_path": model_path,
        "data_path": data_path,
    })


@app.route("/api/butterfly/train_classifier", methods=["POST"])
def butterfly_train_classifier():
    """Run the butterfly classifier training script."""
    test_file = "IMAGE_segmentation_clasification_butterfly.py"
    filepath = os.path.join(TEST_DIR, test_file)
    run_id = str(uuid.uuid4())[:8]
    _outputs[run_id] = []

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["BUTTERFLY_TASK"] = "train_classifier"

    proc = subprocess.Popen(
        [sys.executable, "-u", filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=BASE_DIR,
        env=env,
    )
    _processes[run_id] = proc

    def _reader(proc, rid):
        for line in iter(proc.stdout.readline, ""):
            _outputs[rid].append(line)
        proc.wait()
        _outputs[rid].append(f"\n[Process exited with code {proc.returncode}]")

    threading.Thread(target=_reader, args=(proc, run_id), daemon=True).start()
    return jsonify({"run_id": run_id})


@app.route("/api/butterfly/train_segmentation", methods=["POST"])
def butterfly_train_segmentation():
    """Run the butterfly segmentation training script."""
    test_file = "IMAGE_segmentation_clasification_butterfly.py"
    filepath = os.path.join(TEST_DIR, test_file)
    run_id = str(uuid.uuid4())[:8]
    _outputs[run_id] = []

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["BUTTERFLY_TASK"] = "train_segmentation"

    proc = subprocess.Popen(
        [sys.executable, "-u", filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=BASE_DIR,
        env=env,
    )
    _processes[run_id] = proc

    def _reader(proc, rid):
        for line in iter(proc.stdout.readline, ""):
            _outputs[rid].append(line)
        proc.wait()
        _outputs[rid].append(f"\n[Process exited with code {proc.returncode}]")

    threading.Thread(target=_reader, args=(proc, run_id), daemon=True).start()
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
        for line in iter(proc.stdout.readline, ""):
            _outputs[rid].append(line)
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
