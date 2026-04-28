# 🧪 Machine Learning Experiments

A collection of machine learning experiments, utilities, and interactive tools covering supervised/unsupervised learning, NLP, computer vision, audio processing, and psychoacoustics.

## 📂 Project Structure

```
ML-experiments/
├── src/mlexperiments/          # Core library (installable package)
│   ├── deps/                  # Low-level dependencies (VAD, LPC, etc.)
│   ├── load_data/             # Data loaders (audio, images, EEG, NLP, etc.)
│   ├── preprocessing/         # Preprocessing (image, text, signal, psychoacoustics)
│   ├── supervised/            # Supervised models (KNN, Perceptron, Linear Regression)
│   ├── unsupervised/          # Unsupervised models (Autoencoders, Clustering)
│   └── utils/                 # Utilities (types, statistics, image, torch persistence)
├── test/                      # Experiment scripts
│   ├── IMAGE_*.py             # Image experiments (MNIST, GAN, Butterfly segmentation)
│   ├── TEXT_*.py              # NLP experiments (TF-IDF, Sentiment, Topic Modeling)
│   ├── CLUSTERING_*.py        # Clustering benchmarks & Monte Carlo
│   ├── Autoencoder_*.py       # Autoencoder on Iris/MNIST
│   ├── Confusion_Matrices.py  # Confusion matrix visualization
│   ├── psychoacoustic_plots.py# Psychoacoustic scale plots
│   └── load_all_files.py      # Bulk data loader
├── web_gui/                   # Web-based GUI (Flask + pywebview)
│   ├── app.py                 # Flask server with REST API
│   ├── templates/             # HTML pages (Jinja2)
│   └── static/                # CSS & JavaScript
├── data/                      # Datasets (downloaded on demand)
│   ├── sample_data/
│   ├── train_data/            # Kaggle datasets (download via scripts)
│   └── created_models/
├── run_web_gui.py             # Web GUI launcher
├── call_test.py               # CLI: pick & run a test file
├── call_test_gui.py           # Legacy tkinter GUI
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## 🌐 Web GUI

The web GUI replaces the old tkinter interfaces with a browser-based experience powered by **Flask** (backend) and **pywebview** (optional desktop wrapper). It provides three pages:

| Page | Route | Description |
|------|-------|-------------|
| 📁 **Test Files** | `/` | Browse, edit (CodeMirror), save, and run `test/*.py` files with live output |
| 🎵 **Psychoacoustic** | `/psychoacoustic` | Interactive plots: Bark scale, ERB, MEL scale, and Sone vs Phons |
| 🦋 **Butterfly** | `/butterfly` | Image segmentation: train model, upload images, view predictions |

## 🚀 Quick Start

### Option 1 — Web GUI (recommended)

**Local environment:**
```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 2b. Install the exact spaCy model you plan to use for Spanish topic modeling
python -m spacy download es_core_news_lg

# 3a. Launch web server (default: http://0.0.0.0:3004)
python run_web_gui.py

# 3b. Or launch as desktop app (opens a native window)
python run_web_gui.py --desktop

# 3c. Custom host/port
python run_web_gui.py --host 127.0.0.1 --port 8080
```

**Command-line flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `3004` | Port number |
| `--desktop` | off | Open as a pywebview desktop window |
| `--debug` | off | Enable Flask debug mode |

### Option 2 — Docker

```bash
# Build the image
docker build -t erickfmm/mlexp .

# Run the web GUI (exposed on port 3004)
docker run -it --rm -p 3004:3004 erickfmm/mlexp

# Or use the convenience scripts:
bash docker_run_gui.sh    # Linux/macOS
docker_run_gui.bat        # Windows
```

The Docker image uses `uv` for fast dependency installation and automatically starts the web GUI on port `3004`.

The topic-modeling pipeline loads the exact spaCy model you configure. If that model is not installed, the run fails with a clear error instead of silently switching to a different pipeline.

### Option 3 — CLI (simple file picker)

```bash
python call_test.py
```
Lists all `test/*.py` files, you pick one by number, and it runs.

### Option 4 — Install as a Python package

```bash
pip install git+https://github.com/erickfmm/ML-experiments.git
```

Then import the modules you need:

```python
import mlexperiments.unsupervised.clustering.cluster_sklearn as clustering_sk
import numpy as np

np.random.seed(844)
clust1 = np.random.normal(5, 2, (1000, 2))
clust2 = np.random.normal(15, 3, (1000, 2))
clust3 = np.random.multivariate_normal([17, 3], [[1, 0], [0, 1]], 1000)
clust4 = np.random.multivariate_normal([2, 16], [[1, 0], [0, 1]], 1000)
dataset1 = np.concatenate((clust1, clust2, clust3, clust4))

assignments = clustering_sk.dbscan(dataset1)
```

## 📥 Downloading Data

Datasets are downloaded automatically when using the `download()` function of each loader. The first time, it will prompt for your **Kaggle** username and API key (from `kaggle.json`).

Some datasets have manual download scripts in `data/train_data/`:
```bash
cd data/train_data/NLP_ENG && bash download.sh
cd data/train_data/Manga_Anime && bash download.sh
# ... etc.
```

## 🧩 Key Dependencies

| Category | Libraries |
|----------|-----------|
| ML / Deep Learning | `torch`, `torchvision`, `scikit-learn` |
| NLP | `nltk`, `spacy`, `gensim`, `wordcloud`, `pyldavis`, `beautifulsoup4` |
| Audio / Signal | `librosa`, `pyedflib`, `pywavelets` |
| Image | `opencv-python`, `Pillow`, `pypng` |
| Visualization | `matplotlib`, `seaborn` |
| Data | `numpy`, `pandas`, `scipy` |
| Web GUI | `flask`, `pywebview` |

## ✅ TODO

- [ ] More data loaders
- [ ] API documentation (pydoc or similar)
- [ ] UML diagrams
- [x] ~~Better menu — replaced with Web GUI~~
- [ ] Documentation inside test/ files
- [ ] More ML models and experiments

## 📸 Screenshots

**Web GUI — Test Files editor:**

![selector of test file](https://github.com//erickfmm/ML-experiments/blob/master/docs/selector_of_test.png?raw=true)

**Web GUI — Psychoacoustic plots:**

![Psychoacoustic Plots](https://github.com//erickfmm/ML-experiments/blob/master/docs/psychoacoustic_plots.png?raw=true)

**Web GUI — Butterfly Segmentation:**

![Segmentation of Butterflies](https://github.com//erickfmm/ML-experiments/blob/master/docs/segmentator_of_butterflies.png?raw=true)