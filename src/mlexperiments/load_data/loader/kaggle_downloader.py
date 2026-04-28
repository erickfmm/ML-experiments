"""
Drop-in replacement for ``opendatasets`` using ``kagglehub``.

Usage (same interface as opendatasets)::

    from mlexperiments.load_data.loader.kaggle_downloader import download

    download("https://www.kaggle.com/datasets/owner/dataset", "data/train_data/Folder")

Authentication is handled via the ``KAGGLE_API_TOKEN`` environment variable,
which can be set in a ``.env`` file at the project root or through the
web GUI settings page.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Lazy-loaded kagglehub ─────────────────────────────────────────────
_kagglehub = None


def _get_kagglehub():  # pragma: no cover
    global _kagglehub
    if _kagglehub is None:
        import kagglehub
        _kagglehub = kagglehub
    return _kagglehub


# ── Environment helpers ───────────────────────────────────────────────

def _load_dotenv() -> None:
    """Load .env from the project root (best-effort)."""
    try:
        from dotenv import load_dotenv
        # Walk up from this file to find the project root (where .env lives)
        project_root = Path(__file__).resolve().parents[4]  # src/mlexperiments/load_data/loader/ -> root
        load_dotenv(project_root / ".env", override=False)
    except ImportError:
        pass  # python-dotenv not installed – rely on system env


# Load once at import time
_load_dotenv()


# ── Public API ────────────────────────────────────────────────────────

def download(url: str, folder_path: str = ".") -> str:
    """Download a Kaggle dataset.

    Mimics the ``opendatasets.download()`` interface using ``kagglehub``.

    Parameters
    ----------
    url : str
        Full Kaggle dataset URL (e.g.
        ``"https://www.kaggle.com/datasets/owner/dataset"``) **or** a
        ``kagglehub``-style handle (``"owner/dataset"``).
    folder_path : str
        Parent directory where the dataset folder will be created.

    Returns
    -------
    str
        Path to the downloaded dataset directory.
    """
    kagglehub = _get_kagglehub()

    # ── Resolve handle from URL ────────────────────────────────────
    if url.startswith("http"):
        handle = url.rstrip("/").split("/datasets/")[-1]
    else:
        handle = url

    dataset_name = handle.split("/")[-1]
    output_dir = os.path.join(folder_path, dataset_name)

    # ── Ensure the parent directory exists ─────────────────────────
    os.makedirs(folder_path, exist_ok=True)

    # ── Download ───────────────────────────────────────────────────
    logger.info("Downloading dataset %s to %s", handle, output_dir)
    result_path = kagglehub.dataset_download(handle, output_dir=output_dir)

    # kagglehub may return the cache path when output_dir is not used,
    # or the actual output_dir.  We normalise to the output_dir.
    return str(result_path)
