"""Compute transformer text embeddings concatenated with numeric features."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import RANDOM_SEED

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBEDDING_DIM = 384

# The raw 384-dim text embedding includes many near-noise components. Without
# reduction they swamp the small block of numeric features inside Isolation
# Forest (which picks features uniformly at random). Compressing the text
# block to its top principal components — enough to retain 99% of its
# variance — balances the two halves of the fingerprint and lets numeric
# signals (amount, single-source method, end-of-period) influence ranking.
TEXT_PCA_VARIANCE = 0.99


def _build_text_inputs(df: pd.DataFrame) -> list[str]:
    """Compose the per-row text string fed to the sentence transformer."""
    return [
        f"{row.vendor_name} | {row.category} | {row.procurement_method} | {row.contract_description}"
        for row in df.itertuples(index=False)
    ]


def _is_end_of_quarter(d: date) -> int:
    """Return 1 if the date sits in the last 14 days of any calendar quarter."""
    quarter_ends = [
        date(d.year, 3, 31),
        date(d.year, 6, 30),
        date(d.year, 9, 30),
        date(d.year, 12, 31),
    ]
    for end in quarter_ends:
        delta = (end - d).days
        if 0 <= delta < 14:
            return 1
    return 0


def _build_numeric_features(df: pd.DataFrame) -> np.ndarray:
    """Build the ~17-dim numeric feature block (scaled and one-hot encoded)."""
    dates = pd.to_datetime(df["date"])
    fiscal_year_start = pd.to_datetime(dates.dt.year.astype(str) + "-01-01")
    days_since_start = (dates - fiscal_year_start).dt.days.to_numpy().reshape(-1, 1)
    amount_log = np.log1p(df["amount"].to_numpy()).reshape(-1, 1)
    payment_terms = df["payment_terms_days"].to_numpy().reshape(-1, 1).astype(float)
    is_eop = np.array([_is_end_of_quarter(d.date()) for d in dates]).reshape(-1, 1).astype(float)
    year_index = (dates.dt.year - dates.dt.year.min()).to_numpy().reshape(-1, 1).astype(float)

    scaler = StandardScaler()
    scaled_continuous = scaler.fit_transform(
        np.hstack([amount_log, payment_terms, days_since_start, year_index])
    )

    month_one_hot = np.zeros((len(df), 12), dtype=float)
    months = dates.dt.month.to_numpy()
    month_one_hot[np.arange(len(df)), months - 1] = 1.0

    return np.hstack([scaled_continuous, is_eop, month_one_hot])


def _compute_text_embeddings(texts: list[str]) -> np.ndarray:
    """Run the sentence transformer and return a (N, 384) numpy array."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers and torch are required. "
            "Install dependencies via `pip install -r requirements.txt`."
        ) from exc

    torch.manual_seed(RANDOM_SEED)
    try:
        model = SentenceTransformer(TEXT_MODEL_NAME)
    except Exception as exc:  # pragma: no cover - depends on network state
        raise RuntimeError(
            f"Failed to load {TEXT_MODEL_NAME}. This requires an internet connection "
            "on first run to download the model (~80MB), after which it works offline."
        ) from exc

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def _reduce_text(text_embeddings: np.ndarray) -> np.ndarray:
    """Project the 384-dim text embedding onto its top-variance PCA basis."""
    pca = PCA(n_components=TEXT_PCA_VARIANCE, random_state=RANDOM_SEED)
    return pca.fit_transform(text_embeddings)


def _cache_is_fresh(csv_path: Path, cache_path: Path) -> bool:
    """Return True if the cache file is newer than the source CSV."""
    if not cache_path.exists():
        return False
    return cache_path.stat().st_mtime >= csv_path.stat().st_mtime


def compute_embeddings(
    df: pd.DataFrame,
    cache_path: str | Path = "data/embeddings.npy",
    csv_path: str | Path = "data/synthetic_procurement.csv",
) -> np.ndarray:
    """Return the full feature matrix, with caching."""
    cache_path = Path(cache_path)
    csv_path = Path(csv_path)

    if csv_path.exists() and _cache_is_fresh(csv_path, cache_path):
        cached = np.load(cache_path)
        if cached.shape[0] == len(df):
            return cached

    text_embeddings = _compute_text_embeddings(_build_text_inputs(df))
    text_reduced = _reduce_text(text_embeddings)
    numeric_features = _build_numeric_features(df)
    matrix = np.hstack([text_reduced, numeric_features]).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix)
    return matrix
