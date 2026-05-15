"""Anomaly scoring with Isolation Forest."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import RANDOM_SEED


def score_transactions(embeddings: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """Adds anomaly_score and rank columns. Returns a new DataFrame."""
    if embeddings.shape[0] != len(df):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) do not match DataFrame rows ({len(df)})."
        )

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(embeddings)

    raw = -model.decision_function(embeddings)
    lo, hi = raw.min(), raw.max()
    normalized = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)

    scored = df.copy()
    scored["anomaly_score"] = normalized
    scored["rank"] = (
        scored["anomaly_score"].rank(method="first", ascending=False).astype(int)
    )
    return scored


def get_top_k(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Returns the top-k most anomalous transactions sorted by rank."""
    if "rank" not in df.columns:
        raise ValueError("DataFrame must be scored first (missing 'rank' column).")
    return df.sort_values("rank").head(k).reset_index(drop=True)


def recall_at_k(df: pd.DataFrame, k: int = 100) -> float:
    """Fraction of planted irregularities appearing in the top-k by rank."""
    if "_planted_pattern" not in df.columns:
        raise ValueError("DataFrame must contain '_planted_pattern' column.")
    planted = df["_planted_pattern"].astype(str).str.len() > 0
    total_planted = int(planted.sum())
    if total_planted == 0:
        return 0.0
    top = df.nsmallest(k, "rank")
    found = (top["_planted_pattern"].astype(str).str.len() > 0).sum()
    return float(found) / float(total_planted)
