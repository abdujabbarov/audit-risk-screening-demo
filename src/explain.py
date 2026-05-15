"""SHAP attribution over human-readable features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from . import RANDOM_SEED

FEATURE_NAMES: tuple[str, ...] = (
    "amount_log",
    "amount_to_threshold_ratio",
    "is_single_source",
    "is_end_of_period",
    "vendor_history_count",
    "vendor_is_new",
    "category_method_consistency",
)

_AMOUNT_THRESHOLD = 100_000.0


def _quarter_end_offset(dt: pd.Timestamp) -> int:
    """Days until the next calendar-quarter end, or a large number if not in window."""
    year = dt.year
    quarter_ends = [
        pd.Timestamp(year=year, month=3, day=31),
        pd.Timestamp(year=year, month=6, day=30),
        pd.Timestamp(year=year, month=9, day=30),
        pd.Timestamp(year=year, month=12, day=31),
    ]
    deltas = [(end - dt).days for end in quarter_ends if (end - dt).days >= 0]
    return min(deltas) if deltas else 365


def _build_features(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (feature_matrix, frame_with_named_columns) for attribution."""
    dates = pd.to_datetime(df["date"])

    amount_log = np.log1p(df["amount"].to_numpy().astype(float))
    amount_ratio = np.minimum(df["amount"].to_numpy() / _AMOUNT_THRESHOLD, 2.0)
    is_single_source = (df["procurement_method"] == "Single-source").astype(int).to_numpy()
    is_end_of_period = np.array(
        [1 if _quarter_end_offset(d) < 14 else 0 for d in dates],
        dtype=int,
    )

    vendor_counts = df["vendor_id"].value_counts()
    history = df["vendor_id"].map(vendor_counts).to_numpy().astype(float)
    vendor_is_new = (history < 3).astype(int)

    pair_counts = df.groupby(["category", "procurement_method"]).size()
    category_totals = df.groupby("category").size()
    consistency = np.zeros(len(df), dtype=int)
    for i, row in enumerate(df.itertuples(index=False)):
        pair = pair_counts.get((row.category, row.procurement_method), 0)
        total = category_totals.get(row.category, 1)
        consistency[i] = 1 if (pair / total) >= 0.05 else 0

    features = pd.DataFrame({
        "amount_log": amount_log,
        "amount_to_threshold_ratio": amount_ratio,
        "is_single_source": is_single_source,
        "is_end_of_period": is_end_of_period,
        "vendor_history_count": history,
        "vendor_is_new": vendor_is_new,
        "category_method_consistency": consistency,
    })
    return features.to_numpy(), features


def fit_attribution_model(df_with_scores: pd.DataFrame) -> tuple:
    """Returns (model, feature_names, explainer)."""
    if "anomaly_score" not in df_with_scores.columns:
        raise ValueError("DataFrame must contain 'anomaly_score' from src.score.")

    X, _ = _build_features(df_with_scores)
    threshold = df_with_scores["anomaly_score"].quantile(0.99)
    y = (df_with_scores["anomaly_score"] >= threshold).astype(int).to_numpy()

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X, y)
    explainer = shap.LinearExplainer(model, X)
    return model, FEATURE_NAMES, explainer


def explain_transaction(
    transaction_id: str,
    df: pd.DataFrame,
    model: LogisticRegression,
    explainer: shap.LinearExplainer,
) -> dict[str, float]:
    """Returns {feature_name: shap_value} for one transaction."""
    if transaction_id not in set(df["transaction_id"]):
        raise ValueError(f"Transaction {transaction_id} not found in DataFrame.")

    X, _ = _build_features(df)
    idx = df.index[df["transaction_id"] == transaction_id][0]
    shap_values = explainer.shap_values(X[idx : idx + 1])
    values = np.asarray(shap_values).reshape(-1)
    return {name: float(value) for name, value in zip(FEATURE_NAMES, values)}
