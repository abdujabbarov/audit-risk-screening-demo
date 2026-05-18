"""Matplotlib figures for the audit risk screening demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import RANDOM_SEED
from .explain import FEATURE_NAMES, explain_transaction

COLORS = {
    "ink":   "#0F172A",
    "blue":  "#0369A1",
    "red":   "#DC2626",
    "green": "#059669",
    "amber": "#D97706",
    "muted": "#64748B",
    "soft":  "#F1F5F9",
    "bg":    "#F8FAFC",
}

CATEGORY_COLORS = {
    "Construction":      "#0369A1",
    "IT services":       "#10B981",
    "Office supplies":   "#8B5CF6",
    "Consulting":        "#D97706",
    "Medical equipment": "#DB2777",
    "Transport":         "#0891B2",
}


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save a figure at 300 DPI with consistent settings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, transparent=False, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_space(embeddings: np.ndarray, df: pd.DataFrame, output_path: Path) -> None:
    """Render the UMAP scatter of the transaction fingerprints."""
    import umap

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_SEED)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    for category, color in CATEGORY_COLORS.items():
        mask = (df["category"] == category).to_numpy()
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=14, c=color, alpha=0.55, label=category, linewidths=0,
        )

    top_mask = df["rank"].to_numpy() <= 30
    ax.scatter(
        coords[top_mask, 0], coords[top_mask, 1],
        s=70, facecolor=COLORS["red"], edgecolor="black",
        linewidths=0.9, label="Top 30 anomalies", zorder=5,
    )

    c004_idx = df.index[df["transaction_id"] == "T-04004"]
    if len(c004_idx) > 0:
        x, y = coords[c004_idx[0]]
        ax.annotate(
            "C004",
            xy=(x, y), xytext=(x + 1.2, y + 1.2),
            fontsize=11, color=COLORS["ink"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["ink"], lw=1.0),
        )

    ax.set_title("What 'similar fingerprints sit close' looks like",
                 fontsize=14, color=COLORS["ink"], pad=12)
    ax.set_xlabel("UMAP-1", color=COLORS["muted"])
    ax.set_ylabel("UMAP-2", color=COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["muted"])
    ax.legend(loc="best", frameon=False, fontsize=9)

    _save_figure(fig, Path(output_path))


def plot_ranked_top20(df_with_scores: pd.DataFrame, output_path: Path) -> None:
    """Horizontal bar chart of the top-20 most anomalous transactions."""
    top = df_with_scores.sort_values("rank").head(20).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 9))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    y_positions = np.arange(len(top))[::-1]
    bar_colors = [
        COLORS["red"] if str(p) else COLORS["blue"]
        for p in top["_planted_pattern"].astype(str)
    ]
    ax.barh(y_positions, top["anomaly_score"], color=bar_colors, edgecolor="none")

    labels = [f"#{r}  {_friendly_label(tid)}" for r, tid in zip(top["rank"], top["transaction_id"])]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9, color=COLORS["ink"])
    ax.set_xlabel("Anomaly score", color=COLORS["muted"])
    ax.set_title("Top 20 flagged transactions", fontsize=14, color=COLORS["ink"], pad=12)
    ax.tick_params(colors=COLORS["muted"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["muted"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["red"], label="Planted irregularity"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["blue"], label="Unplanted"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=9)

    _save_figure(fig, Path(output_path))


def plot_attribution(
    transaction_id: str,
    df: pd.DataFrame,
    model,
    explainer,
    output_path: Path,
) -> None:
    """SHAP bar chart for a single transaction."""
    attributions = explain_transaction(transaction_id, df, model, explainer)
    sorted_items = sorted(attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
    names = [k for k, _ in sorted_items][::-1]
    values = [v for _, v in sorted_items][::-1]

    max_abs = max(abs(v) for v in values) if values else 1.0
    colors = [
        COLORS["red"] if abs(v) >= 0.05 * max_abs and v > 0
        else COLORS["muted"]
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    y = np.arange(len(names))
    ax.barh(y, values, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=COLORS["ink"], fontsize=10)
    ax.axvline(0, color=COLORS["muted"], lw=0.8)
    ax.set_xlabel("SHAP value (contribution to anomaly probability)", color=COLORS["muted"])
    ax.set_title(f"Why was {_friendly_label(transaction_id)} flagged?",
                 fontsize=14, color=COLORS["ink"], pad=12)
    ax.tick_params(colors=COLORS["muted"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["muted"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save_figure(fig, Path(output_path))


def _friendly_label(transaction_id: str) -> str:
    """Map T-04004 to its slide-deck label C004."""
    return "C004" if transaction_id == "T-04004" else transaction_id


def plot_pipeline_diagram(output_path: Path) -> None:
    """Five-box pipeline schematic matching the slide deck."""
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    stages = ["Raw data", "Transformer", "Forest", "Attribute", "Review"]
    palette = [COLORS["muted"], COLORS["blue"], COLORS["green"],
               COLORS["amber"], COLORS["red"]]

    box_w, box_h = 1.6, 1.0
    gap = 0.9
    y = 0.5
    centers = []
    for i, (stage, color) in enumerate(zip(stages, palette)):
        x = i * (box_w + gap) + 0.4
        box = FancyBboxPatch(
            (x, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            linewidth=1.2, edgecolor=color, facecolor="white",
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, y, stage,
                ha="center", va="center",
                fontsize=12, color=color, fontweight="bold")
        centers.append((x + box_w / 2, y, x, x + box_w))

    for i in range(len(centers) - 1):
        _, y0, _, right = centers[i]
        _, y1, left, _ = centers[i + 1]
        arrow = FancyArrowPatch(
            (right + 0.05, y0), (left - 0.05, y1),
            arrowstyle="->", mutation_scale=15,
            color=COLORS["ink"], lw=1.2,
        )
        ax.add_patch(arrow)

    total_width = len(stages) * (box_w + gap)
    ax.set_xlim(0, total_width + 0.2)
    ax.set_ylim(0, 1.4)
    ax.set_axis_off()

    _save_figure(fig, Path(output_path))
