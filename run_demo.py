"""End-to-end demo runner. Generates data, scores, explains, and plots."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import pandas as pd

from src.embed import compute_embeddings
from src.explain import fit_attribution_model
from src.generate_data import write_dataset
from src.score import recall_at_k, score_transactions
from src.report import build_interactive_report
from src.visualize import (
    plot_attribution,
    plot_embedding_space,
    plot_pipeline_diagram,
    plot_ranked_top20,
)

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "synthetic_procurement.csv"
EMBEDDING_CACHE = ROOT / "data" / "embeddings.npy"
FIGURES_DIR = ROOT / "figures"


def _step(idx: int, total: int, title: str) -> None:
    """Print a consistent progress header."""
    print(f"[{idx}/{total}] {title}")


def main() -> None:
    """Run the full pipeline end-to-end with progress printed to stdout."""
    started = time.perf_counter()
    total_steps = 6

    _step(1, total_steps, "Generating synthetic dataset...")
    df = write_dataset(DATA_CSV)
    pattern_counts = Counter(p for p in df["_planted_pattern"] if p)
    pattern_summary = ", ".join(
        f"Pattern {label[0]}: {pattern_counts[label]}"
        for label in sorted(pattern_counts)
    )
    print(f"      ✓ {len(df):,} transactions written to {DATA_CSV.relative_to(ROOT)}")
    print(f"      ✓ {sum(pattern_counts.values())} irregularities planted ({pattern_summary})")

    _step(2, total_steps, "Computing transformer embeddings...")
    if not EMBEDDING_CACHE.exists() or EMBEDDING_CACHE.stat().st_mtime < DATA_CSV.stat().st_mtime:
        print("      Loading sentence-transformers/all-MiniLM-L6-v2...")
    embeddings = compute_embeddings(df, cache_path=EMBEDDING_CACHE, csv_path=DATA_CSV)
    print(f"      ✓ {embeddings.shape[0]:,} × {embeddings.shape[1]} feature matrix "
          f"cached to {EMBEDDING_CACHE.relative_to(ROOT)}")

    _step(3, total_steps, "Scoring with Isolation Forest...")
    scored = score_transactions(embeddings, df)
    planted_mask = scored["_planted_pattern"].astype(str).str.len() > 0
    mean_planted = scored.loc[planted_mask, "anomaly_score"].mean()
    mean_normal = scored.loc[~planted_mask, "anomaly_score"].mean()
    r100 = recall_at_k(scored, k=100)
    caught = int(round(r100 * planted_mask.sum()))
    total_planted = int(planted_mask.sum())
    print(f"      ✓ Mean score for planted rows: {mean_planted:.2f}")
    print(f"      ✓ Mean score for normal rows:  {mean_normal:.2f}")
    print(f"      ✓ Recall @ top-100: {r100:.2f} ({caught}/{total_planted} planted irregularities found)")

    _step(4, total_steps, "Computing SHAP attribution for top-20...")
    model, _, explainer = fit_attribution_model(scored)
    print("      ✓ Attribution computed")

    _step(5, total_steps, "Generating figures...")
    fig1 = FIGURES_DIR / "01_embedding_space.png"
    fig2 = FIGURES_DIR / "02_ranked_top20.png"
    fig3 = FIGURES_DIR / "03_attribution_C004.png"
    fig4 = FIGURES_DIR / "04_pipeline_diagram.png"

    plot_embedding_space(embeddings, scored, fig1)
    print(f"      ✓ {fig1.relative_to(ROOT)}")
    plot_ranked_top20(scored, fig2)
    print(f"      ✓ {fig2.relative_to(ROOT)}")
    plot_attribution("T-04004", scored, model, explainer, fig3)
    print(f"      ✓ {fig3.relative_to(ROOT)}")
    plot_pipeline_diagram(fig4)
    print(f"      ✓ {fig4.relative_to(ROOT)}")

    _step(6, total_steps, "Building interactive HTML report...")
    report_path = FIGURES_DIR / "report.html"
    build_interactive_report(scored, model, explainer, report_path)
    print(f"      ✓ {report_path.relative_to(ROOT)}")

    elapsed = time.perf_counter() - started
    print(f"\nDone. Total runtime: {elapsed:.0f}s.")


if __name__ == "__main__":
    main()
