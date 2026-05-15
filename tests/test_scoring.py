"""Quality checks on the anomaly scoring pipeline."""

from pathlib import Path

from src.embed import compute_embeddings
from src.generate_data import write_dataset
from src.score import recall_at_k, score_transactions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_planted_rows_score_higher_and_recall_meets_threshold(tmp_path):
    csv_path = tmp_path / "synthetic_procurement.csv"
    cache_path = tmp_path / "embeddings.npy"
    df = write_dataset(csv_path)
    embeddings = compute_embeddings(df, cache_path=cache_path, csv_path=csv_path)
    scored = score_transactions(embeddings, df)

    planted_mask = scored["_planted_pattern"].astype(str).str.len() > 0
    mean_planted = scored.loc[planted_mask, "anomaly_score"].mean()
    mean_normal = scored.loc[~planted_mask, "anomaly_score"].mean()
    assert mean_planted > mean_normal

    assert recall_at_k(scored, k=100) >= 0.70
