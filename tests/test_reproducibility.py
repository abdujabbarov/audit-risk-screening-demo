"""End-to-end reproducibility check on the top-20 ranking."""

from src.embed import compute_embeddings
from src.generate_data import write_dataset
from src.score import get_top_k, score_transactions


def _run_pipeline(tmp_dir):
    csv_path = tmp_dir / "synthetic_procurement.csv"
    cache_path = tmp_dir / "embeddings.npy"
    df = write_dataset(csv_path)
    embeddings = compute_embeddings(df, cache_path=cache_path, csv_path=csv_path)
    scored = score_transactions(embeddings, df)
    return get_top_k(scored, k=20)["transaction_id"].tolist()


def test_top20_identical_across_two_runs(tmp_path):
    run_a = _run_pipeline(tmp_path / "a")
    run_b = _run_pipeline(tmp_path / "b")
    assert run_a == run_b
