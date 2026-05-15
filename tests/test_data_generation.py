"""Shape and content checks on the generated dataset."""

from src.generate_data import generate_dataset

EXPECTED_COLUMNS = {
    "transaction_id", "date", "ministry", "vendor_id", "vendor_name",
    "category", "amount", "payment_terms_days", "procurement_method",
    "contract_description", "_planted_pattern",
}


def test_row_count_and_columns():
    df = generate_dataset()
    assert len(df) == 10_000
    assert set(df.columns) == EXPECTED_COLUMNS


def test_exactly_thirty_planted_irregularities():
    df = generate_dataset()
    planted = df[df["_planted_pattern"].astype(str).str.len() > 0]
    assert len(planted) == 30

    counts = planted["_planted_pattern"].value_counts().to_dict()
    assert counts["A_threshold_avoidance"] == 10
    assert counts["B_new_vendor_concentration"] == 8
    assert counts["C_end_of_period_burst"] == 7
    assert counts["D_round_number_bias"] == 5


def test_c004_is_planted_and_targets_threshold():
    df = generate_dataset()
    c004 = df[df["transaction_id"] == "T-04004"].iloc[0]
    assert c004["_planted_pattern"] == "B_new_vendor_concentration"
    assert c004["amount"] == 99_500
    assert c004["vendor_id"] == "V-NEW-001"
