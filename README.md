# Audit Risk Screening Demo

A companion demo for the talk **"Transformer-Based Risk Scoring: A Tool for Prioritizing Large Transaction Populations"** (Young Auditors Conference 2026, Tashkent).

## What this is

A working demonstration of how a modern AI model can rank transactions from most-suspicious to least-suspicious, so reviewers focus their attention on the cases most likely to matter. The pipeline uses a pretrained transformer (the same family as ChatGPT) to turn each transaction into a fingerprint, then a statistical method to identify the loners.

In plain terms, the demo:

1. Builds a realistic synthetic procurement dataset (10,000 transactions, 24 months, 6 ministries) and quietly plants 30 known-bad transactions in four irregularity patterns.
2. Turns every transaction into a numeric fingerprint by combining a transformer reading of its text columns with its numeric attributes.
3. Asks an Isolation Forest which fingerprints look most unlike the rest, and orders them from most to least suspicious.
4. For the top-ranked cases, explains *why* they were flagged using SHAP attribution over plain-English features (amount, vendor history, procurement method, and so on).
5. Packages the ranked list and all explanations into a self-contained interactive HTML report — no server or Python install needed to review it.

## What this is NOT

This is not production audit software. It runs on synthetic data with planted irregularities, demonstrates the methodology, and provides a starting template for adapting to real data. Treat its output as a prioritised review queue, not as a verdict.

## Requirements

- Python 3.10, 3.11, or 3.12
- About 200 MB free disk (the transformer model weighs ~80 MB and is downloaded on first run)
- An internet connection on first run only (subsequent runs work offline)

## How to run

Three commands:

```bash
pip install -r requirements.txt
python run_demo.py
jupyter notebook notebook.ipynb
```

`run_demo.py` regenerates the dataset, computes embeddings, scores anomalies, runs the attribution, writes four figures into `figures/`, and produces an interactive HTML report at `figures/report.html`. Total runtime is roughly 1–2 minutes on a standard laptop, CPU only — no GPU required.

The notebook walks through the same pipeline with explanations between the steps.

## What you'll see

After `python run_demo.py` finishes, look in `figures/`:

- **`01_embedding_space.png`** — every transaction projected onto a 2D map. Similar transactions sit close together; the top 30 anomalies are marked in red. Transaction **C004** (the slide-deck example) is annotated.
- **`02_ranked_top20.png`** — the 20 most-suspicious transactions, ranked. Bars in red are the planted irregularities — these are the ones a reviewer would want to inspect first.
- **`03_attribution_C004.png`** — for the highlighted transaction C004, the SHAP chart shows which features pushed it up the ranking (amount close to the 100,000 threshold, single-source method, brand-new vendor).
- **`04_pipeline_diagram.png`** — the five-stage flow: raw data → transformer → forest → attribute → review.
- **`report.html`** — interactive review tool. Open in any browser: click a bar in the top-20 chart to see that transaction's details and SHAP explanation side by side. Self-contained — no server or Python needed.

## Repository layout

```
audit-risk-screening-demo/
├── README.md                       # this file
├── requirements.txt
├── run_demo.py                     # one-command end-to-end runner
├── notebook.ipynb                  # narrated walk-through
├── data/
│   └── synthetic_procurement.csv   # generated dataset (committed)
├── figures/                        # generated PNGs (committed)
├── src/
│   ├── generate_data.py            # synthetic data generator
│   ├── embed.py                    # transformer + numeric features
│   ├── score.py                    # Isolation Forest
│   ├── explain.py                  # SHAP attribution
│   ├── visualize.py                # the four static figures
│   └── report.py                   # interactive HTML report
└── tests/                          # pytest suite (run with `pytest tests/`)
```

## How to adapt to your own data

The pipeline expects a CSV with one transaction per row. To swap in your own data:

1. **Match the column names.** The pipeline expects: `transaction_id`, `date` (ISO format), `ministry` (or any organisational unit), `vendor_id`, `vendor_name`, `category`, `amount`, `payment_terms_days`, `procurement_method`, `contract_description`. Rename your columns to match, or edit `src/embed.py` and `src/explain.py` to point at yours.
2. **Drop the `_planted_pattern` column** if you don't have known-bad rows — it is only used for evaluation here.
3. **Save your CSV** to `data/synthetic_procurement.csv` (or change the path in `run_demo.py`).
4. **Re-run** `python run_demo.py`. The transformer doesn't need retraining; it works on any text. The Isolation Forest re-fits on your data automatically.

### Runtime expectations

| Rows       | Approx. runtime (laptop, CPU) |
|------------|-------------------------------|
| 10,000     | 1–2 minutes                   |
| 100,000    | 10–15 minutes                 |
| 1,000,000  | A few hours; consider running on a workstation or splitting by period |

The transformer encode step dominates runtime. A GPU shortens it dramatically, but is not required.

## Tests

```bash
pytest tests/
```

Three tests verify dataset shape, scoring quality (recall at top-100 ≥ 0.70), and end-to-end reproducibility (the top-20 ranking is identical across two runs with the same seed).

## Credits and references

- Built as a companion demo to the talk at the **Young Auditors Conference 2026, Tashkent**.
- Underlying methods:
  - Liu, Ting & Zhou (2008), *Isolation Forest*
  - Reimers & Gurevych (2019), *Sentence-BERT* / `sentence-transformers`
  - Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions* (SHAP)

## License

MIT.
