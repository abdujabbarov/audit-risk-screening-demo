"""Build a standalone interactive HTML report for auditor review."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .explain import FEATURE_NAMES, _build_features

_FRIENDLY: dict[str, str] = {"T-04004": "C004"}


def _label(tid: str) -> str:
    return _FRIENDLY.get(tid, tid)


def _safe(val) -> object:
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return None if math.isnan(float(val)) else float(val)
    return val


def _build_report_data(
    df_scored: pd.DataFrame,
    model,
    explainer,
    top_n: int = 20,
) -> list[dict]:
    X, _ = _build_features(df_scored)
    raw_shap = explainer.shap_values(X)
    all_shap = np.array(raw_shap)
    if all_shap.ndim == 3:
        all_shap = all_shap[1]

    top = df_scored.sort_values("rank").head(top_n)
    rows = []
    for _, row in top.iterrows():
        tid = str(row["transaction_id"])
        iloc_idx = df_scored.index.get_loc(row.name)
        shap_vals = {
            name: round(float(all_shap[iloc_idx, i]), 5)
            for i, name in enumerate(FEATURE_NAMES)
        }

        planted_raw = row.get("_planted_pattern", "")
        is_planted = (
            not pd.isna(planted_raw)
            and str(planted_raw).strip() not in ("", "nan")
        )

        rows.append({
            "transaction_id": _label(tid),
            "rank": int(row["rank"]),
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "is_planted": is_planted,
            "fields": {
                "vendor":      _safe(row.get("vendor_name", "")),
                "category":    _safe(row.get("category", "")),
                "amount":      round(float(_safe(row.get("amount", 0)) or 0), 2),
                "date":        str(_safe(row.get("date", ""))),
                "method":      _safe(row.get("procurement_method", "")),
                "ministry":    _safe(row.get("ministry", "")),
                "description": _safe(row.get("contract_description", "")),
            },
            "shap": shap_vals,
        })
    return rows


_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Audit Risk Screening</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #F1F5F9; color: #0F172A; }
  header { padding: 18px 32px; background: #0F172A; }
  header h1 { font-size: 18px; font-weight: 700; color: #F8FAFC; letter-spacing: -.01em; }
  header p  { font-size: 12px; color: #94A3B8; margin-top: 3px; }
  .layout { display: flex; gap: 16px; padding: 20px 32px; align-items: flex-start; }
  .panel  { background: white; border-radius: 10px; padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .left  { flex: 0 0 400px; }
  .right { flex: 1; min-height: 200px; }
  .hint  { color: #94A3B8; font-size: 13px; text-align: center; padding: 60px 0; }
  .tx-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
  .tx-id   { font-size: 16px; font-weight: 700; color: #0F172A; }
  .tx-badge { background: #FEF2F2; color: #DC2626; border-radius: 20px;
              padding: 3px 10px; font-size: 11px; font-weight: 600; }
  .meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }
  .meta-item { background: #F8FAFC; border-radius: 6px; padding: 8px 12px;
               border: 1px solid #E2E8F0; }
  .meta-label { font-size: 10px; text-transform: uppercase; letter-spacing: .06em;
                color: #94A3B8; }
  .meta-value { font-size: 13px; font-weight: 500; color: #0F172A;
                margin-top: 2px; word-break: break-word; }
  .desc-box { background: #F8FAFC; border-radius: 6px; padding: 10px 12px;
              border: 1px solid #E2E8F0; font-size: 12px; color: #475569;
              line-height: 1.55; margin-bottom: 16px; }
  .section-title { font-size: 12px; font-weight: 600; text-transform: uppercase;
                   letter-spacing: .06em; color: #64748B; margin-bottom: 8px; }
  .legend { display: flex; gap: 16px; justify-content: flex-end;
            font-size: 11px; color: #64748B; margin-top: 6px; }
  .legend-dot { display: inline-block; width: 10px; height: 10px;
                border-radius: 2px; margin-right: 4px; vertical-align: middle; }
</style>
</head>
<body>
<header>
  <h1>Audit Risk Screening &#8212; Top 20 Review</h1>
  <p>Click any bar to inspect transaction details and feature contributions.</p>
</header>
<div class="layout">
  <div class="panel left">
    <div id="bar-chart"></div>
    <div class="legend">
      <span><span class="legend-dot" style="background:#DC2626"></span>Planted irregularity</span>
      <span><span class="legend-dot" style="background:#0369A1"></span>Unplanted</span>
    </div>
  </div>
  <div class="panel right" id="right-panel">
    <p class="hint">&#8592; Select a transaction to inspect it</p>
  </div>
</div>

<script>
const RECORDS = __DATA_JSON__;

const labels = RECORDS.map(r => "#" + r.rank + "  " + r.transaction_id);
const scores = RECORDS.map(r => r.anomaly_score);
const colors = RECORDS.map(r => r.is_planted ? "#DC2626" : "#0369A1");

const barTrace = {
  type: "bar", orientation: "h",
  x: scores,
  y: labels,
  marker: { color: colors, opacity: 0.85 },
  hovertemplate: "<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>",
};

const barLayout = {
  title: { text: "Top 20 flagged transactions", font: { size: 13, color: "#0F172A" } },
  margin: { l: 120, r: 16, t: 38, b: 40 },
  xaxis: { title: { text: "Anomaly score", font: { size: 11 } }, range: [0, 1.05],
           tickfont: { size: 10 } },
  yaxis: { automargin: true, tickfont: { size: 10 } },
  paper_bgcolor: "white", plot_bgcolor: "white",
  showlegend: false,
  height: 500,
};

const barDiv = document.getElementById("bar-chart");
Plotly.newPlot(barDiv, [barTrace], barLayout, { displayModeBar: false, responsive: true });

let selectedIdx = null;

barDiv.on("plotly_click", function(ev) {
  const idx = ev.points[0].pointIndex;
  selectedIdx = idx;

  // Highlight selected bar
  const updated = RECORDS.map((r, i) => i === idx ? "#F59E0B" : (r.is_planted ? "#DC2626" : "#0369A1"));
  Plotly.restyle(barDiv, { "marker.color": [updated] }, [0]);

  renderDetail(RECORDS[idx]);
});

function fmt(val) {
  if (typeof val === "number") return val.toLocaleString("en-US", { maximumFractionDigits: 0 });
  return val || "&#8212;";
}

function renderDetail(rec) {
  const f = rec.fields;
  const panel = document.getElementById("right-panel");

  const meta = [
    ["Rank",        "#" + rec.rank],
    ["Anomaly score", rec.anomaly_score.toFixed(4)],
    ["Vendor",      f.vendor],
    ["Ministry",    f.ministry],
    ["Category",    f.category],
    ["Amount",      "$ " + fmt(f.amount)],
    ["Date",        f.date],
    ["Method",      f.method],
  ];

  const metaHtml = meta.map(([label, value]) =>
    `<div class="meta-item"><div class="meta-label">${label}</div>` +
    `<div class="meta-value">${value}</div></div>`
  ).join("");

  panel.innerHTML =
    `<div class="tx-header">` +
      `<span class="tx-id">${rec.transaction_id}</span>` +
      `<span class="tx-badge">Score: ${rec.anomaly_score.toFixed(4)}</span>` +
    `</div>` +
    `<div class="section-title">Transaction details</div>` +
    `<div class="meta-grid">${metaHtml}</div>` +
    `<div class="desc-box">${f.description || "&#8212;"}</div>` +
    `<div class="section-title">Why was this flagged?</div>` +
    `<div id="shap-chart"></div>`;

  const shap = rec.shap;
  const sorted = Object.entries(shap).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const names  = sorted.map(d => d[0]).reverse();
  const values = sorted.map(d => d[1]).reverse();
  const barColors = values.map(v => v > 0 ? "#DC2626" : "#64748B");

  Plotly.newPlot("shap-chart", [{
    type: "bar", orientation: "h",
    x: values, y: names,
    marker: { color: barColors, opacity: 0.85 },
    hovertemplate: "%{y}: %{x:.5f}<extra></extra>",
  }], {
    margin: { l: 210, r: 16, t: 10, b: 40 },
    xaxis: { title: { text: "SHAP value (contribution to anomaly score)", font: { size: 11 } },
             zeroline: true, zerolinecolor: "#CBD5E1", zerolinewidth: 1.5,
             tickfont: { size: 10 } },
    yaxis: { automargin: true, tickfont: { size: 11 } },
    paper_bgcolor: "white", plot_bgcolor: "white",
    showlegend: false,
    height: 260,
  }, { displayModeBar: false, responsive: true });
}
</script>
</body>
</html>
"""


def build_interactive_report(
    df_scored: pd.DataFrame,
    model,
    explainer,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Write a self-contained interactive HTML report to output_path."""
    records = _build_report_data(df_scored, model, explainer, top_n)
    json_str = json.dumps(records, ensure_ascii=False).replace("</script>", r"<\/script>")
    html = _HTML.replace("__DATA_JSON__", json_str)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
