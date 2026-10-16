"""Gauge-only QC for ANA stations (14 Oct 2026): checks on the 15-min stage records alone, no SWOT data involved.

Per station (2023-07-26 .. 2025-12-31): records, flat_frac (share of records inside runs of >= 96 identical consecutive
values, i.e. >= 1 day stuck), jump_rate (|step| > 0.5 m between records <= 30 min apart, per 1000 records),
range_total (max - min stage, m). gauge_qc_pass = flat_frac < 0.2 and jump_rate < 1 and range_total < 30.
Output: results/amazon_v1/gauge_qc.csv
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGF = ROOT / "data" / "raw" / "ana_brazil" / "ana_stage_full"
rows = []
for p in sorted(AGF.glob("*.parquet")):
    g = pd.read_parquet(p)
    t = pd.to_datetime(g.DataHora, errors="coerce"); s = pd.to_numeric(g.Nivel, errors="coerce") / 100.0
    d = pd.DataFrame({"t": t, "s": s}).dropna().drop_duplicates("t").sort_values("t")
    if len(d) < 100:
        rows.append(dict(code=int(p.stem), records=len(d))); continue
    v = d.s.to_numpy(); dt = np.diff(d.t.to_numpy()).astype("timedelta64[s]").astype(float)
    run_id = np.r_[0, np.cumsum(np.diff(v) != 0)]
    run_len = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
    jumps = np.sum((np.abs(np.diff(v)) > 0.5) & (dt <= 1800))
    rows.append(dict(code=int(p.stem), records=len(d), flat_frac=float(np.mean(run_len >= 96)),
                     jump_rate=1000 * jumps / len(d), range_total=float(v.max() - v.min())))
q = pd.DataFrame(rows)
q["gauge_qc_pass"] = (q.flat_frac < 0.2) & (q.jump_rate < 1) & (q.range_total < 30)
q.to_csv(ROOT / "results" / "amazon_v1" / "gauge_qc.csv", index=False)
print(f"stations {len(q)}; QC pass {int(q.gauge_qc_pass.sum())}; flat_frac >= 0.2: {int((q.flat_frac >= 0.2).sum())}; "
      f"jump_rate >= 1: {int((q.jump_rate >= 1).sum())}; range >= 30 m: {int((q.range_total >= 30).sum())}")
