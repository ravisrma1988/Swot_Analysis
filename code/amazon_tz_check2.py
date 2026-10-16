"""ANA time-zone discrimination, check 2 (13 Oct 2026): Brasilia time (UTC-3) everywhere, or local legal time?

The estuary check ruled out UTC and supported UTC-3, but its tidal stations lie where local time = Brasilia time.
Here: stations in UTC-4 states (AM, RR, RO, MT, MS; western AM is UTC-5, treated as -4) and UTC-5 (AC), with the
largest sub-daily stage swings in the downloaded 15-min data (e.g. regulated rivers). For each, SWOT node WSE
(Hydrocron) vs ANA stage interpolated at SWOT time + offset, offsets -8..+8 h; robust spread (MAD) per offset.
If ANA uses Brasilia time, these stations also favour -3; if local time, they favour -4 / -5.
Uses only downloaded ANA files (no extra ANA requests). Outputs: results/amazon_v1/tz_check2.csv, tz_check2_report.txt
"""
import io, pathlib, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from amazon_tz_check import node_series, gaps, OFFSETS

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "amazon_v1"
AG = ROOT / "data" / "raw" / "ana_brazil" / "ana_stage_full"
ZONE = {"AC": -5, "AM": -4, "RR": -4, "RO": -4, "MT": -4, "MS": -4}
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def main():
    m = pd.read_csv(OUT / "matches_qc.csv")
    m["uf"] = m["Municipio-UF"].astype(str).str.split("-").str[-1].str.strip().str.upper()
    m["zone"] = m.uf.map(ZONE).fillna(-3)
    cand = []
    for _, s in m[m.zone < -3].iterrows():
        f = AG / f"{int(s.code):08d}.parquet"
        if not f.exists():
            continue
        g = pd.read_parquet(f)
        g["t"] = pd.to_datetime(g.DataHora, errors="coerce")
        g["stage"] = pd.to_numeric(g.Nivel, errors="coerce") / 100.0
        g = g.dropna(subset=["t", "stage"]).drop_duplicates("t").sort_values("t")
        if len(g) < 500:
            continue
        rng = g.groupby(g.t.dt.date).stage.agg(lambda x: x.max() - x.min()).median()
        cand.append((rng, s, g))
    cand.sort(key=lambda x: -x[0])
    out(f"stations in UTC-4/-5 states with downloaded stage: {len(cand)}; top daily ranges (m): "
        f"{[round(c[0], 2) for c in cand[:15]]}")
    rows = []
    for rng, s, g in [c for c in cand if c[0] >= 0.15][:15]:
        sw = node_series(s.node_id); time.sleep(0.3)
        if len(sw) < 8:
            continue
        gt = (g.t.astype("int64") / 1e9).to_numpy(); gs = g.stage.to_numpy()
        ts = (sw.t.astype("int64") / 1e9).to_numpy(); ws = sw.wse.to_numpy()
        spread = []
        for h in OFFSETS:
            tq = ts + h * 3600; ok = gaps(tq, gt) <= 1800
            if ok.sum() < 6:
                spread.append(np.nan); continue
            d = ws[ok] - np.interp(tq[ok], gt, gs)
            spread.append(float(np.median(np.abs(d - np.median(d)))))
        spread = np.array(spread)
        if np.all(np.isnan(spread)):
            continue
        at = lambda h: spread[np.argmin(np.abs(OFFSETS - h))]
        best = OFFSETS[np.nanargmin(spread)]
        rows.append(dict(code=int(s.code), uf=s.uf, zone=s.zone, river=s.ana_name_fixed, daily_range_m=rng, n_swot=len(sw),
                         best_offset_h=best, mad_m3=at(-3), mad_m4=at(-4), mad_m5=at(-5), mad_0=at(0)))
        out(f"  {int(s.code):08d} {s.uf} {str(s.ana_name_fixed)[:18]:18s} range {rng:.2f} m: best UTC{best:+.2f}; "
            f"MAD -3: {at(-3):.3f}, -4: {at(-4):.3f}, -5: {at(-5):.3f}, 0: {at(0):.3f}")
    res = pd.DataFrame(rows); res.to_csv(OUT / "tz_check2.csv", index=False)
    if not res.empty:
        res["favours"] = np.where(res.mad_m3 < res[["mad_m4", "mad_m5"]].min(axis=1), "Brasilia (-3)", "local (-4/-5)")
        out("\nverdict per station:", res.favours.value_counts().to_dict())
    (OUT / "tz_check2_report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
