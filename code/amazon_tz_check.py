"""Empirical time-zone check of ANA legacy telemetry timestamps (13 Oct 2026).

Tidal stations near the Amazon mouth / Para estuary (lat -3 to 1.5, lon -56 to -47) with >= 10 SWOT D passes:
SWOT node WSE (Hydrocron, node matched in amazon_step1) vs ANA 15-min stage around each overpass.
For clock offsets h in [-8, 8] (gauge clock = UTC + h), stage is interpolated at SWOT time; the robust spread
(median absolute deviation) of WSE - stage is computed; on tidal rivers the true offset minimises it sharply.
Outputs: results/amazon_v1/tz_check.csv, tz_check_report.txt
"""
import io, json, sys, time, urllib.parse, urllib.request
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from amazon_step1 import records, ANA, HYDROCRON, OUT

buf = io.StringIO()
OFFSETS = np.arange(-8, 8.001, 0.25)


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception:
            time.sleep(5 * (k + 1))
    return None


def node_series(node_id):
    q = {"feature": "Node", "feature_id": str(int(node_id)), "start_time": "2023-07-26T00:00:00Z",
         "end_time": "2025-12-31T23:59:59Z", "output": "csv", "fields": "node_id,time_str,wse,node_q",
         "collection_name": "SWOT_L2_HR_RiverSP_D"}
    raw = get(HYDROCRON + "?" + urllib.parse.urlencode(q))
    if raw is None:
        return pd.DataFrame()
    rows = [l.split(",") for l in json.loads(raw)["results"]["csv"].strip().splitlines()[1:]]
    df = pd.DataFrame([r[:4] for r in rows if len(r) >= 4 and r[1] != "no_data"], columns=["node_id", "t", "wse", "q"])
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df.t, utc=True).dt.tz_localize(None)
    df["wse"], df["q"] = df.wse.astype(float), df.q.astype(float)
    return df[(df.wse > -1e10) & (df.q <= 2)]


def ana_window(code, day):
    a, b = (day - pd.Timedelta(days=1)).strftime("%d/%m/%Y"), (day + pd.Timedelta(days=1)).strftime("%d/%m/%Y")
    raw = get(f"{ANA}/DadosHidrometeorologicos?codEstacao={int(code):08d}&DataInicio={a}&DataFim={b}")
    if raw is None:
        return pd.DataFrame()
    df = pd.DataFrame(records(raw))
    if df.empty or "Nivel" not in df:
        return pd.DataFrame()
    df["t"] = pd.to_datetime(df.DataHora, errors="coerce")
    df["stage"] = pd.to_numeric(df.Nivel, errors="coerce") / 100.0
    return df.dropna(subset=["t", "stage"])[["t", "stage"]]


def gaps(tq, gt):
    i = np.clip(np.searchsorted(gt, tq), 1, len(gt) - 1)
    return np.minimum(np.abs(tq - gt[i - 1]), np.abs(gt[i] - tq))


def main():
    m = pd.read_csv(OUT / "matches.csv")
    cand = (m[(m.n_swot_D >= 10) & m.lat.between(-3, 1.5) & m.lon.between(-56, -47)]
            .sort_values("n_swot_D", ascending=False).drop_duplicates("code").head(20))
    out(f"candidate stations (estuary box, >= 10 SWOT D passes): {len(cand)}")
    rows = []
    for _, s in cand.iterrows():
        sw = node_series(s.node_id)
        if len(sw) < 8:
            out(f"  {int(s.code):08d}: only {len(sw)} valid SWOT node obs, skipped"); continue
        sw = sw.sort_values("t").head(40).reset_index(drop=True)
        parts = []
        for d in sw.t.dt.normalize().unique():
            parts.append(ana_window(s.code, pd.Timestamp(d))); time.sleep(0.3)
        g = pd.concat(parts) if parts else pd.DataFrame()
        if g.empty:
            out(f"  {int(s.code):08d}: no ANA stage around the overpasses, skipped"); continue
        g = g.drop_duplicates("t").sort_values("t")
        drange = g.groupby(g.t.dt.date).stage.agg(lambda x: x.max() - x.min()).median()
        gt = (g.t.astype("int64") / 1e9).to_numpy(); gs = g.stage.to_numpy()
        ts = (sw.t.astype("int64") / 1e9).to_numpy()
        spread = []
        for h in OFFSETS:
            tq = ts + h * 3600
            ok = gaps(tq, gt) <= 1800
            if ok.sum() < 6:
                spread.append(np.nan); continue
            dlt = sw.wse.to_numpy()[ok] - np.interp(tq[ok], gt, gs)
            spread.append(float(np.median(np.abs(dlt - np.median(dlt)))))
        spread = np.array(spread)
        if np.all(np.isnan(spread)):
            out(f"  {int(s.code):08d}: no overlap at any offset, skipped"); continue
        at = lambda h: spread[np.argmin(np.abs(OFFSETS - h))]
        best = OFFSETS[np.nanargmin(spread)]
        rows.append(dict(code=int(s.code), river=s.NomeRio, n_swot=len(sw), daily_range_m=drange, best_offset_h=best,
                         mad_best=np.nanmin(spread), mad_utc_minus3=at(-3), mad_utc_minus4=at(-4), mad_utc=at(0)))
        out(f"  {int(s.code):08d} {str(s.NomeRio)[:20]:20s} daily range {drange:.2f} m; best offset UTC{best:+.2f} h "
            f"(MAD {np.nanmin(spread):.3f}); MAD at UTC-3 {at(-3):.3f}, UTC-4 {at(-4):.3f}, UTC {at(0):.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "tz_check.csv", index=False)
    if not res.empty:
        tidal = res[res.daily_range_m >= 0.5]
        out(f"\ntidal stations (median daily range >= 0.5 m): {len(tidal)}; best offsets: "
            f"{tidal.best_offset_h.value_counts().sort_index().to_dict()}")
    (OUT / "tz_check_report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
