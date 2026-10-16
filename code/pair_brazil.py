"""Build Brazil SWOT-gauge pairs (13 Oct 2026). Usage: python3 pair_brazil.py [limit]

For every downloaded station: SWOT D reach series (Hydrocron, data/raw/ana_brazil/swot_reach_D) and ANA 15-min
stage (data/raw/ana_brazil/ana_stage_full). Each valid SWOT pass gets the nearest gauge record within +/- 1 h,
under two readings of ANA timestamps: 'brasilia' (UTC-3 everywhere) and 'local' (legal time of the station's
state: AC -5; AM, RR, RO, MT, MS -4; others -3). Cleaning mirrors clean_v1: Hydrocron fill values -> NaN,
one row per (reach, gauge, cycle, pass); conflicting WSE within a group -> group dropped.
Flags (not filters): hydro_site (UHE/PCH/CGH), dam_or_reservoir, name_status, dist_km, tidal_candidate (median daily
stage range >= 0.3 m), amazon_basin (ANA Bacia 1). Columns are harmonised with swotio for features and label B.
Output: data/processed/brazil_pairs_v1.parquet, logs counts per stage.
"""
import pathlib, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SW = ROOT / "data" / "raw" / "ana_brazil" / "swot_reach_D"
AGF = ROOT / "data" / "raw" / "ana_brazil" / "ana_stage_full"
ZONE = {"AC": -5, "AM": -4, "RR": -4, "RO": -4, "MT": -4, "MS": -4}
HYDRO = ("UHE", "PCH", "CGH")
DAM = ("BARRAMENTO", "RESERVAT", "BARRAGEM")


def load_swot(rid):
    p = SW / f"{int(rid)}.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p)
    if s.empty or "time_str" not in s:
        return None
    s = s[s.time_str != "no_data"].copy()
    for c in s.columns:
        if c not in ("time_str", "crid"):
            s[c] = pd.to_numeric(s[c], errors="coerce")
            s.loc[s[c] <= -1e11, c] = np.nan
    s = s[s.wse.notna()]
    s["t_swot"] = pd.to_datetime(s.time_str, utc=True, errors="coerce").dt.tz_localize(None)
    return s.dropna(subset=["t_swot"])


def load_gauge(code):
    p = AGF / f"{int(code):08d}.parquet"
    if not p.exists():
        return None
    g = pd.read_parquet(p)
    g["t_local"] = pd.to_datetime(g.DataHora, errors="coerce")
    g["stage"] = pd.to_numeric(g.Nivel, errors="coerce") / 100.0
    g = g.dropna(subset=["t_local", "stage"]).drop_duplicates("t_local").sort_values("t_local")
    return g if len(g) else None


def nearest(ts, gt):
    i = np.clip(np.searchsorted(gt, ts), 1, len(gt) - 1)
    left, right = gt[i - 1], gt[i]
    j = np.where(np.abs(ts - left) <= np.abs(right - ts), i - 1, i)
    return j, np.abs(ts - gt[j])


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    m = pd.read_csv(ROOT / "results" / "amazon_v1" / "matches_qc.csv")
    m = m[m.n_swot_D >= 10].drop_duplicates("code").copy()
    m["uf"] = m["Municipio-UF"].astype(str).str.split("-").str[-1].str.strip().str.upper()
    m["zone"] = m.uf.map(ZONE).fillna(-3)
    done = {int(p.stem) for p in AGF.glob("*.done")}
    m = m[m.code.astype(int).isin(done)]
    if limit:
        m = m.head(limit)
    parts, n_noswot, n_nogauge = [], 0, 0
    for _, s in m.iterrows():
        sw, g = load_swot(s.reach_id), load_gauge(s.code)
        if sw is None or sw.empty:
            n_noswot += 1; continue
        if g is None:
            n_nogauge += 1; continue
        drange = g.groupby(g.t_local.dt.date).stage.agg(lambda x: x.max() - x.min()).median()
        gl = (g.t_local.astype("int64") / 1e9).to_numpy()
        ts = (sw.t_swot.astype("int64") / 1e9).to_numpy()
        name = str(s.get("station_name_fixed", "")).upper()
        for mode, off in (("brasilia", -3), ("local", s.zone)):
            gu = gl - off * 3600                                   # gauge clock = UTC + off
            j, dt = nearest(ts, gu)
            ok = dt <= 3600
            p = sw[ok].copy()
            p["gage_stage_m"] = g.stage.to_numpy()[j[ok]]
            p["gage_time_UTC"] = pd.to_datetime(gu[j[ok]], unit="s")
            p["dt_s"], p["tz_mode"] = dt[ok], mode
            p["code"], p["gauge"], p["uf"], p["zone"] = int(s.code), f"ANA:{int(s.code):08d}", s.uf, s.zone
            p["amazon_basin"] = pd.to_numeric(s.Bacia, errors="coerce") == 1
            p["name_status"], p["dist_km"], p["daily_range_m"] = s.name_status, s.dist_km, drange
            p["station_name"], p["ana_river"], p["sword_river"] = s.get("station_name_fixed", ""), s.ana_name_fixed, s.river_name
            p["hydro_site"] = any(k in name for k in HYDRO)
            p["dam_or_reservoir"] = any(k in name for k in DAM)
            parts.append(p)
    d = pd.concat(parts, ignore_index=True)
    d["type"], d["v"], d["phase"] = "reach", "D", "science"
    d["fid"] = d.reach_id.astype("int64")
    d["unit"] = d.fid // 10
    d["qflag"], d["qbit"] = d.reach_q, d.reach_q_b
    d["pdiff"] = (d.wse - d.p_wse).abs()
    d["area_ratio"] = d.area_detct / d.area_total
    d["tidal_candidate"] = d.daily_range_m >= 0.3
    d["tidal_gauge"] = d.tidal_candidate
    d["SWOT_time"] = d.time_str
    key = ["fid", "gauge", "cycle_id", "pass_id", "tz_mode"]
    n0 = len(d)
    d = d[d.groupby(key).wse.transform("nunique") == 1].drop_duplicates(key)
    print(f"stations used {len(m)} (no SWOT {n_noswot}, no gauge {n_nogauge}); rows {n0} -> {len(d)} after overpass dedup")
    for mode, x in d.groupby("tz_mode"):
        print(f"  {mode}: {len(x)} pairs, {x.gauge.nunique()} gauges, {x.fid.nunique()} reaches; "
              f"amazon-basin pairs {int(x.amazon_basin.sum())}; dt_s q50/q90 {np.quantile(x.dt_s, [.5, .9]).round(0).tolist()}; "
              f"flags: hydro {x.hydro_site.mean():.2f}, dam/reservoir {x.dam_or_reservoir.mean():.2f}, "
              f"tidal {x.tidal_candidate.mean():.2f}, name mismatch {(x.name_status == 'mismatch').mean():.2f}")
    out = ROOT / "data" / "processed" / ("brazil_pairs_v1.parquet" if not limit else "brazil_pairs_smoke.parquet")
    for c in d.select_dtypes("object").columns:
        d[c] = d[c].astype("string")
    d.to_parquet(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
