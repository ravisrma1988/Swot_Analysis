"""Brazil node-level pairs (14 Oct 2026): Hydrocron SWOT_L2_HR_RiverSP_D node series for each station's matched node,
paired with the same ANA 15-min records as the reach pairs (pair_brazil.py logic: nearest record within 1 h,
'brasilia' and 'local' readings, overpass dedup, same flags). Output: data/processed/brazil_pairs_nodes_v1.parquet
"""
import concurrent.futures as cf, json, pathlib, sys, time, urllib.parse
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from amazon_download import http, log, HYDROCRON
from pair_brazil import load_gauge, nearest, ZONE, HYDRO, DAM

ROOT = pathlib.Path(__file__).resolve().parents[1]
SWN = ROOT / "data" / "raw" / "ana_brazil" / "swot_node_D"; SWN.mkdir(parents=True, exist_ok=True)
FULL = ("node_id,reach_id,time_str,cycle_id,pass_id,wse,wse_u,wse_r_u,width,width_u,area_total,area_tot_u,area_detct,"
        "area_det_u,area_wse,node_q,node_q_b,dark_frac,ice_clim_f,n_good_pix,xovr_cal_q,xtrk_dist,p_wse,p_width,p_dam_id,"
        "p_n_ch_mod,crid")
CORE = "node_id,reach_id,time_str,cycle_id,pass_id,wse,wse_u,width,node_q,dark_frac,xtrk_dist,p_wse,p_width"


def url(nid, fields):
    q = {"feature": "Node", "feature_id": str(int(nid)), "start_time": "2023-07-26T00:00:00Z", "end_time": "2025-12-31T23:59:59Z",
         "output": "csv", "fields": fields, "collection_name": "SWOT_L2_HR_RiverSP_D"}
    return HYDROCRON + "?" + urllib.parse.urlencode(q)


def get_node(nid, fields):
    p = SWN / f"{int(nid)}.csv"
    if p.exists():
        return True
    code, body = http(url(nid, fields)); time.sleep(0.3)
    if code != 200:
        return False
    p.write_text(json.loads(body)["results"]["csv"], encoding="utf-8")
    return True


def load_node(nid):
    p = SWN / f"{int(nid)}.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p)
    if s.empty or "time_str" not in s:
        return None
    s = s[s.time_str != "no_data"].copy()
    for c in s.columns:
        if c not in ("time_str", "crid"):
            s[c] = pd.to_numeric(s[c], errors="coerce"); s.loc[s[c] <= -1e11, c] = np.nan
    s = s[s.wse.notna()]
    s["t_swot"] = pd.to_datetime(s.time_str, utc=True, errors="coerce").dt.tz_localize(None)
    return s.dropna(subset=["t_swot"])


def main():
    m = pd.read_csv(ROOT / "results" / "amazon_v1" / "matches_qc.csv")
    m = m[m.n_swot_D >= 10].drop_duplicates("code").copy()
    m["uf"] = m["Municipio-UF"].astype(str).str.split("-").str[-1].str.strip().str.upper()
    m["zone"] = m.uf.map(ZONE).fillna(-3)
    nids = m.node_id.unique()
    code, body = http(url(nids[0], FULL))
    fields = FULL if code == 200 else CORE
    log(f"{len(nids)} nodes; Hydrocron field probe HTTP {code}; using {'FULL' if fields == FULL else 'CORE'}"
        + ("" if code == 200 else f"; server said {body[:200]!r}"))
    with cf.ThreadPoolExecutor(4) as ex:
        ok = list(ex.map(lambda n: get_node(n, fields), nids))
    log(f"node series saved: {sum(ok)}/{len(nids)}")
    parts = []
    for _, s in m.iterrows():
        sw, g = load_node(s.node_id), load_gauge(s.code)
        if sw is None or sw.empty or g is None:
            continue
        drange = g.groupby(g.t_local.dt.date).stage.agg(lambda x: x.max() - x.min()).median()
        gl = (g.t_local.astype("int64") / 1e9).to_numpy(); ts = (sw.t_swot.astype("int64") / 1e9).to_numpy()
        name = str(s.get("station_name_fixed", "")).upper()
        for mode, off in (("brasilia", -3), ("local", s.zone)):
            gu = gl - off * 3600
            j, dt = nearest(ts, gu); okm = dt <= 3600
            p = sw[okm].copy()
            p["gage_stage_m"], p["dt_s"], p["tz_mode"] = g.stage.to_numpy()[j[okm]], dt[okm], mode
            p["code"], p["gauge"], p["uf"], p["zone"] = int(s.code), f"ANA:{int(s.code):08d}", s.uf, s.zone
            p["amazon_basin"] = pd.to_numeric(s.Bacia, errors="coerce") == 1
            p["name_status"], p["dist_km"], p["daily_range_m"] = s.name_status, s.dist_km, drange
            p["hydro_site"] = any(k in name for k in HYDRO); p["dam_or_reservoir"] = any(k in name for k in DAM)
            parts.append(p)
    d = pd.concat(parts, ignore_index=True)
    d["type"], d["v"], d["phase"] = "node", "D", "science"
    d["fid"] = d.node_id.astype("int64"); d["unit"] = d.fid // 10000
    d["qflag"], d["qbit"] = d.node_q, d.get("node_q_b", np.nan)
    d["pdiff"] = (d.wse - d.p_wse).abs(); d["area_ratio"] = d.area_detct / d.area_total
    d["tidal_candidate"] = d.daily_range_m >= 0.3; d["tidal_gauge"] = d.tidal_candidate; d["SWOT_time"] = d.time_str
    key = ["fid", "gauge", "cycle_id", "pass_id", "tz_mode"]
    d = d[d.groupby(key).wse.transform("nunique") == 1].drop_duplicates(key)
    for c in d.select_dtypes("object").columns:
        d[c] = d[c].astype("string")
    d.to_parquet(ROOT / "data" / "processed" / "brazil_pairs_nodes_v1.parquet", index=False)
    log(f"node pairs saved: {len(d)} rows; brasilia {int((d.tz_mode == 'brasilia').sum())}, gauges {d.gauge.nunique()}, nodes {d.fid.nunique()}")


if __name__ == "__main__":
    main()
