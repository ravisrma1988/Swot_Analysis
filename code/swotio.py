"""Shared loader for the USGS paired SWOT-gage release (Harlan et al. 2026, doi:10.5066/P1FE9W9E).

load_paired() returns one long table (reach, node, lake) with harmonised helper columns:
  type, fid (feature id), gauge (agency:id), v (SWOT version), qflag (summary flag),
  qbit (bitwise flag, NaN for lakes), t_swot, t_gage, dt_s (|SWOT - gauge time| in s),
  pdiff (|wse - prior wse|: SWORD p_wse for rivers, p_ref_wse for lakes),
  area_ratio (area_detct / area_total), row_id (position in the concatenated raw table).
Space-padded 'NA' tokens in numeric columns become NaN; the counts are returned as notes.
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "usgs_paired"
FID = {"reach": "reach_id", "node": "node_id", "lake": "lake_id"}
QF = {"reach": "reach_q", "node": "node_q", "lake": "quality_f"}
QB = {"reach": "reach_q_b", "node": "node_q_b"}
STR_COLS = {"gage_agency", "gage_id", "gage_name", "gage_coord_datum", "gage_alt_datum", "gage_time_UTC",
            "SWOT_time", "collection_shortname", "crid", "SWOT_data_version", "sword_version"}
SITE = ["type", "fid", "gauge", "v"]


def load_paired(types=("reach", "node", "lake")):
    notes, frames = [], []
    for t in types:
        d = pd.read_csv(RAW / f"paired_{t}_SWOT_gage.csv", dtype={"gage_id": str}, low_memory=False)
        for col in d.columns:
            if d[col].dtype == object and col not in STR_COLS:
                num = pd.to_numeric(d[col], errors="coerce")
                bad = d[col].notna() & num.isna()
                notes.append((t, col, int(bad.sum()), d.loc[bad, col].str.strip().value_counts().head(3).to_dict()))
                d[col] = num
        d.insert(0, "type", t)
        d["fid"] = d[FID[t]]
        d["qflag"] = d[QF[t]]
        d["qbit"] = d[QB[t]] if t in QB else np.nan
        frames.append(d)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["v"] = df["SWOT_data_version"]
    df["gauge"] = df["gage_agency"].astype(str) + ":" + df["gage_id"].astype(str)
    df["t_swot"] = pd.to_datetime(df["SWOT_time"], utc=True)
    df["t_gage"] = pd.to_datetime(df["gage_time_UTC"], utc=True)
    df["dt_s"] = (df.t_swot - df.t_gage).abs().dt.total_seconds()
    prior = np.where(df.type == "lake", df.get("p_ref_wse"), df.get("p_wse"))
    df["pdiff"] = (df.wse - prior).abs()
    df["area_ratio"] = df.area_detct / df.area_total
    df["row_id"] = np.arange(len(df))
    return df, notes


def label_B(df):
    """Bias-removed error (primary label, decided 13 Oct 2026).

    eB = d - median_site(d), d = wse - gage_stage_m, site = (type, feature, gauge, version).
    The offset uses every row of the site in `df`, never a screening rule, so labels stay fixed
    while thresholds move. Recompute on the analysis subset actually used (e.g. science phase only).
    Returns (eB, offset).
    """
    d = df.wse - df.gage_stage_m
    off = d.groupby([df[c] for c in SITE]).transform("median")
    return d - off, off
