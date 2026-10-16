"""Freeze the cleaned analysis table v1 (rules agreed 13 Oct 2026). Raw data are only read.

 R1 pairing  : drop BOR gauges (daily values); drop any pair with |SWOT_time - gage_time| > 1 h.
 R2 overpass : one row per (type, feature, gauge, cycle, pass, version).
               a) conflicting wse and a PGC0 row present: keep the PGC0 rows;
               b) wse still conflicting: drop the group;
               c) same SWOT observation: keep the gauge's dominant datum series, then a row with EGM2008, then lowest row_id.
 R3 flags    : phase (calval if cycle >= 400), tidal_gauge (any negative flow in raw data), mixed_datum_site,
               shared_swot_obs (one SWOT observation paired with >1 gauge), egm_offset_stable (offset range <= 5 cm).
 R4 values   : fill values (>= 1e11) and negative widths/areas set to NaN; rows kept.
 R5 labels   : eB via swotio.label_B; eA = wse - gauge EGM2008 WSE (native, or node offset transferred to reaches),
               stable gauges only, source in egm_source.
 R6 crosswalk: Version C/D rows of the same overpass (gauge, cycle, pass, |dt| <= 60 s).
Outputs: data/processed/paired_clean_v1.parquet, data/processed/cd_crosswalk_v1.parquet, docs/cleaning_log_v1.md
"""
import datetime, hashlib, subprocess, sys, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE, load_paired, label_B

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
PROC = ROOT / "data" / "processed"; PROC.mkdir(parents=True, exist_ok=True)
K = ["type", "fid", "gauge", "cycle_id", "pass_id", "v"]
LOG = []


def log(s=""):
    print(s); LOG.append(s)


def block(obj):
    log("```"); log(obj.to_string() if hasattr(obj, "to_string") else str(obj)); log("```")


def counts(df):
    return df.groupby(["type", "v"]).size()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def save(df, stem):
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].astype("string")
    try:
        p = PROC / f"{stem}.parquet"; df.to_parquet(p, index=False)
    except Exception as e:
        log(f"parquet failed ({e.__class__.__name__}: {e}); writing csv.gz")
        p = PROC / f"{stem}.csv.gz"; df.to_csv(p, index=False)
    return p


def main():
    r = subprocess.run([sys.executable, str(ROOT / "code" / "verify_usgs_paired.py")], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("raw data failed verification; stopping")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log(f"# Cleaning log v1\n\nRun {stamp} by code/clean_v1.py; raw checksums {r.stdout.strip().splitlines()[-1]}. Rules in the script docstring.\n")
    raw, _ = load_paired()
    steps = [("raw", counts(raw))]

    # inputs for flags and label A, taken from the raw data
    tidal = set(raw.loc[raw.gage_flow_m3s < 0, "gauge"])
    nl = raw[raw.type.isin(["node", "lake"])].assign(off=lambda x: x.gage_wse_EGM2008 - x.gage_stage_m).dropna(subset=["off"])
    og = nl.groupby("gauge").off.agg(["min", "max", "median"])
    og["stable"] = (og["max"] - og["min"]) <= 0.05
    node_og = og[og.index.isin(nl.loc[nl.type == "node", "gauge"]) & og.stable]["median"]

    # R1
    df = raw[raw.gage_agency != "BOR"]; steps.append(("R1 drop BOR gauges", counts(df)))
    df = df[df.dt_s <= 3600].copy(); steps.append(("R1 drop |dt| > 1 h", counts(df)))

    # R2
    df["datum_lbl"] = df.gage_alt_datum.fillna("<empty>")
    dom = df.groupby(["type", "gauge", "v"]).datum_lbl.agg(
        lambda s: s.value_counts().sort_index().sort_values(ascending=False, kind="stable").index[0]).rename("dom_datum")
    df = df.join(dom, on=["type", "gauge", "v"])
    df["is_pgc0"] = df.crid.eq("PGC0")
    g = df.groupby(K)
    conf, has_pgc0 = g.wse.transform("nunique") > 1, g.is_pgc0.transform("max")
    df = df[~(conf & has_pgc0 & ~df.is_pgc0)]; steps.append(("R2a conflict: keep PGC0 rows", counts(df)))
    df = df[~(df.groupby(K).wse.transform("nunique") > 1)]; steps.append(("R2b conflict unresolved: drop group", counts(df)))
    df["is_dom"], df["has_egm"] = df.datum_lbl.eq(df.dom_datum), df.gage_wse_EGM2008.notna()
    df = df.sort_values(K + ["is_dom", "has_egm", "row_id"], ascending=[True] * len(K) + [False, False, True])
    df = df[~df.duplicated(K, keep="first")].copy(); steps.append(("R2c collapse repeated SWOT obs", counts(df)))

    log("## Rows remaining after each rule\n")
    t = pd.concat([s for _, s in steps], axis=1, keys=[n for n, _ in steps]).T
    t["total"] = t.sum(axis=1); block(t)

    # R3
    df["phase"] = np.where(df.cycle_id >= 400, "calval", "science")
    df["tidal_gauge"] = df.gauge.isin(tidal)
    df["mixed_datum_site"] = df.groupby(SITE).datum_lbl.transform("nunique") > 1
    df["shared_swot_obs"] = df.groupby(["type", "fid", "SWOT_time", "v"]).gauge.transform("nunique") > 1
    df["egm_offset_stable"] = df.gauge.map(og.stable).astype("boolean")
    log("\n## Flags (rows flagged; rows are kept)\n")
    fl = df.assign(calval=df.phase.eq("calval"), egm_unstable=df.egm_offset_stable.eq(False).fillna(False)).groupby(["type", "v"])[
        ["calval", "tidal_gauge", "mixed_datum_site", "shared_swot_obs", "egm_unstable"]].sum()
    block(fl)
    log(f"gauges: tidal candidates {df[df.tidal_gauge].gauge.nunique()}, EGM offset unstable {int((~og.stable).sum())} of {len(og)}")

    # R4
    fixes = {}
    for col in ["area_tot_u", "area_det_u"]:
        m = df[col] >= 1e11; fixes[f"{col} >= 1e11"] = int(m.sum()); df.loc[m, col] = np.nan
    for col in ["width", "area_total", "area_detct", "area_wse"]:
        m = df[col] < 0; fixes[f"{col} < 0"] = int(m.sum()); df.loc[m, col] = np.nan
    df["area_ratio"] = df.area_detct / df.area_total
    log(f"\n## Values set to NaN (R4)\n\n{fixes}")

    # R5
    df["eB"], df["offB"] = label_B(df)
    df["n_site"] = df.groupby(SITE).wse.transform("size")
    stable = df.egm_offset_stable.fillna(False).astype(bool)
    native = df.gage_wse_EGM2008.notna() & stable & df.type.ne("reach")
    transfer = df.type.eq("reach") & df.gauge.isin(node_og.index)
    egm = df.gage_wse_EGM2008.where(native)
    egm = egm.where(~transfer, df.gage_stage_m + df.gauge.map(node_og))
    df["egm_source"] = np.select([native, transfer], ["native", "node_transfer"], "none")
    df["eA"] = df.wse - egm
    log("\n## Label A coverage (rows by egm_source)\n")
    block(pd.crosstab([df.type, df.v], df.egm_source))

    # R6
    key = ["type", "gauge", "cycle_id", "pass_id"]
    cols = key + ["row_id", "fid", "wse", "t_swot", "qflag", "eB", "phase"]
    c, d = df[df.v == "C"][cols], df[df.v == "D"][cols]
    log(f"\ncrosswalk key duplicates within version: C {int(c.duplicated(key).sum())}, D {int(d.duplicated(key).sum())}")
    cw = c.merge(d, on=key, suffixes=("_C", "_D"))
    cw["tdiff_s"] = (cw.t_swot_D - cw.t_swot_C).abs().dt.total_seconds()
    cw = cw[cw.tdiff_s <= 60]
    log("C/D matched overpasses by type: " + str(cw.groupby("type").size().to_dict()))

    # checks
    df = df.drop(columns=["is_pgc0", "is_dom", "has_egm"])
    assert not df.duplicated(K).any(), "duplicate overpass rows remain"
    assert (df.dt_s <= 3600).all() and not df.gage_agency.eq("BOR").any(), "pairing rule violated"
    assert df.groupby(SITE).eB.median().abs().max() < 1e-6, "eB not centred per site"
    assert df.row_id.is_unique, "row_id not unique"
    log("\nchecks passed: unique overpass rows, pairing rule, eB centred per site, unique row_id")

    log("\n## Final table\n")
    ok = df.n_site >= 5
    fin = df.groupby(["type", "v"]).agg(rows=("wse", "size"), sites=("n_site", lambda x: np.nan), gauges=("gauge", "nunique"),
                                        features=("fid", "nunique"), science_rows=("phase", lambda x: int((x == "science").sum())))
    fin["sites"] = df.groupby(["type", "v"]).apply(lambda x: x[SITE].drop_duplicates().shape[0])
    fin["share_|eB|>0.25m"] = df[ok].assign(b=df.eB.abs() > 0.25).groupby(["type", "v"]).b.mean().round(3)
    block(fin)

    p1, p2 = save(df, "paired_clean_v1"), save(cw, "cd_crosswalk_v1")
    log("\n## Outputs\n")
    for p in (p1, p2):
        log(f"- {p.relative_to(ROOT)}: {p.stat().st_size} bytes, sha256 {sha256(p)}")
    (ROOT / "docs" / "cleaning_log_v1.md").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
