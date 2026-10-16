"""Data audit v0 for the USGS paired SWOT-gage release (Harlan et al. 2026, doi:10.5066/P1FE9W9E).

Error definitions (decided 13 Oct 2026):
  B (primary)  : eB = d - median_site(d), d = wse - gage_stage_m, site = (type, feature, gauge, version).
                 The offset is fixed from ALL pairs at the site, independent of any screening rule.
  A (secondary): eA = wse - gage_wse_EGM2008 (node and lake only; the reach file has no EGM2008 column).
  U (comparison only): USGS-style, (wse - median wse) - (gauge - median gauge) per site.
Writes results/audit_v0/audit_report.txt plus CSV tables. Reads raw data only.
"""
import io, pathlib, subprocess, sys, warnings
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "usgs_paired"
OUT = ROOT / "results" / "audit_v0"; OUT.mkdir(parents=True, exist_ok=True)
FID = {"reach": "reach_id", "node": "node_id", "lake": "lake_id"}
QF = {"reach": "reach_q", "node": "node_q", "lake": "quality_f"}
SITE = ["type", "fid", "gauge", "v"]
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s); buf.write(s + "\n")


def section(t):
    out("\n======== " + t + " ========")


def qtab(df, col, by, qs=(0.5, 0.68, 0.9, 0.95, 0.99)):
    g = df.groupby(by)[col]
    t = g.quantile(list(qs)).unstack(); t.columns = [f"q{int(q * 100)}" for q in qs]
    t.insert(0, "n", g.size())
    for thr in (0.25, 1.0, 5.0):
        t[f">{thr}m"] = g.apply(lambda s: (s > thr).mean())
    return t.round(3)


def main():
    section("0. checksum verification")
    r = subprocess.run([sys.executable, str(ROOT / "code" / "verify_usgs_paired.py")], capture_output=True, text=True)
    out((r.stdout.strip().splitlines() or [r.stderr])[-1])
    if r.returncode != 0:
        sys.exit("raw data failed verification; stopping")

    section("0b. non-numeric tokens in numeric columns (reported, then set to NaN)")
    STR_COLS = {"gage_agency", "gage_id", "gage_name", "gage_coord_datum", "gage_alt_datum", "gage_time_UTC",
                "SWOT_time", "collection_shortname", "crid", "SWOT_data_version", "sword_version"}
    frames = {}
    for t in FID:
        d = pd.read_csv(RAW / f"paired_{t}_SWOT_gage.csv", dtype={"gage_id": str}, low_memory=False)
        for col in d.columns:
            if d[col].dtype == object and col not in STR_COLS:
                num = pd.to_numeric(d[col], errors="coerce")
                bad = d[col].notna() & num.isna()
                out(f"{t}.{col}: {int(bad.sum())} non-numeric, e.g. {d.loc[bad, col].value_counts().head(5).to_dict()}")
                d[col] = num
        d.insert(0, "type", t); d["fid"] = d[FID[t]]; d["qflag"] = d[QF[t]]
        frames[t] = d
    df = pd.concat(frames.values(), ignore_index=True, sort=False)
    df["v"] = df["SWOT_data_version"]
    df["gauge"] = df["gage_agency"].astype(str) + ":" + df["gage_id"].astype(str)
    df["t_swot"] = pd.to_datetime(df["SWOT_time"], utc=True, errors="coerce")
    df["t_gage"] = pd.to_datetime(df["gage_time_UTC"], utc=True, errors="coerce")
    df["npairs"] = df.groupby(SITE)["wse"].transform("size")

    section("1. size and structure")
    for t, d in frames.items():
        out(f"{t}: {len(d)} rows, exact duplicate rows {int(d.duplicated().sum())}")
    out("unique gauges overall:", df["gauge"].nunique(), "(release text: 2,662 river and lake sites)")
    out(df.groupby("gage_agency")["gauge"].nunique().to_string())
    sites = df.groupby(SITE).size().rename("n").reset_index()
    tab = df.groupby(["type", "v"]).agg(rows=("wse", "size"), gauges=("gauge", "nunique"), features=("fid", "nunique"))
    tab["sites"] = sites.groupby(["type", "v"]).size()
    tab["med_pairs_per_site"] = sites.groupby(["type", "v"])["n"].median()
    tab["share_sites_n<5"] = sites.assign(s=sites.n < 5).groupby(["type", "v"])["s"].mean().round(3)
    out(tab.to_string())
    dup = df.duplicated(["type", "fid", "gauge", "cycle_id", "pass_id", "v"], keep=False)
    out("rows sharing (feature, gauge, cycle, pass, version) with another row:", int(dup.sum()))
    if dup.any():
        out(df[dup].groupby(["type", "v"])["crid"].value_counts().to_string())
    multi = df.groupby(["type", "fid", "SWOT_time", "v"])["gauge"].transform("nunique") > 1
    out("rows whose SWOT observation is paired with >1 gauge:", int(multi.sum()))

    section("2. time pairing, cycles, product versions")
    out("unparsed timestamps: swot", int(df.t_swot.isna().sum()), "gage", int(df.t_gage.isna().sum()))
    dt = (df.t_swot - df.t_gage).abs().dt.total_seconds()
    out("|SWOT_time - gage_time| s: q50 %.0f q90 %.0f max %.0f; >3600 s: %d"
        % (dt.quantile(.5), dt.quantile(.9), dt.max(), int((dt > 3600).sum())))
    df["cyc_bin"] = pd.cut(df.cycle_id, [0, 99, 399, 10000], labels=["1-99", "100-399", ">=400"])
    out(df.groupby(["v", "cyc_bin"], observed=True).agg(rows=("wse", "size"), first=("t_swot", "min"), last=("t_swot", "max")).to_string())
    out(df.groupby(["v", "crid"]).agg(rows=("wse", "size"), first=("t_swot", "min"), last=("t_swot", "max")).to_string())

    section("3. datum coverage (EGM2008) and offset constancy")
    nl = df[df.type.isin(["node", "lake"])].copy()
    nl["off"] = nl.gage_wse_EGM2008 - nl.gage_stage_m
    out(nl.groupby("type")["off"].agg(rows="size", share_rows_with_EGM=lambda s: s.notna().mean()).round(3).to_string())
    have = nl.groupby(["type", "gauge"])["off"].apply(lambda s: s.notna().any())
    out("share of gauges with any EGM2008 value:"); out(have.groupby(level=0).mean().round(3).to_string())
    g = nl.dropna(subset=["off"]).groupby(["type", "gauge"])["off"]
    out("per-gauge range (max - min) of offset EGM2008 - stage, m:")
    out((g.max() - g.min()).groupby(level=0).describe(percentiles=[.5, .9, .99]).round(4).to_string())
    never = have[~have].index.get_level_values(1)
    out("vertical datum of never-converted gauges (top 8):")
    out(nl[nl.gauge.isin(never)].drop_duplicates("gauge")["gage_alt_datum"].fillna("<empty>").value_counts().head(8).to_string())
    node_off = nl[nl.type == "node"].dropna(subset=["off"]).groupby("gauge")["off"].median()
    rr = df[df.type == "reach"]
    out("reach gauges with an offset available from the node file: %.3f of gauges, %.3f of reach rows"
        % (rr.gauge.drop_duplicates().isin(node_off.index).mean(), rr.gauge.isin(node_off.index).mean()))

    section("4. error definitions: B (primary), U (USGS-style), A (absolute)")
    df["d"] = df.wse - df.gage_stage_m
    df["absB"] = (df.d - df.groupby(SITE)["d"].transform("median")).abs()
    df["absU"] = ((df.wse - df.groupby(SITE)["wse"].transform("median"))
                  - (df.gage_stage_m - df.groupby(SITE)["gage_stage_m"].transform("median"))).abs()
    df["absA"] = (df.wse - df.gage_wse_EGM2008).abs()
    ok = df.npairs >= 5
    out(f"error tables use sites with >=5 pairs ({ok.mean():.3f} of rows); singleton sites have B = U = 0 by construction")
    for name in ("absB", "absU", "absA"):
        t = qtab(df[ok & df[name].notna()], name, ["type", "v"])
        out(f"-- |{name[3:]}| in m"); out(t.to_string()); t.to_csv(OUT / f"err_{name[3:]}_by_type_version.csv")
    t = qtab(df[ok], "absB", ["type", "v", "qflag"])
    out("-- |B| by summary quality flag (0 good, 1 suspect, 2 degraded, 3 bad)"); out(t.to_string())
    t.to_csv(OUT / "err_B_by_qflag.csv")

    section("5. does wse_u track B errors?")
    s = df[ok & df.wse_u.notna()]
    r5 = s.groupby(["type", "v"]).apply(lambda x: pd.Series({
        "frac_absB<=wse_u": (x.absB <= x.wse_u).mean(),
        "spearman(wse_u,absB)": x[["wse_u", "absB"]].corr(method="spearman").iloc[0, 1],
        "median_wse_u": x.wse_u.median()}))
    out(r5.round(3).to_string())

    section("6. same overpass in Version C and Version D")
    key = ["type", "fid", "gauge", "cycle_id", "pass_id"]
    cols = key + ["wse", "t_swot", "qflag", "absB"]
    c = df[df.v == "C"][cols].drop_duplicates(key)
    d = df[df.v == "D"][cols].drop_duplicates(key)
    m = c.merge(d, on=key, suffixes=("_C", "_D"))
    m["dt"] = (m.t_swot_D - m.t_swot_C).abs().dt.total_seconds()
    m["dw"] = (m.wse_D - m.wse_C).abs()
    summ = m.groupby("type").agg(matched=("dw", "size"), dt_q99_s=("dt", lambda x: x.quantile(.99)),
                                 dwse_q50=("dw", "median"), dwse_q90=("dw", lambda x: x.quantile(.9)))
    summ["share_of_C"] = summ.matched / c.groupby("type").size()
    summ["share_of_D"] = summ.matched / d.groupby("type").size()
    out(summ.round(3).to_string())
    mr = m[m.type == "reach"]
    out("reach_q in C (rows) vs D (columns) for matched overpasses:")
    out(pd.crosstab(mr.qflag_C, mr.qflag_D).to_string())

    section("7. implausible and fill values")
    checks = {"width < 0": df.width < 0, "area_total < 0": df.area_total < 0,
              "area_tot_u >= 1e11 (fill)": df.area_tot_u >= 1e11, "wse_u missing": df.wse_u.isna(),
              "|B| > 100 m": df.absB > 100, "gage_flow_m3s < 0": df.gage_flow_m3s < 0}
    for k_, msk in checks.items():
        out(f"{k_}: {int(msk.sum())} rows {df[msk].groupby('type').size().to_dict()}")
    out("gauges with any negative flow (tidal or reversal candidates):", df[df.gage_flow_m3s < 0].gauge.nunique())

    section("8. USGS filtering grid (fidelity anchor preview)")
    fa = pd.read_csv(RAW / "paired_filtering_analysis.csv")
    for f_ in ["wse_pdiff", "dark_frac", "outlier_flag", "ice_clim_f", "waterbody_size", "overall_quality",
               "bitwise_q", "xovr_cal_q", "xtrk_flag", "partial_f", "area_frac"]:
        vals = fa[f_].drop_duplicates().sort_values().tolist()
        out(f"{f_}: {len(vals)} values {vals[:10]}")
    out(fa.groupby(["type", "SWOT_data_version"]).size().to_string())
    top = fa.loc[fa.groupby(["type", "SWOT_data_version"]).percent_remaining.idxmax()]
    out("max-retention grid row per type and version:"); out(top.to_string(index=False))
    out("our unfiltered pooled q68 of |U| over ALL rows (no filters, singletons included):")
    out(df.groupby(["type", "v"]).absU.quantile(0.68).round(4).to_string())

    (OUT / "audit_report.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nreport -> {OUT / 'audit_report.txt'}")


if __name__ == "__main__":
    main()
