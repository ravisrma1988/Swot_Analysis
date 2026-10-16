"""Data audit v1: duplicates, >1 h pairs, C/D re-match, and decoding + reproducing the USGS filter grid.

A. rows sharing (feature, gauge, cycle, pass, version): what differs between them
B. pairs with |SWOT_time - gage_time| > 1 h: where they come from
C. same overpass in Version C and D, matched on gauge + cycle + pass + time (feature ids differ across SWORD versions)
D. grid decoding: for each filter, does moving one level change percent_remaining up, down or not at all
E. reproduction of single-factor grid rows (one filter tightened, all others relaxed), two NaN conventions:
   N1 = a NaN filter variable fails the filter even at an infinite threshold (R/dplyr behaviour); N2 = NaN passes.
No cleaning is applied here; outputs go to results/audit_v1.
"""
import io, sys, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, RAW, SITE, load_paired

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
OUT = ROOT / "results" / "audit_v1"; OUT.mkdir(parents=True, exist_ok=True)
buf = io.StringIO()
FILTERS = ["wse_pdiff", "dark_frac", "outlier_flag", "ice_clim_f", "waterbody_size", "overall_quality",
           "bitwise_q", "xovr_cal_q", "xtrk_flag", "partial_f", "area_frac"]
RELAXED = dict(wse_pdiff=np.inf, dark_frac=np.inf, outlier_flag=1, ice_clim_f=2, waterbody_size=1,
               overall_quality=3, bitwise_q=np.inf, xovr_cal_q=2, xtrk_flag=1, partial_f=1, area_frac=0.0)


def out(*a):
    s = " ".join(str(x) for x in a); print(s); buf.write(s + "\n")


def section(t):
    out("\n======== " + t + " ========")


def part_A(df):
    section("A. rows sharing (feature, gauge, cycle, pass, version)")
    key = ["type", "fid", "gauge", "cycle_id", "pass_id", "v"]
    dd = df[df.duplicated(key, keep=False)]
    g = dd.groupby(key)
    a = g.agg(n=("wse", "size"), n_wse=("wse", "nunique"), n_ts=("SWOT_time", "nunique"),
              n_tg=("gage_time_UTC", "nunique"), n_crid=("crid", "nunique"), t0=("t_swot", "min"), t1=("t_swot", "max"))
    a["crids"] = g.crid.agg(lambda s: "+".join(sorted(set(s))))
    a["span_h"] = (a.t1 - a.t0).dt.total_seconds() / 3600
    a["cat"] = np.select([(a.n_ts == 1) & (a.n_wse == 1), (a.n_ts == 1) & (a.n_wse > 1), a.n_ts > 1],
                         ["same time, same wse", "same time, diff wse", "diff SWOT times"], "other")
    a = a.reset_index()
    out(f"{len(dd)} rows in {len(a)} groups; group size counts {a.n.value_counts().to_dict()}")
    out(pd.crosstab([a.type, a.v], a.cat).to_string())
    for c_ in a.cat.unique():
        s = a[a.cat == c_]
        out(f"-- {c_}: crid combos {s.crids.value_counts().head(6).to_dict()}; distinct gauge times per group {s.n_tg.value_counts().head(4).to_dict()}")
        if c_ == "diff SWOT times":
            out("   span between SWOT times (h): " + s.span_h.quantile([.1, .5, .9, 1]).round(2).to_string().replace("\n", "; "))
    a.to_csv(OUT / "duplicate_groups.csv", index=False)


def part_B(df):
    section("B. pairs with |SWOT_time - gage_time| > 1 h")
    big = df.dt_s > 3600
    out(f"{int(big.sum())} rows; counts by agency x type:")
    out(pd.crosstab(df.gage_agency, df.type, values=big, aggfunc="sum").fillna(0).astype(int).to_string())
    out("share of each agency's rows:", df.assign(b=big).groupby("gage_agency").b.mean().round(4).to_dict())
    out("gap (h) quantiles q10 q50 q90 max:", (df.loc[big, "dt_s"] / 3600).quantile([.1, .5, .9, 1]).round(2).tolist())
    out("share of these with gauge time exactly 00:00:00 UTC:", round(df.loc[big, "t_gage"].dt.strftime("%H:%M:%S").eq("00:00:00").mean(), 3))
    gs = df.assign(b=big).groupby("gauge").b.agg(["sum", "mean"])
    gs = gs[gs["sum"] > 0]
    out(f"gauges with any such row: {len(gs)}; of these, share whose rows are ALL >1 h: {(gs['mean'] == 1).mean():.3f}")
    out("top gauges:", gs.sort_values("sum", ascending=False).head(8).round(3).to_dict("index"))


def part_C(df):
    section("C. same overpass in C and D, matched on gauge + cycle + pass, |dt| <= 60 s")
    end_overlap = df.loc[(df.v == "D") & (df.crid == "PGD0"), "t_swot"].max()
    key = ["type", "gauge", "cycle_id", "pass_id"]
    cols = key + ["fid", "wse", "t_swot", "qflag"]
    c = df[df.v == "C"][cols].drop_duplicates(key)
    d = df[df.v == "D"][cols].drop_duplicates(key)
    m = c.merge(d, on=key, suffixes=("_C", "_D"))
    m["tdiff"] = (m.t_swot_D - m.t_swot_C).abs().dt.total_seconds()
    out(f"overlap window ends {end_overlap}; candidate matches {len(m)}, dropped for |dt| > 60 s: {int((m.tdiff > 60).sum())}")
    m = m[m.tdiff <= 60]
    m["dw"] = (m.wse_D - m.wse_C).abs()
    cw = c[c.t_swot <= end_overlap].groupby("type").size()
    s = m.groupby("type").agg(matched=("dw", "size"), dw_q50=("dw", "median"),
                              dw_q90=("dw", lambda x: x.quantile(.9)), dw_q99=("dw", lambda x: x.quantile(.99)))
    s["same_fid"] = m.assign(sf=m.fid_C == m.fid_D).groupby("type").sf.mean()
    s["share_of_C_in_overlap"] = s.matched / cw
    out(s.round(3).to_string())
    for t in ("reach", "node"):
        mt = m[m.type == t]
        out(f"{t}: summary flag C (rows) vs D (cols)"); out(pd.crosstab(mt.qflag_C, mt.qflag_D).to_string())
    m.to_csv(OUT / "cd_matched_overpasses.csv", index=False)


def part_D(fa):
    section("D. grid decoding: change in percent_remaining when one filter moves one level up, others fixed")
    rows = []
    for f in FILTERS:
        others = [c for c in FILTERS if c != f] + ["type", "SWOT_data_version"]
        p = fa.pivot_table(index=others, columns=f, values="percent_remaining", aggfunc="first")
        lv = sorted(p.columns)
        for a, b in zip(lv[:-1], lv[1:]):
            diff = p[b] - p[a]
            for t in ("reach", "node", "lake"):
                x = diff[diff.index.get_level_values("type") == t]
                rows.append(dict(filter=f, step=f"{a}->{b}", type=t, up=(x > 1e-9).mean(),
                                 equal=(x.abs() <= 1e-9).mean(), down=(x < -1e-9).mean()))
    r = pd.DataFrame(rows)
    out(r.round(3).to_string(index=False))
    r.to_csv(OUT / "grid_directions.csv", index=False)


def keep_mask(d, s, nan_fails):
    def thr(x, lim, ge=False):
        m = (x >= lim) if ge else (x <= lim)
        return m if nan_fails else (m | x.isna())
    t = d.type.iloc[0]
    m = thr(d.pdiff, s["wse_pdiff"]) & thr(d.dark_frac, s["dark_frac"]) & thr(d.area_ratio, s["area_frac"], ge=True)
    m &= thr(d.qflag, s["overall_quality"]) & thr(d.xovr_cal_q, s["xovr_cal_q"])
    if t != "lake":
        m &= thr(d.qbit, s["bitwise_q"])
    m &= thr(d.ice_clim_f.where(d.ice_clim_f != -999, 0), s["ice_clim_f"])  # -999 = no ice data, kept as ice-free
    if s["xtrk_flag"] == 0:
        ax = d.xtrk_dist.abs(); m &= (ax >= 10000) & (ax <= 60000)
    if s["partial_f"] == 0 and t != "node":
        m &= d.partial_f == 0
    return m


def stats(k):
    """USGS-style metrics on kept rows: medians and correlations recomputed per site after filtering."""
    g = k.groupby(SITE)
    u = ((k.wse - g.wse.transform("median")) - (k.gage_stage_m - g.gage_stage_m.transform("median"))).abs()
    dx = k.wse - g.wse.transform("mean"); dy = k.gage_stage_m - g.gage_stage_m.transform("mean")
    s = pd.DataFrame({"xy": dx * dy, "xx": dx ** 2, "yy": dy ** 2, "n": 1}).groupby([k[c] for c in SITE]).sum()
    s = s[s.n >= 3]
    r = s.xy / np.sqrt(s.xx * s.yy)
    return u.quantile(.68), u.median(), u.mean(), r.median(), (r ** 2).median()


def part_E(df, fa):
    section("E. reproduce single-factor grid rows (one filter tightened, others relaxed)")
    idx = fa.set_index(FILTERS + ["type", "SWOT_data_version"]).sort_index()
    settings = [("relaxed", {})]
    for f, levels in [("overall_quality", [0, 1, 2]), ("bitwise_q", [0.0, 262144.0, 4194304.0]), ("xovr_cal_q", [0, 1]),
                      ("dark_frac", [0.25, 0.5]), ("wse_pdiff", [10.0, 50.0]), ("area_frac", [0.5, 0.75]),
                      ("ice_clim_f", [0, 1]), ("xtrk_flag", [0]), ("partial_f", [0])]:
        settings += [(f"{f}={l}", {f: l}) for l in levels]
    rows = []
    for (t, v), d in df.groupby(["type", "v"]):
        for name, ov in settings:
            s = {**RELAXED, **ov}
            try:
                gr = idx.loc[tuple(s[f] for f in FILTERS) + (t, v)]
            except KeyError:
                out(f"no grid row for {t} {v} {name}"); continue
            if isinstance(gr, pd.DataFrame):
                gr = gr.iloc[0]
            row = dict(type=t, v=v, setting=name, pr_grid=gr.percent_remaining, q68_grid=gr.abs_WSE_diff_68,
                       med_grid=gr.median_abs_WSE_diff, mean_grid=gr.mean_abs_WSE_diff, R2_grid=gr.median_R2)
            for tag, nf in (("N1", True), ("N2", False)):
                k = d[keep_mask(d, s, nf)]
                row[f"pr_{tag}"] = 100 * len(k) / len(d)
                if len(k):
                    q68, med, mean, r, r2 = stats(k)
                    row.update({f"q68_{tag}": q68, f"med_{tag}": med, f"mean_{tag}": mean, f"r_{tag}": r, f"r2_{tag}": r2})
            rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "grid_reproduction_single_factor.csv", index=False)
    show = ["type", "v", "setting", "pr_grid", "pr_N1", "pr_N2", "q68_grid", "q68_N1", "q68_N2",
            "med_grid", "med_N1", "mean_grid", "mean_N1", "R2_grid", "r_N1", "r2_N1"]
    out(r[show].round(3).to_string(index=False))


def main():
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "code" / "verify_usgs_paired.py")], capture_output=True, text=True)
    out("checksums:", (r.stdout.strip().splitlines() or [r.stderr])[-1])
    if r.returncode != 0:
        sys.exit("raw data failed verification; stopping")
    df, notes = load_paired()
    out("NA tokens converted:", [(t, c, n) for t, c, n, _ in notes])
    part_A(df); part_B(df); part_C(df)
    fa = pd.read_csv(RAW / "paired_filtering_analysis.csv")
    part_D(fa); part_E(df, fa)
    (OUT / "audit_v1_report.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nreport -> {OUT / 'audit_v1_report.txt'}")


if __name__ == "__main__":
    main()
