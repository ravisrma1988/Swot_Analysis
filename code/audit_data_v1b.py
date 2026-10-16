"""Audit v1b: loose ends from v1 before fixing cleaning rules.

1. In duplicate groups with the same SWOT time and wse, which columns differ?
2. BOR gauges: time-of-day of gauge stamps and size of the pairing gaps.
3. Missingness of filter variables by type and version (what the NaN-fails rule removes).
4. percent_remaining / q68 variants against the grid: dedup of repeated SWOT observations,
   dropping >1 h pairs, and SKIP (a filter at its relaxed level is not applied, so NaN passes).
"""
import io, sys, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, RAW, SITE, load_paired
from audit_data_v1 import FILTERS, RELAXED, stats

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40)
OUT = ROOT / "results" / "audit_v1"
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s); buf.write(s + "\n")


def keep(d, s, mode):
    t = d.type.iloc[0]
    def thr(x, f, ge=False):
        if mode == "SKIP" and s[f] == RELAXED[f]:
            return pd.Series(True, index=d.index)
        return (x >= s[f]) if ge else (x <= s[f])
    m = thr(d.pdiff, "wse_pdiff") & thr(d.dark_frac, "dark_frac") & thr(d.area_ratio, "area_frac", ge=True)
    m &= thr(d.qflag, "overall_quality") & thr(d.xovr_cal_q, "xovr_cal_q")
    if t != "lake":
        m &= thr(d.qbit, "bitwise_q")
    m &= thr(d.ice_clim_f.where(d.ice_clim_f != -999, 0), "ice_clim_f")
    if s["xtrk_flag"] == 0:
        ax = d.xtrk_dist.abs(); m &= (ax >= 10000) & (ax <= 60000)
    if s["partial_f"] == 0 and t != "node":
        m &= d.partial_f == 0
    return m


HELPER = {"type", "fid", "qflag", "qbit", "v", "gauge", "t_swot", "t_gage", "dt_s", "pdiff", "area_ratio", "row_id"}


def main():
    df, _ = load_paired()
    raw_cols = [c for c in df.columns if c not in HELPER]

    out("== 1. columns that differ within duplicate groups having one SWOT time and one wse (share of groups)")
    key = ["type", "fid", "gauge", "cycle_id", "pass_id", "v"]
    dd = df[df.duplicated(key, keep=False)]
    g = dd.groupby(key)
    ds = dd[(g.SWOT_time.transform("nunique") == 1) & (g.wse.transform("nunique") == 1)]
    share = (ds.groupby(key)[raw_cols].nunique(dropna=False) > 1).mean().sort_values(ascending=False)
    out(share[share > 0].round(4).to_string() if (share > 0).any() else "none: rows identical in every raw column")

    out("\n== 2. BOR gauges")
    b = df[df.gage_agency == "BOR"]
    out("gauge time-of-day (UTC), top 6:", b.t_gage.dt.strftime("%H:%M").value_counts().head(6).to_dict())
    out("gap shares: <=1 h %.3f, <=12 h %.3f, <=24 h %.3f, >48 h %.3f"
        % tuple((b.dt_s <= x).mean() if x > 0 else (b.dt_s > -x).mean() for x in (3600, 43200, 86400, -172800)))

    out("\n== 3. missing share of filter variables")
    cols = ["pdiff", "dark_frac", "area_ratio", "qflag", "qbit", "xovr_cal_q", "ice_clim_f", "xtrk_dist", "partial_f", "p_ref_wse"]
    out(df.groupby(["type", "v"])[cols].apply(lambda x: x.isna().mean()).round(3).to_string())

    out("\n== 4. grid reproduction variants (percent_remaining against the RAW row count of each type x version)")
    fa = pd.read_csv(RAW / "paired_filtering_analysis.csv")
    idx = fa.set_index(FILTERS + ["type", "SWOT_data_version"]).sort_index()
    settings = [("relaxed", {}), ("overall_quality=1", {"overall_quality": 1}), ("xovr_cal_q=0", {"xovr_cal_q": 0}),
                ("dark_frac=0.5", {"dark_frac": 0.5}), ("wse_pdiff=10", {"wse_pdiff": 10.0}), ("ice_clim_f=0", {"ice_clim_f": 0})]
    variants = {"N1": ("N1", 0, 0), "N1+dedup": ("N1", 1, 0), "N1+gap": ("N1", 0, 1), "N1+dedup+gap": ("N1", 1, 1),
                "SKIP+gap": ("SKIP", 0, 1), "SKIP+dedup+gap": ("SKIP", 1, 1)}
    rows = []
    for (t, v), d0 in df.groupby(["type", "v"]):
        for vn, (mode, dedup, gap) in variants.items():
            d = d0[d0.dt_s <= 3600] if gap else d0
            if dedup:
                d = d.drop_duplicates(["type", "fid", "gauge", "v", "SWOT_time"])
            for sn, ov in settings:
                s = {**RELAXED, **ov}
                gr = idx.loc[tuple(s[f] for f in FILTERS) + (t, v)]
                k = d[keep(d, s, mode)]
                rows.append(dict(type=t, v=v, setting=sn, variant=vn, pr=100 * len(k) / len(d0), pr_grid=gr.percent_remaining,
                                 q68=stats(k)[0] if len(k) else np.nan, q68_grid=gr.abs_WSE_diff_68))
    r = pd.DataFrame(rows)
    r["pr_err"] = r.pr - r.pr_grid
    r["q68_err"] = r.q68 - r.q68_grid
    r.to_csv(OUT / "grid_variants.csv", index=False)
    summ = r.groupby(["type", "variant"]).agg(pr_err_mean=("pr_err", "mean"), pr_err_maxabs=("pr_err", lambda x: x.abs().max()),
                                              q68_err_mean=("q68_err", "mean"), q68_err_maxabs=("q68_err", lambda x: x.abs().max()))
    out(summ.round(3).to_string())
    (OUT / "audit_v1b_report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
