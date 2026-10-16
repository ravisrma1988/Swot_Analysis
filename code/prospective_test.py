"""Prospective protocol test (14 Oct 2026), GPU-bound, tuned model, same unit splits/seeds as final_main.
Version D, tau 0.25, alpha {0.10, 0.20}, reach and node. Everything a forward deployment would know only:
 - features: causal set (current and earlier passes only; features_v2.features_causal)
 - test labels (held-out units, observations from 2025): error relative to each site's offset estimated from data
   before 2025-01-01 (prospective; >= 5 earlier pairs required)
 - calibration / training, two schemes on identical test labels:
     naive       : observations before 2025, labelled against the same-period (pre-2025) offset
     drift-aware : observations in 2024, labelled against offsets from data before 2024-01-01 (lagged, like the test)
Calibration-phase pairs are used only to estimate offsets (and as past passes for causal features).
Outputs: results/audit_response/H_prospective_{per_split.csv, summary.csv}, H_prospective_report.txt
"""
import io, sys, time, warnings
import numpy as np
import pandas as pd
import cupy as cp
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE
from features_v2 import features_causal
from prototype_site_level import evaluate
from run_frontier_v5 import mu_good
from run_robustness_v6 import subset, oof, choose_lam, fact
from final_extras import certify

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "audit_response"
TAU, ALPHAS, M = 0.25, (0.10, 0.20), 3
T24, T25 = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def site_median(dd, mask, keys):
    return dd.where(mask).groupby(keys).transform("median"), pd.Series(mask.astype(int), index=dd.index).groupby(keys).transform("sum")


def main():
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
    rows = []
    for t in ("reach", "node"):
        d = subset(df, t, "D", calval=True)
        keys = [d[c] for c in SITE]
        dd = d.wse - d.gage_stage_m
        tt = d.t_swot
        off24, n24 = site_median(dd, (tt < T24).to_numpy(), keys)
        off25, n25 = site_median(dd, (tt < T25).to_numpy(), keys)
        lab_drift = (dd - off24).to_numpy(); lab_same = (dd - off25).to_numpy(); lab_test = (dd - off25).to_numpy()
        in24 = ((tt >= T24) & (tt < T25)).to_numpy(); pre25 = (tt < T25).to_numpy(); in25 = (tt >= T25).to_numpy()
        sci = (d.phase == "science").to_numpy()
        ok24, ok25 = (n24 >= 5).to_numpy(), (n25 >= 5).to_numpy()
        X = features_causal(d, t).replace([np.inf, -np.inf], np.nan)
        Xg = cp.asarray(X.to_numpy(np.float32))
        unit, site = d.unit.to_numpy(), d.groupby(SITE).ngroup().to_numpy(); units = np.unique(unit)
        bad = {"naive": np.abs(lab_same) > TAU, "drift-aware": np.abs(lab_drift) > TAU}
        bad_test = np.abs(lab_test) > TAU
        rowsel = {"naive": sci & pre25 & ok25, "drift-aware": sci & in24 & ok24}
        test_sel = sci & in25 & ok25
        out(f"{t}: test rows {int(test_sel.sum())} (bad {bad_test[test_sel].mean():.3f}); calibration-label bad rate "
            f"naive {bad['naive'][rowsel['naive']].mean():.3f}, drift-aware {bad['drift-aware'][rowsel['drift-aware']].mean():.3f}; "
            f"median |off25 - off24| {np.nanmedian(np.abs((off25 - off24).to_numpy())):.3f} m")
        for q in range(4):
            folds = np.array_split(np.random.default_rng(5000 * q + 17 * (t == "node")).permutation(units), 5)
            for f in range(5):
                rng = np.random.default_rng(100 * q + f); seed = 1000 * q + 10 * f
                U = np.isin(unit, folds[f]); gg = rng.permutation(np.unique(unit[~U])); h = len(gg) // 2
                inT, inC = np.isin(unit, gg[:h]), np.isin(unit, gg[h:])
                iU = np.nonzero(U & test_sel)[0]; sU = fact(site[iU])
                for scheme in ("naive", "drift-aware"):
                    iT = np.nonzero(inT & rowsel[scheme])[0]; iC = np.nonzero(inC & rowsel[scheme])[0]
                    b = bad[scheme]; y = cp.asarray(b.astype(np.int32))
                    mu_C, mu_U = mu_good(Xg, y, iT, [iC, iU], seed); mu_T = oof(Xg, y, iT, unit[iT], seed, rng)
                    sT, sC = fact(site[iT]), fact(site[iC])
                    lams = np.unique(np.quantile(mu_T, np.linspace(0.02, 0.98, 30)))
                    for a in ALPHAS:
                        lam = choose_lam(mu_T, b[iT], sT, sT.max() + 1, lams, a, M)
                        _, keep = certify(mu_C, b[iC], sC, mu_U, sU, lam, a, "homo", seed)
                        rows.append(dict(type=t, scheme=scheme, alpha=a, rep=q, fold=f, **evaluate(keep, bad_test[iU], sU, sU.max() + 1)))
            out(f"   {t} done at {time.time() - t0:.0f} s")
            pd.DataFrame(rows).to_csv(OUT / "H_prospective_per_split.csv", index=False)
    r = pd.DataFrame(rows); r.to_csv(OUT / "H_prospective_per_split.csv", index=False)
    s = r.groupby(["type", "scheme", "alpha"]).agg(retention=("retention", "mean"), coverage=("site_coverage", "mean"),
                                                   sdr=("site_fdp_kept", "mean"), sd=("site_fdp_kept", "std"), n=("site_fdp_kept", "size"))
    s["se"] = s.sd / np.sqrt(s.n); s["excess_over_2se"] = s.sdr - s.index.get_level_values("alpha") - 2 * s.se
    s.to_csv(OUT / "H_prospective_summary.csv")
    out("\n== prospective protocol (causal features; test labels vs pre-2025 offsets; 2025 observations)\n" + s.round(4).to_string())
    (OUT / "H_prospective_report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
