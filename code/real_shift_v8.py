"""Real-shift experiment v8 (13 Oct 2026), GPU-bound: do site-level guarantees transfer across real, labelled shifts
inside the USGS release, with and without SWORD-based weights?

Version D rivers, tau = 0.25 m, alpha in {0.10, 0.20}, 20 repeats per design. Designs (unit level):
  US->CA (source USGS, target ECCC + Quebec), CA->US, S->N55 (source south of 55 N, target north),
  E->W110 (source east of 110 W, target west). Location uses SWORD reach centroids (nodes: parent reach).
Each repeat: source units 50/50 into T (score + lambda via 3-fold OOF) and C (calibration); target units 50/50 into
  'fit' (unlabelled covariates for the density ratio) and 'eval' (labels used only for evaluation).
Weights: XGBoost (CUDA) classifier source-T vs target-fit on SWORD covariates (log width, prior WSE, lat, lon, channels,
  obstruction, log length, log flow accumulation, slope); applied to C and eval sites.
Methods: S6 unweighted and S6w weighted SCoRE-SDR (prune 'homo'), GPU ports verified against the authors' code.
Outputs: results/real_shift_v8/{per_split.csv, summary.csv, report.txt}
"""
import io, sys, time, warnings
import numpy as np
import pandas as pd
import cupy as cp
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE
from features_v2 import features_v2, twin_map
from prototype_site_level import evaluate
from prototype_score_sdr import site_scores
from score_sdr_gpu import score_sdr_gpu, score_sdr_w_gpu
from run_frontier_v5 import mu_good
from run_robustness_v6 import subset, oof, choose_lam, fact
from run_shift_v7 import ratio_weights, ess
import sword_support as ss

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 300)
OUT = ROOT / "results" / "real_shift_v8"; OUT.mkdir(parents=True, exist_ok=True)
TAU, ALPHAS, REPEATS, M = 0.25, (0.10, 0.20), 20, 3
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def site_table(d, t, sw):
    a = d.groupby(SITE).agg(unit=("unit", "first"), fid=("fid", "first"), agency=("gage_agency", "first"))
    fid = a.fid.astype("int64")
    a["reach_id"] = fid if t == "reach" else (fid // 10000) * 10 + fid % 10
    a = a.reset_index(drop=True).merge(sw, on="reach_id", how="left")
    return a


DESIGNS = {
    "US->CA": (lambda u: u.agency == "USGS", lambda u: u.agency.isin(["ECCC", "Quebec"])),
    "CA->US": (lambda u: u.agency.isin(["ECCC", "Quebec"]), lambda u: u.agency == "USGS"),
    "S->N55": (lambda u: u.y < 55, lambda u: u.y >= 55),
    "E->W110": (lambda u: u.x >= -110, lambda u: u.x < -110),
}


def main():
    t0 = time.time()
    r = ss.load_na()
    cov = ss.covariates(r).reset_index(drop=True)
    COV = list(cov.columns)
    sw = pd.concat([r[["reach_id", "x", "y"]].reset_index(drop=True), cov], axis=1)
    df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
    twin = twin_map(df, pd.read_parquet(ROOT / "data" / "processed" / "cd_crosswalk_v1.parquet"))
    rows = []
    for t in ("reach", "node"):
        d = subset(df, t, "D")
        X = features_v2(d, t, twin.loc[d.index]).replace([np.inf, -np.inf], np.nan)
        Xg = cp.asarray(X.to_numpy(np.float32))
        bad = (d.eB.abs() > TAU).to_numpy(); y = cp.asarray(bad.astype(np.int32))
        site = d.groupby(SITE).ngroup().to_numpy(); unit = d.unit.to_numpy()
        st = site_table(d, t, sw)                                   # row k = site id k
        Zs = st[COV].to_numpy(float)
        ut = st.groupby("unit").agg(agency=("agency", "first"), x=("x", "first"), y=("y", "first"))
        out(f"{t}: {len(st)} sites, SWORD match {st.x.notna().mean():.3f}; agencies {ut.agency.value_counts().to_dict()}")
        for di, (name, (fsrc, ftgt)) in enumerate(DESIGNS.items()):
            src_u, tgt_u = ut.index[fsrc(ut)].to_numpy(), ut.index[ftgt(ut)].to_numpy()
            t1 = time.time()
            for rep in range(REPEATS):
                rng = np.random.default_rng(11000 * rep + 101 * di + 7 * (t == "node"))
                su, tu = rng.permutation(src_u), rng.permutation(tgt_u)
                T_u, C_u, F_u, E_u = su[: len(su) // 2], su[len(su) // 2:], tu[: len(tu) // 2], tu[len(tu) // 2:]
                iT, iC, iU = (np.nonzero(np.isin(unit, x))[0] for x in (T_u, C_u, E_u))
                seed = 100 * rep + 13 * di + 1
                mu_C, mu_U = mu_good(Xg, y, iT, [iC, iU], seed)
                mu_T = oof(Xg, y, iT, unit[iT], seed, rng)
                uC, uU = pd.unique(site[iC]), pd.unique(site[iU])
                sC, sU = fact(site[iC]), fact(site[iU])
                w_all = ratio_weights(Zs[np.unique(site[iT])], Zs[np.unique(site[np.isin(unit, F_u)])], Zs[np.r_[uC, uU]], seed)
                wC, wU = w_all[: len(uC)], w_all[len(uC):]
                lams = np.unique(np.quantile(mu_T, np.linspace(0.02, 0.98, 30)))
                for a in ALPHAS:
                    sT = fact(site[iT])
                    lam = choose_lam(mu_T, bad[iT], sT, sT.max() + 1, lams, a, M)
                    s_C, L_C, _ = site_scores(mu_C, bad[iC], sC, sC.max() + 1, lam, M)
                    s_U, _, _ = site_scores(mu_U, None, sU, sU.max() + 1, lam, M)
                    eC, eU = np.isfinite(s_C), np.isfinite(s_U)
                    for meth, W in (("S6 unweighted", None), ("S6w weighted", (wC, wU))):
                        keep = np.zeros(len(iU), bool)
                        if eC.any() and eU.any():
                            if W is None:
                                sel = score_sdr_gpu(L_C[eC], s_C[eC], s_U[eU], a, a, prune="homo", random_state=seed)
                            else:
                                sel = score_sdr_w_gpu(L_C[eC], s_C[eC], W[0][eC], s_U[eU], W[1][eU], a, a, prune="homo", random_state=seed)
                            cert = np.zeros(sU.max() + 1, bool); cert[np.nonzero(eU)[0][sel]] = True
                            keep = (mu_U >= lam) & cert[sU]
                        rows.append(dict(design=name, type=t, rep=rep, alpha=a, method=meth,
                                         ess_C=ess(W[0][eC]) if W is not None else np.nan, n_cal_sites=int(eC.sum()),
                                         n_test_sites=int(sU.max() + 1), bad_C=bad[iC].mean(), bad_target=bad[iU].mean(),
                                         **evaluate(keep, bad[iU], sU, sU.max() + 1)))
            out(f"   {name}: source units {len(src_u)}, target units {len(tgt_u)}; {time.time() - t1:.0f} s (total {time.time() - t0:.0f} s)")
            pd.DataFrame(rows).to_csv(OUT / "per_split.csv", index=False)
    res = pd.DataFrame(rows)
    res["none"] = res.retention == 0
    res["exceed"] = res.site_fdp_kept > res.alpha
    res.to_csv(OUT / "per_split.csv", index=False)
    keys = ["type", "design", "alpha", "method"]
    s = res.groupby(keys).agg(retention=("retention", "mean"), site_coverage=("site_coverage", "mean"),
                              sdr=("site_fdp_kept", "mean"), sdr_sd=("site_fdp_kept", "std"), share_gt_alpha=("exceed", "mean"),
                              share_none=("none", "mean"), ess_C=("ess_C", "mean"), n_cal=("n_cal_sites", "mean"),
                              n_test=("n_test_sites", "mean"), bad_C=("bad_C", "mean"), bad_target=("bad_target", "mean"))
    s["excess_over_2se"] = (s.sdr - s.index.get_level_values("alpha")) - 2 * s.sdr_sd / np.sqrt(REPEATS)
    s.to_csv(OUT / "summary.csv")
    out("\nexcess_over_2se > 0: mean realised SDR above alpha by more than two standard errors")
    out(s.round(3).to_string())
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
