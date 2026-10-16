"""External test v2 (14 Oct 2026): sensitivity of the North America -> Brazil result to station-match and gauge QC.

Same source, model, splits, seeds and methods as external_test_v1; ANA time reading fixed to Brasilia (UTC-3, verified).
Common filters: no dam/reservoir sites, no tidal candidates, >= 5 pairs per site, label B.
Strata (site-level):
  main            name not a mismatch, distance <= 0.5 km                      (= v1 brazil_all)
  strict_names    river names match, distance <= 0.25 km                      (label-free)
  strict_qc       strict_names and gauge-only QC pass (gauge_qc_brazil.py)     (label-free)
  hydro_only      main and hydropower monitoring site                          (label-free)
  optimistic      main and site median |eB| <= 1 m                             (LABEL-BASED: optimistic bound only)
Outputs: results/external_v2/{per_split.csv, summary.csv, report.txt}
"""
import io, pathlib, sys, time, warnings
import numpy as np
import pandas as pd
import cupy as cp
import xgboost as xgb
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from swotio import ROOT, SITE, label_B
from external_test_v1 import physical, site_cov, notwin, predict, run_s6, TAU, ALPHAS, M, XGB
from prototype_site_level import evaluate
from prototype_score_sdr import site_scores
from run_robustness_v6 import subset, oof, choose_lam, fact
from run_shift_v7 import ratio_weights, ess

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 300)
TYPE = __import__("os").environ.get("TYPE", "reach")
OUT = ROOT / "results" / ("external_v2" + ("" if TAU == 0.25 else f"_tau{TAU}") + ("" if TYPE == "reach" else "_node"))
OUT.mkdir(parents=True, exist_ok=True)
REPEATS = int(__import__("os").environ.get("REPEATS", 20))
STRATA = ["main", "strict_names", "strict_qc", "hydro_only", "non_hydro", "amazon_basin", "optimistic"]
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def site_cov_t(d, phys):
    """SWORD physical covariates per site; nodes use their parent reach (CBBBBBRRRR + type digit)."""
    if TYPE == "reach":
        return site_cov(d, phys)
    dd = d.assign(fid=(d.fid.astype("int64") // 10000) * 10 + d.fid.astype("int64") % 10)
    return site_cov(dd, phys)


def prep():
    b = pd.read_parquet(ROOT / "data" / "processed" / ("brazil_pairs_v1.parquet" if TYPE == "reach" else "brazil_pairs_nodes_v1.parquet"))
    b = b[(b.tz_mode == "brasilia") & ~b.dam_or_reservoir.astype(bool) & ~b.tidal_candidate.astype(bool)].copy()
    q = pd.read_csv(ROOT / "results" / "amazon_v1" / "gauge_qc.csv")[["code", "gauge_qc_pass"]]
    b = b.merge(q, on="code", how="left")
    b["gauge_qc_pass"] = b.gauge_qc_pass.fillna(False).astype(bool)
    b["n"] = b.groupby(SITE).wse.transform("size")
    b = b[b.n >= 5].copy()
    b["eB"], _ = label_B(b)
    b = b.reset_index(drop=True)
    main = (b.name_status != "mismatch") & (b.dist_km <= 0.5)
    strict = (b.name_status == "match") & (b.dist_km <= 0.25)
    site_med = b.groupby(SITE).eB.transform(lambda x: x.abs().median())
    masks = {"main": main, "strict_names": strict, "strict_qc": strict & b.gauge_qc_pass,
             "hydro_only": main & b.hydro_site.astype(bool), "non_hydro": main & ~b.hydro_site.astype(bool),
             "amazon_basin": main & b.amazon_basin.astype(bool), "optimistic": main & (site_med <= 1.0)}
    return b, {k: v.to_numpy() for k, v in masks.items()}


def main():
    t0 = time.time()
    raw = ROOT / "data" / "raw" / "sword_v17b"
    na_phys, sa_phys = physical(raw / "na_sword_v17b.nc"), physical(raw / "sa_sword_v17b.nc")
    COV = [c for c in na_phys.columns if c != "reach_id"]
    d = subset(pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet"), TYPE, "D")
    Xna = notwin(d, TYPE); cols = Xna.columns
    Xg = cp.asarray(Xna.to_numpy(np.float32))
    bad = (d.eB.abs() > TAU).to_numpy(); y = cp.asarray(bad.astype(np.int32))
    site, unit = d.groupby(SITE).ngroup().to_numpy(), d.unit.to_numpy()
    Z_na = site_cov_t(d, na_phys)[COV].to_numpy(float)
    b, masks = prep()
    Xb = notwin(b, TYPE).reindex(columns=cols)
    bits = [c for c in cols if c.startswith("bit")]
    Xb[bits] = Xb[bits].fillna(0)
    Zb = site_cov_t(b, sa_phys)[COV].to_numpy(float)
    site_b, unit_b, bad_b = b.groupby(SITE).ngroup().to_numpy(), b.unit.to_numpy(), (b.eB.abs() > TAU).to_numpy()
    for k in STRATA:
        sel = masks[k]
        out(f"stratum {k}: {int(sel.sum())} pairs, {b[sel].gauge.nunique()} gauges, bad rate {bad_b[sel].mean():.3f}")
    rows = []
    for r in range(REPEATS):
        rng = np.random.default_rng(31000 + r); seed = 500 + r              # same NA splits and seeds as v1
        uu = rng.permutation(np.unique(unit)); T_u, C_u = uu[: len(uu) // 2], uu[len(uu) // 2:]
        iT, iC = np.nonzero(np.isin(unit, T_u))[0], np.nonzero(np.isin(unit, C_u))[0]
        clf = xgb.XGBClassifier(random_state=seed, **XGB).fit(Xg[cp.asarray(iT)], y[cp.asarray(iT)])
        pr = clf.predict_proba(Xg[cp.asarray(iC)])[:, 1]
        mu_C = 1 - (cp.asnumpy(pr) if isinstance(pr, cp.ndarray) else np.asarray(pr))
        mu_T = oof(Xg, y, iT, unit[iT], seed, rng)
        sT, sC = fact(site[iT]), fact(site[iC]); uC = pd.unique(site[iC])
        lams = np.unique(np.quantile(mu_T, np.linspace(0.02, 0.98, 30)))
        lam = {a: choose_lam(mu_T, bad[iT], sT, sT.max() + 1, lams, a, M) for a in ALPHAS}
        mu_b = predict(clf, Xb)
        for ki, k in enumerate(STRATA):
            sel = masks[k]
            bu = np.random.default_rng(seed * 100 + ki).permutation(np.unique(unit_b[sel]))
            F_u, E_u = bu[: len(bu) // 2], bu[len(bu) // 2:]
            iF = np.nonzero(sel & np.isin(unit_b, F_u))[0]; iE = np.nonzero(sel & np.isin(unit_b, E_u))[0]
            sF, sE = fact(site_b[iF]), fact(site_b[iE]); uE = pd.unique(site_b[iE])
            w = ratio_weights(Z_na[np.unique(site[iT])], Zb[np.unique(site_b[iF])], np.vstack([Z_na[uC], Zb[uE]]), seed)
            wC, wE = w[: len(uC)], w[len(uC):]
            for a in ALPHAS:
                s_C, L_C, _ = site_scores(mu_C, bad[iC], sC, sC.max() + 1, lam[a], M)
                s_F, L_F, _ = site_scores(mu_b[iF], bad_b[iF], sF, sF.max() + 1, lam[a], M)
                s_E, _, _ = site_scores(mu_b[iE], None, sE, sE.max() + 1, lam[a], M)
                for meth, cert in (("S6 unweighted (NA calib)", run_s6(L_C, s_C, s_E, a, seed)),
                                   ("S6w physical weights (NA calib)", run_s6(L_C, s_C, s_E, a, seed, wC, wE)),
                                   ("S6 in-domain reference (Brazil calib)", run_s6(L_F, s_F, s_E, a, seed))):
                    keep = np.zeros(len(iE), bool) if cert is None else (mu_b[iE] >= lam[a]) & cert[sE]
                    rows.append(dict(stratum=k, rep=r, alpha=a, method=meth, bad_target=bad_b[iE].mean(),
                                     n_eval_sites=int(sE.max() + 1), **evaluate(keep, bad_b[iE], sE, sE.max() + 1)))
        out(f"repeat {r + 1}/{REPEATS} done at {time.time() - t0:.0f} s")
    res = pd.DataFrame(rows)
    res["none"] = res.retention == 0
    res["exceed"] = res.site_fdp_kept > res.alpha
    res.to_csv(OUT / "per_split.csv", index=False)
    keys = ["stratum", "alpha", "method"]
    s = res.groupby(keys).agg(retention=("retention", "mean"), site_coverage=("site_coverage", "mean"),
                              sdr=("site_fdp_kept", "mean"), sdr_sd=("site_fdp_kept", "std"), share_gt_alpha=("exceed", "mean"),
                              share_none=("none", "mean"), bad_target=("bad_target", "mean"), n_eval_sites=("n_eval_sites", "mean"))
    s["excess_over_2se"] = (s.sdr - s.index.get_level_values("alpha")) - 2 * s.sdr_sd / np.sqrt(REPEATS)
    s.to_csv(OUT / "summary.csv")
    out("\nexcess_over_2se > 0: mean realised SDR above alpha by more than two standard errors")
    out(s.round(3).to_string())
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
