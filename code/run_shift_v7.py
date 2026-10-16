"""Shift experiment v7 (13 Oct 2026), GPU-bound: does the site-level guarantee survive a gauged -> ungauged covariate shift,
and does weighting (SCoRE_SDR_w, Bai & Jin 2026) restore it?

Semi-synthetic design on Version D rivers (tau = 0.25 m, alpha in {0.10, 0.20}, 20 repeats per configuration):
  units are split into a source (gauged) pool and a target (ungauged) pool with probability
  pi(z) = expit(b * z) on a standardised unit covariate: 'width' (log prior width; gauges favour wide rivers)
  or 'south' (-latitude; gauges favour the south; latitude is not a score feature). b = 0 is no shift.
  Source units: 50/50 T (score training, lambda via 3-fold OOF) and C (calibration).
  Target units: half used only as unlabelled covariates to fit the density ratio, half for evaluation.
Weights: XGBoost (CUDA) classifier of target vs source on site attributes available from SWORD for any reach
  (log width, prior WSE, latitude, longitude, channels, dam flag, log length for reaches), fitted on T sites vs the
  unlabelled target half; w = odds ratio rescaled by pool sizes. Oracle weights: (1 - pi) / pi.
Methods: S6 unweighted, S6w estimated weights, S6w oracle weights (prune 'homo'; GPU port verified against the reference).
Outputs: results/shift_v7/{per_split.csv, summary.csv, report.txt}
"""
import io, sys, time, warnings
import numpy as np
import pandas as pd
import cupy as cp
import xgboost as xgb
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE
from features_v2 import features_v2, twin_map
from prototype_site_level import evaluate
from prototype_score_sdr import site_scores
from score_sdr_gpu import score_sdr_gpu, score_sdr_w_gpu
from run_frontier_v5 import mu_good, XGB
from run_robustness_v6 import subset, oof, choose_lam, fact

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 300)
OUT = ROOT / "results" / "shift_v7"; OUT.mkdir(parents=True, exist_ok=True)
TAU, ALPHAS, REPEATS, M = 0.25, (0.10, 0.20), 20, 3
SHIFTS = [("width", 0.0), ("width", 1.0), ("width", 2.0), ("south", 1.0), ("south", 2.0)]
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def site_attributes(d, t):
    a = d.groupby(SITE).agg(unit=("unit", "first"), p_width=("p_width", "first"), p_wse=("p_wse", "first"),
                            lat=("gage_latitude", "first"), lon=("gage_longitude", "first"),
                            nch=("p_n_ch_mod", "first"), dam=("p_dam_id", "first"), p_length=("p_length", "first"))
    Z = pd.DataFrame({"log_width": np.log(a.p_width), "p_wse": a.p_wse, "lat": a.lat, "lon": a.lon,
                      "nch": a.nch, "dam": (a.dam.fillna(0) > 0).astype(float)}, index=a.index)
    if t == "reach":
        Z["log_length"] = np.log(a.p_length)
    return a.unit.to_numpy(), Z


def ratio_weights(Z_src_fit, Z_tgt_fit, Z_apply, seed):
    X = cp.asarray(np.vstack([Z_src_fit, Z_tgt_fit]).astype(np.float32))
    y = cp.asarray(np.r_[np.zeros(len(Z_src_fit)), np.ones(len(Z_tgt_fit))].astype(np.int32))
    clf = xgb.XGBClassifier(random_state=seed, **{**XGB, "n_estimators": 200, "max_depth": 4}).fit(X, y)
    p = clf.predict_proba(cp.asarray(Z_apply.astype(np.float32)))[:, 1]
    p = np.clip(cp.asnumpy(p) if isinstance(p, cp.ndarray) else np.asarray(p), 1e-4, 1 - 1e-4)
    return p / (1 - p) * len(Z_src_fit) / len(Z_tgt_fit)


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


def main():
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
    twin = twin_map(df, pd.read_parquet(ROOT / "data" / "processed" / "cd_crosswalk_v1.parquet"))
    rows = []
    for t in ("reach", "node"):
        d = subset(df, t, "D")
        X = features_v2(d, t, twin.loc[d.index]).replace([np.inf, -np.inf], np.nan)
        Xg = cp.asarray(X.to_numpy(np.float32))
        bad = (d.eB.abs() > TAU).to_numpy(); y = cp.asarray(bad.astype(np.int32))
        site = d.groupby(SITE).ngroup().to_numpy(); unit = d.unit.to_numpy()
        s_unit, Z = site_attributes(d, t)                 # rows in the same (sorted) order as ngroup ids
        Zs = Z.to_numpy(float)
        ucov = pd.DataFrame({"unit": s_unit, "lw": Z.log_width.to_numpy(), "lat": Z.lat.to_numpy()}).groupby("unit").mean()
        units = ucov.index.to_numpy()
        zstd = {"width": ((ucov.lw - ucov.lw.mean()) / ucov.lw.std()).to_numpy(),
                "south": (-(ucov.lat - ucov.lat.mean()) / ucov.lat.std()).to_numpy()}
        unit_pos = pd.Series(np.arange(len(units)), index=units)
        for kind, b in SHIFTS:
            pi = 1 / (1 + np.exp(-b * zstd[kind]))
            t1 = time.time()
            for r in range(REPEATS):
                rng = np.random.default_rng(9000 * r + 31 * (t == "node") + int(10 * b) + 500 * (kind == "south"))
                src = rng.uniform(size=len(units)) < pi
                tu = rng.permutation(units[~src]); su = rng.permutation(units[src])
                tgt_fit, tgt_eval = tu[: len(tu) // 2], tu[len(tu) // 2:]
                iT, iC, iU = (np.nonzero(np.isin(unit, x))[0] for x in (su[: len(su) // 2], su[len(su) // 2:], tgt_eval))
                seed = 100 * r + 7
                mu_C, mu_U = mu_good(Xg, y, iT, [iC, iU], seed)
                mu_T = oof(Xg, y, iT, unit[iT], seed, rng)
                uC, uU = pd.unique(site[iC]), pd.unique(site[iU])          # same order as fact()
                sC, sU = fact(site[iC]), fact(site[iU])
                w_all = ratio_weights(Zs[np.unique(site[iT])], Zs[np.unique(site[np.isin(unit, tgt_fit)])],
                                      Zs[np.r_[uC, uU]], seed)
                wC_est, wU_est = w_all[: len(uC)], w_all[len(uC):]
                pi_site = pi[unit_pos[s_unit].to_numpy()]
                wC_or, wU_or = ((1 - pi_site) / pi_site)[uC], ((1 - pi_site) / pi_site)[uU]
                lams = np.unique(np.quantile(mu_T, np.linspace(0.02, 0.98, 30)))
                for a in ALPHAS:
                    lam = choose_lam(mu_T, bad[iT], fact(site[iT]), fact(site[iT]).max() + 1, lams, a, M)
                    s_C, L_C, _ = site_scores(mu_C, bad[iC], sC, sC.max() + 1, lam, M)
                    s_U, _, _ = site_scores(mu_U, None, sU, sU.max() + 1, lam, M)
                    eC, eU = np.isfinite(s_C), np.isfinite(s_U)
                    for name, W in (("S6 unweighted", None), ("S6w estimated", (wC_est, wU_est)), ("S6w oracle", (wC_or, wU_or))):
                        keep = np.zeros(len(iU), bool)
                        if eC.any() and eU.any():
                            if W is None:
                                sel = score_sdr_gpu(L_C[eC], s_C[eC], s_U[eU], a, a, prune="homo", random_state=seed)
                            else:
                                sel = score_sdr_w_gpu(L_C[eC], s_C[eC], W[0][eC], s_U[eU], W[1][eU], a, a, prune="homo", random_state=seed)
                            cert = np.zeros(sU.max() + 1, bool); cert[np.nonzero(eU)[0][sel]] = True
                            keep = (mu_U >= lam) & cert[sU]
                        rows.append(dict(type=t, shift=kind, b=b, rep=r, alpha=a, method=name,
                                         ess_C=ess(W[0][eC]) if W is not None else np.nan, n_cal_sites=int(eC.sum()),
                                         bad_rate_C=bad[iC].mean(), bad_rate_target=bad[iU].mean(),
                                         **evaluate(keep, bad[iU], sU, sU.max() + 1)))
            out(f"{t} shift={kind} b={b}: {time.time() - t1:.0f} s (total {time.time() - t0:.0f} s)")
            pd.DataFrame(rows).to_csv(OUT / "per_split.csv", index=False)
    res = pd.DataFrame(rows)
    res["none"] = res.retention == 0
    res["exceed"] = res.site_fdp_kept > res.alpha
    res.to_csv(OUT / "per_split.csv", index=False)
    keys = ["type", "shift", "b", "alpha", "method"]
    s = res.groupby(keys).agg(retention=("retention", "mean"), site_coverage=("site_coverage", "mean"),
                              sdr=("site_fdp_kept", "mean"), sdr_sd=("site_fdp_kept", "std"),
                              share_gt_alpha=("exceed", "mean"), share_none=("none", "mean"), ess_C=("ess_C", "mean"),
                              n_cal_sites=("n_cal_sites", "mean"), bad_C=("bad_rate_C", "mean"), bad_target=("bad_rate_target", "mean"))
    s["excess_over_2se"] = (s.sdr - s.index.get_level_values("alpha")) - 2 * s.sdr_sd / np.sqrt(REPEATS)
    s.to_csv(OUT / "summary.csv")
    out("\nexcess_over_2se > 0 means mean realised SDR exceeds alpha by more than two standard errors")
    out(s.round(3).to_string())
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
