"""Frontier run v5 (13 Oct 2026), GPU-bound: SCoRE-SDR at the site level vs pooled conformal selection.

Data      : data/processed/paired_clean_v1.parquet; rivers; science phase; tidal gauges excluded; sites with >= 5 pairs;
            label B recomputed per subset; bad = |eB| > tau.
Splits    : 4 repeats x 5-fold over river units (reach-level SWORD id): U = one fold ("ungauged"), G = the rest.
            G units split 50/50 into T (score training) and C (calibration).
Score     : XGBoost on CUDA (data resident on GPU via CuPy); mu = P(good). Features: base (prototypes) or extended
            (features_v2: site-series consistency, width-elevation coherence, site history, C/D twin disagreement).
lam       : chosen from 3-fold out-of-fold predictions inside T only (plug-in retention at level alpha); C and U untouched.
Methods   : M3 pooled conformal selection (Jin & Candes p-values + BH) calibrated on C observations;
            S6 SCoRE-SDR (Bai & Jin 2026) with sites as units, eligibility >= 3 kept observations fixed in advance,
            via the GPU port code/score_sdr_gpu.py (exact equivalence: logs/score_sdr_gpu_equivalence.log);
            prune None and 'homo' (authors' recommendation; randomised, seeded).
Grid      : tau in {0.10, 0.25, 0.50} m x alpha in {0.10, 0.20}; feature ablation (base vs ext) at tau = 0.25.
Outputs   : results/frontier_v5/{per_split.csv, summary.csv, report.txt}
"""
import io, sys, time, warnings
import numpy as np
import pandas as pd
import cupy as cp
import xgboost as xgb
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE, label_B
from check_failure_mode import features as features_base, conf_select
from features_v2 import features_v2, twin_map
from prototype_site_level import evaluate
from prototype_score_sdr import site_scores
from prototype_score_sdr_v4 import choose_lam
from score_sdr_gpu import score_sdr_gpu

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 400)
OUT = ROOT / "results" / "frontier_v5"; OUT.mkdir(parents=True, exist_ok=True)
TAUS, ALPHAS, REPEATS, FOLDS, OOF, M = (0.10, 0.25, 0.50), (0.10, 0.20), 4, 5, 3, 3
XGB = dict(device="cuda", tree_method="hist", n_estimators=400, learning_rate=0.05, max_depth=6,
           subsample=0.8, colsample_bytree=0.8, objective="binary:logistic")
_CFG = ROOT / "results" / "final" / "model_config.json"
if _CFG.exists():   # tuned settings (tune_score.py) override the defaults for every script importing XGB / mu_good
    XGB.update({k: v for k, v in __import__("json").loads(_CFG.read_text()).items()
                if k in ("max_depth", "learning_rate", "n_estimators", "min_child_weight", "subsample", "colsample_bytree")})
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def mu_good(Xg, y, train_idx, pred_idx_list, seed):
    clf = xgb.XGBClassifier(random_state=seed, **XGB)
    clf.fit(Xg[cp.asarray(train_idx)], y[cp.asarray(train_idx)])
    res = []
    for p in pred_idx_list:
        pr = clf.predict_proba(Xg[cp.asarray(p)])[:, 1]
        res.append(1.0 - (cp.asnumpy(pr) if isinstance(pr, cp.ndarray) else np.asarray(pr)))
    return res


def one_split(Xg, y_gpu, bad, unit, site, U_units, rng, seed, tag):
    U = np.isin(unit, U_units)
    g_units = rng.permutation(np.unique(unit[~U])); half = len(g_units) // 2
    iT, iC, iU = (np.nonzero(np.isin(unit, g_units[:half]))[0], np.nonzero(np.isin(unit, g_units[half:]))[0], np.nonzero(U)[0])
    mu_C, mu_U = mu_good(Xg, y_gpu, iT, [iC, iU], seed)
    mu_T = np.empty(len(iT)); unit_T = unit[iT]
    for k, fu in enumerate(np.array_split(rng.permutation(g_units[:half]), OOF)):
        held = np.isin(unit_T, fu)
        mu_T[held] = mu_good(Xg, y_gpu, iT[~held], [iT[held]], seed * 10 + k)[0]
    sT, sC, sU = (pd.factorize(site[i])[0] for i in (iT, iC, iU))
    nsU = sU.max() + 1
    bad_T, bad_C, bad_U = bad[iT], bad[iC], bad[iU]
    lams = np.unique(np.quantile(mu_T, np.linspace(0.02, 0.98, 30)))
    rows = []
    for a in ALPHAS:
        rows.append(dict(tag, alpha=a, method="M3 pooled conformal selection", lam=np.nan,
                         **evaluate(conf_select(mu_C, bad_C, mu_U, a), bad_U, sU, nsU)))
        lam = choose_lam(mu_T, bad_T, sT, sT.max() + 1, lams, a)
        s_C, L_C, _ = site_scores(mu_C, bad_C, sC, sC.max() + 1, lam, M)
        s_U, _, _ = site_scores(mu_U, None, sU, nsU, lam, M)
        eC, eU = np.isfinite(s_C), np.isfinite(s_U)
        for prune in (None, "homo"):
            keep = np.zeros(len(iU), bool)
            if eC.any() and eU.any():
                sel = score_sdr_gpu(L_C[eC], s_C[eC], s_U[eU], a, a, prune=prune, random_state=seed)
                cert = np.zeros(nsU, bool); cert[np.nonzero(eU)[0][sel]] = True
                keep = (mu_U >= lam) & cert[sU]
            rows.append(dict(tag, alpha=a, method=f"S6 SCoRE-SDR sites (prune={prune})", lam=lam,
                             **evaluate(keep, bad_U, sU, nsU)))
    return rows


def main():
    t0 = time.time()
    out(f"xgboost {xgb.__version__} (CUDA build: {xgb.build_info().get('USE_CUDA')}), cupy {cp.__version__}")
    df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
    twin = twin_map(df, pd.read_parquet(ROOT / "data" / "processed" / "cd_crosswalk_v1.parquet"))
    rows = []
    for t in ("reach", "node"):
        for v in ("D", "C"):
            d = df[(df.type == t) & (df.v == v) & (df.phase == "science") & (~df.tidal_gauge)].copy()
            d["n"] = d.groupby(SITE).wse.transform("size")
            d = d[d.n >= 5].copy()
            d["eB"], _ = label_B(d)
            unit = (d.fid.astype("int64") // (10 if t == "reach" else 10000)).to_numpy()
            site = d.groupby(SITE).ngroup().to_numpy()
            Xs = {"ext": features_v2(d, t, twin.loc[d.index]), "base": features_base(d, t)}
            Xg = {k: cp.asarray(X.replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)) for k, X in Xs.items()}
            units = np.unique(unit)
            out(f"{t} {v}: rows {len(d)}, units {len(units)}, features ext {Xs['ext'].shape[1]} / base {Xs['base'].shape[1]}, "
                f"twin coverage {Xs['ext'].has_twin.mean():.3f}")
            for tau in TAUS:
                bad = (d.eB.abs() > tau).to_numpy()
                y_gpu = cp.asarray(bad.astype(np.int32))
                for fs in (("ext", "base") if tau == 0.25 else ("ext",)):
                    t1 = time.time()
                    for q in range(REPEATS):
                        folds = np.array_split(np.random.default_rng(5000 * q + 17 * (t == "node") + 3 * (v == "C")).permutation(units), FOLDS)
                        for f in range(FOLDS):
                            tag = dict(type=t, v=v, tau=tau, features=fs, rep=q, fold=f)
                            rows += one_split(Xg[fs], y_gpu, bad, unit, site, folds[f],
                                              np.random.default_rng(100 * q + f), 1000 * q + 10 * f, tag)
                    out(f"   tau {tau} {fs}: bad rate {bad.mean():.3f}; {time.time() - t1:.0f} s (total {time.time() - t0:.0f} s)")
                    pd.DataFrame(rows).to_csv(OUT / "per_split.csv", index=False)
    res = pd.DataFrame(rows)
    res["exceed"] = res.site_fdp_kept > res.alpha
    res["none"] = res.retention == 0
    res.to_csv(OUT / "per_split.csv", index=False)
    keys = ["type", "v", "tau", "features", "alpha", "method"]
    s = res.groupby(keys).agg(retention=("retention", "mean"), site_coverage=("site_coverage", "mean"),
                              pooled_fdp=("fdp", "mean"), site_fdp_where_kept=("site_fdp_kept", "mean"),
                              share_splits_gt_alpha=("exceed", "mean"), share_splits_none=("none", "mean"))
    s["site_fdp_if_any"] = res[~res.none].groupby(keys).site_fdp_kept.mean()
    s.to_csv(OUT / "summary.csv")
    out(f"\n{REPEATS} x {FOLDS}-fold over units; site_fdp_where_kept counts splits that certify nothing as 0 (SDR convention);"
        " site_fdp_if_any averages only splits that certify something")
    out(s.round(3).to_string())
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
