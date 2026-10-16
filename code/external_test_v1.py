"""External test v1 (13-14 Oct 2026), GPU-bound: does a guarantee built on North America hold on Brazilian reaches?

Source (gauged, labelled): USGS release, Version D reaches (science, non-tidal, >= 5 pairs), label B.
Target: Brazil pairs (pair_brazil.py); main filters: no dam/reservoir sites, no river-name mismatch, distance <= 0.5 km,
  no tidal candidates, >= 5 pairs per site; label B. Strata: brazil_all, amazon_basin (ANA Bacia 1), brazil_no_hydro.
  ANA time readings: 'brasilia' (UTC-3) and 'local' (legal time per state).
Features: features_v2 without the C/D twin (Brazil has Version D only); target columns aligned to the source.
Each of 20 repeats: NA units 50/50 into T (score, lambda via OOF) and C (calibration); Brazil units 50/50 into
  'fit' (covariates only, for weights; labels used only by the in-domain reference) and 'eval'.
Methods on eval sites: S6 unweighted (NA calibration); S6w physical weights (SWORD v17b physical attributes, NA vs SA);
  S6 in-domain reference (calibrated on the Brazil fit half: what Brazilian gauges would give).
Outputs: results/external_v1/{per_split.csv, summary.csv, report.txt}
"""
import io, pathlib, sys, time, warnings
import numpy as np
import pandas as pd
import cupy as cp
import xgboost as xgb
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "external" / "pylib"))
import h5py
from swotio import ROOT, SITE, label_B
from features_v2 import features_v2
from prototype_site_level import evaluate
from prototype_score_sdr import site_scores
from score_sdr_gpu import score_sdr_gpu, score_sdr_w_gpu
from run_frontier_v5 import XGB
from run_robustness_v6 import subset, oof, choose_lam, fact
from run_shift_v7 import ratio_weights, ess

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 300)
OUT = ROOT / "results" / "external_v1"; OUT.mkdir(parents=True, exist_ok=True)
TAU = float(__import__("os").environ.get("TAU", 0.25))
ALPHAS, M = (0.10, 0.20), 3
REPEATS = int(__import__("os").environ.get("REPEATS", 20))
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def physical(nc):
    with h5py.File(nc, "r") as f:
        g = f["reaches"]
        r = pd.DataFrame({v: np.asarray(g[v][...]).astype(float) for v in
                          ["reach_id", "width", "wse", "slope", "facc", "reach_length", "n_chan_mod", "obstr_type", "stream_order", "dist_out"]})
        ice = np.asarray(g["iceflag"][...]).astype(float)
    r = r.replace(-9999.0, np.nan); ice[ice < 0] = np.nan
    valid = ~np.isnan(ice); nv = valid.sum(axis=0).astype(float); nv[nv == 0] = np.nan
    lg = lambda s: np.log(np.clip(s, 1, None))
    return pd.DataFrame({"reach_id": r.reach_id.astype("int64"), "log_width": lg(r.width), "wse": r.wse, "slope": r.slope,
                         "log_facc": lg(r.facc), "log_len": lg(r.reach_length), "nch": r.n_chan_mod,
                         "obstr": (r.obstr_type > 0).astype(float), "ice_frac": ((ice >= 1) & valid).sum(0) / nv,
                         "ice_full_frac": ((ice == 2) & valid).sum(0) / nv, "stream_order": r.stream_order,
                         "log_dist_out": lg(r.dist_out)})


def site_cov(d, phys):
    s = d.groupby(SITE).agg(fid=("fid", "first"), unit=("unit", "first")).reset_index(drop=True)
    s = s.merge(phys, left_on="fid", right_on="reach_id", how="left")
    return s


def notwin(d, t="reach"):
    X = features_v2(d, t, pd.Series(np.nan, index=d.index)).drop(columns=["twin_dwse", "has_twin"])
    return X.replace([np.inf, -np.inf], np.nan)


def predict(clf, X):
    p = clf.predict_proba(cp.asarray(X.to_numpy(np.float32)))[:, 1]
    return 1 - (cp.asnumpy(p) if isinstance(p, cp.ndarray) else np.asarray(p))


def prep_brazil(path, mode):
    b = pd.read_parquet(path)
    b = b[(b.tz_mode == mode) & ~b.dam_or_reservoir.astype(bool) & (b.name_status != "mismatch")
          & (b.dist_km <= 0.5) & ~b.tidal_candidate.astype(bool)].copy()
    b["n"] = b.groupby(SITE).wse.transform("size")
    b = b[b.n >= 5].copy()
    b["eB"], _ = label_B(b)
    return b.reset_index(drop=True)


def run_s6(L, s, sE, a, seed, wC=None, wE=None):
    eC, eE = np.isfinite(s), np.isfinite(sE)
    if not (eC.any() and eE.any()):
        return None
    if wC is None:
        sel = score_sdr_gpu(L[eC], s[eC], sE[eE], a, a, prune="homo", random_state=seed)
    else:
        sel = score_sdr_w_gpu(L[eC], s[eC], wC[eC], sE[eE], wE[eE], a, a, prune="homo", random_state=seed)
    cert = np.zeros(len(sE), bool); cert[np.nonzero(eE)[0][sel]] = True
    return cert


def main():
    t0 = time.time()
    pairs = ROOT / "data" / "processed" / (sys.argv[1] if len(sys.argv) > 1 else "brazil_pairs_v1.parquet")
    raw = ROOT / "data" / "raw" / "sword_v17b"
    na_phys, sa_phys = physical(raw / "na_sword_v17b.nc"), physical(raw / "sa_sword_v17b.nc")
    COV = [c for c in na_phys.columns if c != "reach_id"]
    d = subset(pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet"), "reach", "D")
    Xna = notwin(d); cols = Xna.columns
    Xg = cp.asarray(Xna.to_numpy(np.float32))
    bad = (d.eB.abs() > TAU).to_numpy(); y = cp.asarray(bad.astype(np.int32))
    site, unit = d.groupby(SITE).ngroup().to_numpy(), d.unit.to_numpy()
    Z_na = site_cov(d, na_phys)[COV].to_numpy(float)
    out(f"source NA reach D: {len(d)} pairs, {site.max() + 1} sites, bad rate {bad.mean():.3f}, {len(cols)} features")
    T = {}
    for mode in ("brasilia", "local"):
        b = prep_brazil(pairs, mode)
        Xb = notwin(b).reindex(columns=cols)
        bits = [c for c in cols if c.startswith("bit")]
        Xb[bits] = Xb[bits].fillna(0)
        sb = site_cov(b, sa_phys)
        T[mode] = dict(b=b, X=Xb, Z=sb[COV].to_numpy(float), site=b.groupby(SITE).ngroup().to_numpy(),
                       unit=b.unit.to_numpy(), bad=(b.eB.abs() > TAU).to_numpy())
        for stratum, sel in (("brazil_all", np.ones(len(b), bool)), ("amazon_basin", b.amazon_basin.astype(bool).to_numpy()),
                             ("brazil_no_hydro", ~b.hydro_site.astype(bool).to_numpy())):
            T[mode][stratum] = sel
            out(f"target {mode}/{stratum}: {int(sel.sum())} pairs, {b[sel].gauge.nunique()} gauges, "
                f"bad rate {T[mode]['bad'][sel].mean():.3f}; SWORD covariates for {np.isfinite(sb.log_width).mean():.3f} of sites")
    rows = []
    for r in range(REPEATS):
        rng = np.random.default_rng(31000 + r); seed = 500 + r
        uu = rng.permutation(np.unique(unit)); T_u, C_u = uu[: len(uu) // 2], uu[len(uu) // 2:]
        iT, iC = np.nonzero(np.isin(unit, T_u))[0], np.nonzero(np.isin(unit, C_u))[0]
        clf = xgb.XGBClassifier(random_state=seed, **XGB).fit(Xg[cp.asarray(iT)], y[cp.asarray(iT)])
        pr = clf.predict_proba(Xg[cp.asarray(iC)])[:, 1]
        mu_C = 1 - (cp.asnumpy(pr) if isinstance(pr, cp.ndarray) else np.asarray(pr))
        mu_T = oof(Xg, y, iT, unit[iT], seed, rng)
        sT, sC = fact(site[iT]), fact(site[iC]); uC = pd.unique(site[iC])
        lams = np.unique(np.quantile(mu_T, np.linspace(0.02, 0.98, 30)))
        lam = {a: choose_lam(mu_T, bad[iT], sT, sT.max() + 1, lams, a, M) for a in ALPHAS}
        for mode, tg in T.items():
            mu_b = predict(clf, tg["X"])
            for stratum in ("brazil_all", "amazon_basin", "brazil_no_hydro"):
                sel = tg[stratum]
                k_cfg = ["brasilia", "local"].index(mode) * 10 + ["brazil_all", "amazon_basin", "brazil_no_hydro"].index(stratum)
                bu = np.random.default_rng(seed * 100 + k_cfg).permutation(np.unique(tg["unit"][sel]))
                F_u, E_u = bu[: len(bu) // 2], bu[len(bu) // 2:]
                iF = np.nonzero(sel & np.isin(tg["unit"], F_u))[0]; iE = np.nonzero(sel & np.isin(tg["unit"], E_u))[0]
                sF, sE = fact(tg["site"][iF]), fact(tg["site"][iE]); uE = pd.unique(tg["site"][iE])
                w = ratio_weights(Z_na[np.unique(site[iT])], tg["Z"][np.unique(tg["site"][iF])],
                                  np.vstack([Z_na[uC], tg["Z"][uE]]), seed)
                wC, wE = w[: len(uC)], w[len(uC):]
                for a in ALPHAS:
                    s_C, L_C, _ = site_scores(mu_C, bad[iC], sC, sC.max() + 1, lam[a], M)
                    s_F, L_F, _ = site_scores(mu_b[iF], tg["bad"][iF], sF, sF.max() + 1, lam[a], M)
                    s_E, _, _ = site_scores(mu_b[iE], None, sE, sE.max() + 1, lam[a], M)
                    eE = np.isfinite(s_E)
                    ratio = wE[eE] / max(wC[np.isfinite(s_C)].sum(), 1e-12) if eE.any() else np.array([np.nan])
                    for meth, cert in (("S6 unweighted (NA calib)", run_s6(L_C, s_C, s_E, a, seed)),
                                       ("S6w physical weights (NA calib)", run_s6(L_C, s_C, s_E, a, seed, wC, wE)),
                                       ("S6 in-domain reference (Brazil calib)", run_s6(L_F, s_F, s_E, a, seed))):
                        keep = np.zeros(len(iE), bool) if cert is None else (mu_b[iE] >= lam[a]) & cert[sE]
                        rows.append(dict(tz_mode=mode, stratum=stratum, rep=r, alpha=a, method=meth,
                                         ess_C=ess(wC[np.isfinite(s_C)]), wratio_med=float(np.nanmedian(ratio)),
                                         bad_target=tg["bad"][iE].mean(), n_eval_sites=int(sE.max() + 1),
                                         **evaluate(keep, tg["bad"][iE], sE, sE.max() + 1)))
        out(f"repeat {r + 1}/{REPEATS} done at {time.time() - t0:.0f} s")
        pd.DataFrame(rows).to_csv(OUT / "per_split.csv", index=False)
    res = pd.DataFrame(rows)
    res["none"] = res.retention == 0
    res["exceed"] = res.site_fdp_kept > res.alpha
    res.to_csv(OUT / "per_split.csv", index=False)
    keys = ["tz_mode", "stratum", "alpha", "method"]
    s = res.groupby(keys).agg(retention=("retention", "mean"), site_coverage=("site_coverage", "mean"),
                              sdr=("site_fdp_kept", "mean"), sdr_sd=("site_fdp_kept", "std"), share_gt_alpha=("exceed", "mean"),
                              share_none=("none", "mean"), ess_C=("ess_C", "mean"), wratio_med=("wratio_med", "mean"),
                              bad_target=("bad_target", "mean"), n_eval_sites=("n_eval_sites", "mean"))
    s["excess_over_2se"] = (s.sdr - s.index.get_level_values("alpha")) - 2 * s.sdr_sd / np.sqrt(REPEATS)
    s.to_csv(OUT / "summary.csv")
    out("\nexcess_over_2se > 0: mean realised SDR above alpha by more than two standard errors")
    out(s.round(3).to_string())
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
