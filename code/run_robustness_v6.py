"""Robustness run v6 (13 Oct 2026), GPU-bound. tau = 0.25 m, alpha in {0.10, 0.20}; S6 = SCoRE-SDR at the site level
(GPU port), M3 = pooled conformal selection (main design only).

Within-version designs (4 repeats x 5 folds), per type x version:
  unit/ext (m in {1, 3, 5}), unit/notwin (no C/D twin feature), unit/causal (past passes only),
  basin/ext (folds over Pfafstetter level-4 basins = first 4 SWORD digits), unit/ext+calval, unit/ext+tidal.
Cross-version design, per type: folds over gauge groups (a gauge's Version D unit, else its Version C unit), because
  SWORD v16 and v17b ids differ. For each split, models are trained on Version C and Version D training gauges
  (notwin features) and evaluated as: A->B stale (train and calibrate on A, test on B), A->B recal (train on A,
  calibrate on B's gauged sites), B->B native.
Stop rule 1: for each configuration, mean realised SDR - alpha versus two standard errors over splits.
Outputs: results/robustness_v6/{per_split.csv, summary.csv, stoprule1.csv, report.txt}
"""
import io, sys, time, warnings
import numpy as np
import pandas as pd
import cupy as cp
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE, label_B
from check_failure_mode import features as features_base, conf_select
from features_v2 import features_v2, features_causal, twin_map
from prototype_site_level import evaluate
from prototype_score_sdr import site_scores, plugin_curve
from score_sdr_gpu import score_sdr_gpu
from run_frontier_v5 import mu_good

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
OUT = ROOT / "results" / "robustness_v6"; OUT.mkdir(parents=True, exist_ok=True)
TAU, ALPHAS, REPEATS, FOLDS, OOF = 0.25, (0.10, 0.20), 4, 5, 3
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def choose_lam(mu, bad, s, ns, lams, alpha, m):
    best = (0.0, lams[len(lams) // 2])
    for lam in lams:
        sc, L, kept = site_scores(mu, bad, s, ns, lam, m)
        so, fin, risk, ret = plugin_curve(sc, L, kept, len(mu))
        feas = np.nonzero(fin & (risk <= alpha))[0]
        if len(feas) and ret[feas.max()] > best[0]:
            best = (ret[feas.max()], lam)
    return best[1]


def select_rows(mu_T, bad_T, sT, mu_C, bad_C, sC, mu_U, bad_U, sU, ms, seed, tag, with_m3):
    rows, nsU = [], sU.max() + 1
    lams = np.unique(np.quantile(mu_T, np.linspace(0.02, 0.98, 30)))
    for a in ALPHAS:
        if with_m3:
            rows.append(dict(tag, alpha=a, m=np.nan, method="M3 pooled", **evaluate(conf_select(mu_C, bad_C, mu_U, a), bad_U, sU, nsU)))
        for m in ms:
            lam = choose_lam(mu_T, bad_T, sT, sT.max() + 1, lams, a, m)
            s_C, L_C, _ = site_scores(mu_C, bad_C, sC, sC.max() + 1, lam, m)
            s_U, _, _ = site_scores(mu_U, None, sU, nsU, lam, m)
            eC, eU = np.isfinite(s_C), np.isfinite(s_U)
            for prune in (None, "homo"):
                keep = np.zeros(len(mu_U), bool)
                if eC.any() and eU.any():
                    sel = score_sdr_gpu(L_C[eC], s_C[eC], s_U[eU], a, a, prune=prune, random_state=seed)
                    cert = np.zeros(nsU, bool); cert[np.nonzero(eU)[0][sel]] = True
                    keep = (mu_U >= lam) & cert[sU]
                rows.append(dict(tag, alpha=a, m=m, method=f"S6 SCoRE-SDR ({prune})", **evaluate(keep, bad_U, sU, nsU)))
    return rows


def subset(df, t, v, calval=False, tidal=False):
    d = df[(df.type == t) & (df.v == v)]
    if not calval:
        d = d[d.phase == "science"]
    if not tidal:
        d = d[~d.tidal_gauge]
    d = d.copy()
    d["n"] = d.groupby(SITE).wse.transform("size")
    d = d[d.n >= 5].copy()
    d["eB"], _ = label_B(d)
    d["unit"] = d.fid.astype("int64") // (10 if t == "reach" else 10000)
    return d


def build_X(d, t, fs, twin):
    if fs == "ext":
        X = features_v2(d, t, twin.loc[d.index])
    elif fs == "notwin":
        X = features_v2(d, t, twin.loc[d.index]).drop(columns=["twin_dwse", "has_twin"])
    else:
        X = features_causal(d, t)
    return X.replace([np.inf, -np.inf], np.nan)


def fact(x):
    return pd.factorize(x)[0]


def oof(fit_X, y, iT, grp_T, seed, rng):
    mu_T = np.empty(len(iT))
    for k, fu in enumerate(np.array_split(rng.permutation(np.unique(grp_T)), OOF)):
        held = np.isin(grp_T, fu)
        mu_T[held] = mu_good(fit_X, y, iT[~held], [iT[held]], seed * 10 + k)[0]
    return mu_T


def within(df, twin, t, v, design, fs, group="unit", ms=(3,), calval=False, tidal=False, with_m3=False):
    d = subset(df, t, v, calval, tidal)
    X = build_X(d, t, fs, twin)
    Xg = cp.asarray(X.to_numpy(np.float32))
    bad = (d.eB.abs() > TAU).to_numpy(); y = cp.asarray(bad.astype(np.int32))
    unit = d.unit.to_numpy(); site = d.groupby(SITE).ngroup().to_numpy()
    grp = unit if group == "unit" else unit // 10 ** 6
    groups = np.unique(grp)
    rows = []
    for q in range(REPEATS):
        folds = np.array_split(np.random.default_rng(5000 * q + 17 * (t == "node") + 3 * (v == "C")).permutation(groups), FOLDS)
        for f in range(FOLDS):
            U = np.isin(grp, folds[f]); rng = np.random.default_rng(100 * q + f); seed = 1000 * q + 10 * f
            gg = rng.permutation(np.unique(grp[~U])); half = len(gg) // 2
            iT, iC, iU = np.nonzero(np.isin(grp, gg[:half]))[0], np.nonzero(np.isin(grp, gg[half:]))[0], np.nonzero(U)[0]
            mu_C, mu_U = mu_good(Xg, y, iT, [iC, iU], seed)
            mu_T = oof(Xg, y, iT, grp[iT], seed, rng)
            tag = dict(design=design, type=t, v=v, rep=q, fold=f, n_groups=len(groups), n_features=X.shape[1])
            rows += select_rows(mu_T, bad[iT], fact(site[iT]), mu_C, bad[iC], fact(site[iC]),
                                mu_U, bad[iU], fact(site[iU]), ms, seed, tag, with_m3)
    return rows


def cross_version(df, twin, t):
    d = {v: subset(df, t, v) for v in ("C", "D")}
    X = {v: build_X(d[v], t, "notwin", twin) for v in d}
    cols = X["C"].columns.union(X["D"].columns)
    for v in X:
        X[v] = X[v].reindex(columns=cols)
        bits = [c for c in cols if c.startswith("bit")]
        X[v][bits] = X[v][bits].fillna(0)
    gmap = d["D"].groupby("gauge").unit.first().combine_first(d["C"].groupby("gauge").unit.first())
    grp = {v: d[v].gauge.map(gmap).to_numpy() for v in d}
    Xg = {v: cp.asarray(X[v].to_numpy(np.float32)) for v in d}
    bad = {v: (d[v].eB.abs() > TAU).to_numpy() for v in d}
    y = {v: cp.asarray(bad[v].astype(np.int32)) for v in d}
    site = {v: d[v].groupby(SITE).ngroup().to_numpy() for v in d}
    groups = np.unique(np.concatenate([grp["C"], grp["D"]]))
    rows = []
    for q in range(REPEATS):
        folds = np.array_split(np.random.default_rng(7000 * q + 17 * (t == "node")).permutation(groups), FOLDS)
        for f in range(FOLDS):
            rng = np.random.default_rng(300 * q + f); seed = 2000 * q + 10 * f
            Ug = folds[f]; gg = rng.permutation(np.setdiff1d(groups, Ug)); half = len(gg) // 2
            idx = {v: {k: np.nonzero(np.isin(grp[v], g))[0] for k, g in (("T", gg[:half]), ("C", gg[half:]), ("U", Ug))} for v in d}
            mu, lamT = {}, {}
            for A in d:                                   # model trained on version A, predictions on both versions
                import xgboost as xgb
                from run_frontier_v5 import XGB
                clf = xgb.XGBClassifier(random_state=seed, **XGB).fit(Xg[A][cp.asarray(idx[A]["T"])], y[A][cp.asarray(idx[A]["T"])])
                for B in d:
                    for k in ("C", "U"):
                        pr = clf.predict_proba(Xg[B][cp.asarray(idx[B][k])])[:, 1]
                        mu[(A, B, k)] = 1 - (cp.asnumpy(pr) if isinstance(pr, cp.ndarray) else np.asarray(pr))
                lamT[A] = oof(Xg[A], y[A], idx[A]["T"], grp[A][idx[A]["T"]], seed, rng)
            for A, B in (("C", "D"), ("D", "C")):
                for mode, calib_v, model_v in (("stale", A, A), ("recal", B, A), ("native", B, B)):
                    tag = dict(design=f"xver/{A}->{B} {mode}", type=t, v=B, rep=q, fold=f, n_groups=len(groups), n_features=len(cols))
                    iT = idx[model_v]["T"]
                    rows += select_rows(lamT[model_v], bad[model_v][iT], fact(site[model_v][iT]),
                                        mu[(model_v, calib_v, "C")], bad[calib_v][idx[calib_v]["C"]], fact(site[calib_v][idx[calib_v]["C"]]),
                                        mu[(model_v, B, "U")], bad[B][idx[B]["U"]], fact(site[B][idx[B]["U"]]),
                                        (3,), seed, tag, False)
    return rows


def main():
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
    twin = twin_map(df, pd.read_parquet(ROOT / "data" / "processed" / "cd_crosswalk_v1.parquet"))
    designs = [("unit/ext", dict(fs="ext", ms=(1, 3, 5), with_m3=True)), ("unit/notwin", dict(fs="notwin")),
               ("unit/causal", dict(fs="causal")), ("basin/ext", dict(fs="ext", group="basin")),
               ("unit/ext+calval", dict(fs="ext", calval=True)), ("unit/ext+tidal", dict(fs="ext", tidal=True))]
    rows = []
    for t in ("reach", "node"):
        for v in ("D", "C"):
            for design, kw in designs:
                t1 = time.time()
                r = within(df, twin, t, v, design, **kw)
                rows += r
                out(f"{t} {v} {design}: {len(r)} rows, groups {r[0]['n_groups']}, features {r[0]['n_features']}, "
                    f"{time.time() - t1:.0f} s (total {time.time() - t0:.0f} s)")
                pd.DataFrame(rows).to_csv(OUT / "per_split.csv", index=False)
        t1 = time.time()
        r = cross_version(df, twin, t)
        rows += r
        out(f"{t} cross-version: {len(r)} rows, {time.time() - t1:.0f} s (total {time.time() - t0:.0f} s)")
        pd.DataFrame(rows).to_csv(OUT / "per_split.csv", index=False)

    res = pd.DataFrame(rows)
    res["none"] = res.retention == 0
    res["exceed"] = res.site_fdp_kept > res.alpha
    res.to_csv(OUT / "per_split.csv", index=False)
    keys = ["design", "type", "v", "alpha", "m", "method"]
    g = res.groupby(keys, dropna=False)
    s = g.agg(retention=("retention", "mean"), site_coverage=("site_coverage", "mean"), pooled_fdp=("fdp", "mean"),
              sdr=("site_fdp_kept", "mean"), sdr_sd=("site_fdp_kept", "std"), n=("site_fdp_kept", "size"),
              share_gt_alpha=("exceed", "mean"), share_none=("none", "mean"))
    s["sdr_if_any"] = res[~res.none].groupby(keys, dropna=False).site_fdp_kept.mean()
    s.to_csv(OUT / "summary.csv")
    sr = s[s.index.get_level_values("method").str.startswith("S6")].copy()
    sr["excess"] = sr.sdr - sr.index.get_level_values("alpha")
    sr["two_se"] = 2 * sr.sdr_sd / np.sqrt(sr.n)
    sr["violation"] = sr.excess > sr.two_se
    sr[["sdr", "excess", "two_se", "violation", "n"]].to_csv(OUT / "stoprule1.csv")
    out(f"\nStop rule 1 (mean realised SDR - alpha > 2 SE): {int(sr.violation.sum())} of {len(sr)} S6 configurations flagged")
    if sr.violation.any():
        out(sr[sr.violation][["sdr", "excess", "two_se", "n"]].round(4).to_string())
    show = s[["retention", "site_coverage", "sdr", "sdr_if_any", "share_gt_alpha", "share_none"]]
    out("\n" + show.round(3).to_string())
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
