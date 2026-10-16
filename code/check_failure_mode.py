"""Failure-mode check (13 Oct 2026): does tuning a screening rule without site-level separation
keep its error rate at new (ungauged-like) sites?

Data: data/processed/paired_clean_v1.parquet; rivers; science phase; tidal gauges excluded;
label B recomputed on the subset (swotio.label_B); sites with >= 5 pairs. bad = |eB| > TAU.
Each repeat splits river units (reach-level SWORD id) 50/50 into G ("gauged") and U ("ungauged").
  M1 grid  : flag-threshold grid tuned in-sample on G (max retention s.t. violation <= alpha), applied to U.
  M2 obs   : learned score; G observations split 40/30/30 train/cal/test at random (same sites in all parts);
             conformal selection (Jin & Candes p-values + BH) at alpha; applied to U.
             'naive_eval' = FDP on G's held-out test observations, i.e. what an observation-level study would report.
  M3 site  : learned score; G units split 50/50 into train and calibration units; same selection; applied to U.
Outputs: results/failure_check_v1/{per_split.csv, summary.csv, report.txt}
"""
import io, itertools, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE, label_B

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
OUT = ROOT / "results" / "failure_check_v1"; OUT.mkdir(parents=True, exist_ok=True)
TAU, ALPHAS, REPEATS = 0.25, (0.10, 0.20), 20
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def hampel(d):
    """3-point Hampel residual on each site's own SWOT series (no gauge data)."""
    s = d.sort_values(SITE + ["t_swot"])
    g = s.groupby(SITE).wse
    arr = np.vstack([g.shift(1).to_numpy(), s.wse.to_numpy(), g.shift(-1).to_numpy()])
    med = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - med), axis=0)
    r = pd.Series(np.abs(s.wse.to_numpy() - med), index=s.index)
    f = pd.Series(r.to_numpy() > 3 * 1.4826 * mad, index=s.index)
    return r.reindex(d.index), f.reindex(d.index)


def features(d, t):
    X = pd.DataFrame(index=d.index)
    for c in ["wse_u", "wse_r_u", "qflag", "dark_frac", "area_ratio", "pdiff", "xovr_cal_q", "p_width", "p_n_ch_mod"]:
        X[c] = d[c].astype(float)
    qb = d.qbit.fillna(0).astype("int64").to_numpy()
    for b in range(27):
        col = ((qb >> b) & 1).astype(float)
        if col.std() > 0:
            X[f"bit{b}"] = col
    X["ice"] = d.ice_clim_f.replace(-999, np.nan).astype(float)
    X["axtrk"] = d.xtrk_dist.abs().astype(float)
    X["width_ratio"] = (d.width / d.p_width).astype(float)
    X["dam"] = (d.p_dam_id.fillna(0) > 0).astype(float)
    X["hampel_r"], hf = hampel(d)
    X["hampel_flag"] = hf.astype(float)
    if t == "reach":
        X["good_nod_frac"] = (d.n_good_nod / d.p_n_nodes).astype(float)
        X["slope_u"] = d.slope_u.astype(float); X["partial"] = d.partial_f.astype(float)
    else:
        X["n_good_pix"] = d.n_good_pix.astype(float)
    return X


def grid_masks(d, X, t):
    """Flag filters as in the USGS grid; the last level of each is 'not applied'. NaN fails an applied filter."""
    ice = d.ice_clim_f.where(d.ice_clim_f != -999, 0)
    ax = d.xtrk_dist.abs()
    F = {"quality": [d.qflag <= l for l in (0, 1, 2)] + [None],
         "xovr": [d.xovr_cal_q <= l for l in (0, 1)] + [None],
         "dark": [d.dark_frac <= l for l in (0.25, 0.5)] + [None],
         "pdiff": [d.pdiff <= l for l in (10.0, 50.0)] + [None],
         "area": [d.area_ratio >= l for l in (0.75, 0.5)] + [None],
         "ice": [ice <= l for l in (0, 1)] + [None],
         "xtrk": [(ax >= 10000) & (ax <= 60000), None],
         "hampel": [X.hampel_flag == 0, None]}
    if t == "reach":
        F["partial"] = [d.partial_f == 0, None]
    return {k: [None if m is None else m.to_numpy(bool) for m in v] for k, v in F.items()}


def tune_grid(F, bad, G, alphas):
    """In-sample: for each alpha, the combination with the most kept G rows whose G violation rate <= alpha."""
    names = list(F)
    Fg = {k: [None if m is None else m[G] for m in F[k]] for k in names}
    badG = bad[G]
    best = {a: (0, None) for a in alphas}
    for combo in itertools.product(*[range(len(F[k])) for k in names]):
        m = np.ones(G.sum(), bool)
        for k, i in zip(names, combo):
            if Fg[k][i] is not None:
                m &= Fg[k][i]
        kept = int(m.sum())
        if kept == 0:
            continue
        v = badG[m].mean()
        for a in alphas:
            if v <= a and kept > best[a][0]:
                best[a] = (kept, combo)
    return {a: best[a][1] for a in alphas}


def apply_grid(F, combo, U):
    m = np.ones(U.sum(), bool)
    if combo is None:
        return np.zeros(U.sum(), bool)
    for k, i in zip(F, combo):
        if F[k][i] is not None:
            m &= F[k][i][U]
    return m


def conf_select(mu_cal, bad_cal, mu_test, alpha):
    """Conformal selection of 'good' rows (null: bad). p_j = (1 + #{cal bad with mu >= mu_j}) / (n + 1), then BH."""
    n = len(mu_cal)
    nulls = np.sort(mu_cal[bad_cal])
    p = (1 + len(nulls) - np.searchsorted(nulls, mu_test, side="left")) / (n + 1)
    m = len(p); o = np.argsort(p, kind="stable")
    ok = np.nonzero(p[o] <= alpha * np.arange(1, m + 1) / m)[0]
    sel = np.zeros(m, bool)
    if len(ok):
        sel[o[: ok.max() + 1]] = True
    return sel


def p_good(X, y, Xs, seed):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed).fit(X, y)
    col = list(clf.classes_).index(False)
    return [clf.predict_proba(x)[:, col] for x in Xs]


def metrics(sel, bad, site):
    kept = int(sel.sum())
    if kept == 0:
        return dict(retention=0.0, fdp=0.0, site_fdp=0.0)
    sf = pd.Series(bad[sel]).groupby(site[sel]).mean().mean()
    return dict(retention=sel.mean(), fdp=bad[sel].mean(), site_fdp=sf)


def main():
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
    rows = []
    for t in ("reach", "node"):
        for v in ("D", "C"):
            d = df[(df.type == t) & (df.v == v) & (df.phase == "science") & (~df.tidal_gauge)].copy()
            d["n"] = d.groupby(SITE).wse.transform("size")
            d = d[d.n >= 5].copy()
            d["eB"], _ = label_B(d)
            bad = (d.eB.abs() > TAU).to_numpy()
            unit = (d.fid.astype("int64") // (10 if t == "reach" else 10000)).to_numpy()
            site = d.groupby(SITE).ngroup().to_numpy()
            X = features(d, t); F = grid_masks(d, X, t); Xn = X.to_numpy(float)
            units = np.unique(unit)
            out(f"{t} {v}: rows {len(d)}, sites {site.max() + 1}, units {len(units)}, bad rate {bad.mean():.3f}, features {X.shape[1]}")
            for r in range(REPEATS):
                rng = np.random.default_rng(10007 * r + (t == "node") * 7 + (v == "C") * 3)
                gu = rng.permutation(units); g_units = gu[: len(gu) // 2]
                G = np.isin(unit, g_units); U = ~G
                bU, sU = bad[U], site[U]
                combos = tune_grid(F, bad, G, ALPHAS)
                for a in ALPHAS:
                    rows.append(dict(type=t, v=v, rep=r, alpha=a, method="M1 grid in-sample", **metrics(apply_grid(F, combos[a], U), bU, sU)))
                idx = rng.permutation(np.nonzero(G)[0]); n = len(idx)
                tr, ca, te = idx[: int(.4 * n)], idx[int(.4 * n): int(.7 * n)], idx[int(.7 * n):]
                mu_ca, mu_te, mu_U = p_good(Xn[tr], bad[tr], [Xn[ca], Xn[te], Xn[U]], r)
                for a in ALPHAS:
                    naive = metrics(conf_select(mu_ca, bad[ca], mu_te, a), bad[te], site[te])["fdp"]
                    rows.append(dict(type=t, v=v, rep=r, alpha=a, method="M2 obs-level conformal", naive_eval_fdp=naive,
                                     **metrics(conf_select(mu_ca, bad[ca], mu_U, a), bU, sU)))
                half = len(g_units) // 2
                tr_m, ca_m = np.isin(unit, g_units[:half]), np.isin(unit, g_units[half:])
                mu_ca, mu_U = p_good(Xn[tr_m], bad[tr_m], [Xn[ca_m], Xn[U]], r)
                for a in ALPHAS:
                    rows.append(dict(type=t, v=v, rep=r, alpha=a, method="M3 site-disjoint conformal",
                                     **metrics(conf_select(mu_ca, bad[ca_m], mu_U, a), bU, sU)))
            out(f"   done {t} {v} at {time.time() - t0:.0f} s")
    res = pd.DataFrame(rows)
    res["exceed"] = res.fdp > res.alpha
    res.to_csv(OUT / "per_split.csv", index=False)
    s = res.groupby(["type", "v", "alpha", "method"]).agg(
        fdp_mean=("fdp", "mean"), fdp_sd=("fdp", "std"), share_splits_fdp_gt_alpha=("exceed", "mean"),
        site_fdp_mean=("site_fdp", "mean"), retention_mean=("retention", "mean"), naive_eval_fdp=("naive_eval_fdp", "mean"))
    s.to_csv(OUT / "summary.csv")
    out("\nFDP = share of kept observations at held-out units with |eB| > %.2f m; %d repeats" % (TAU, REPEATS))
    out(s.round(3).to_string())
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
