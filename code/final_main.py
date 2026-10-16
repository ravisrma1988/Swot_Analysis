"""Final main run (14 Oct 2026), GPU-bound, frozen settings (tuned model from results/final/model_config.json).

North America, Version D and C, reach and node, science phase, non-tidal, >= 5 pairs, label B; tau in {0.10, 0.25, 0.50};
4 repeats x 5-fold over units (same seeds as frontier_v5): U = held-out fold; G split 50/50 into T (score, lambda via
3-fold OOF) and C (calibration). Alpha grid {0.05, 0.10, 0.15, 0.20, 0.25, 0.30} (validity curve).
Methods: B0 no screening; B1 summary flag <= 0 and <= 1; M1 USGS-style flag grid tuned in-sample on G (GPU);
M3 pooled conformal selection; S1 site-level CRC (abstention counted as zero); S6 SCoRE-SDR (prune None, 'homo').
Secondary: label A (absolute EGM2008 error), Version D, tau 0.25, alpha {0.10, 0.20}, S6 and M3.
Outputs: results/final/main_{per_split,summary}.csv, labelA_{per_split,summary}.csv, main_report.txt
"""
import io, itertools, sys, time, warnings
import numpy as np
import pandas as pd
import cupy as cp
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE
from check_failure_mode import conf_select, grid_masks
from features_v2 import features_v2, twin_map
from prototype_site_level import evaluate, unit_losses, crc_index
from prototype_score_sdr import site_scores
from score_sdr_gpu import score_sdr_gpu
from run_frontier_v5 import mu_good, XGB
from run_robustness_v6 import subset, oof, choose_lam, fact

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "final"; OUT.mkdir(parents=True, exist_ok=True)
TAUS, ALPHAS, REPEATS, FOLDS, M = (0.10, 0.25, 0.50), (0.05, 0.10, 0.15, 0.20, 0.25, 0.30), 4, 5, 3
SMOKE = bool(__import__("os").environ.get("SMOKE"))
TYPES, VERSIONS, NF = (("reach",), ("D",), 1) if SMOKE else (("reach", "node"), ("D", "C"), FOLDS)
if SMOKE:
    TAUS, REPEATS = (0.25,), 1
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def gpu_grid(F, bad, G, U, alphas):
    """USGS-style grid on the GPU: per alpha, the combination with most kept G rows whose G violation rate <= alpha."""
    names = list(F)
    Fg = {k: [None if m is None else cp.asarray(m[G]) for m in F[k]] for k in names}
    badg = cp.asarray(bad[G]); n = int(G.sum())
    best = {a: (0, None) for a in alphas}
    for combo in itertools.product(*[range(len(F[k])) for k in names]):
        m = cp.ones(n, dtype=bool)
        for k, i in zip(names, combo):
            if Fg[k][i] is not None:
                m &= Fg[k][i]
        kept = int(m.sum())
        if kept == 0:
            continue
        v = float((badg & m).sum()) / kept
        for a in alphas:
            if v <= a and kept > best[a][0]:
                best[a] = (kept, combo)
    keeps = {}
    for a in alphas:
        combo = best[a][1]
        if combo is None:
            keeps[a] = np.zeros(int(U.sum()), bool); continue
        k_ = np.ones(int(U.sum()), bool)
        for k, i in zip(names, combo):
            if F[k][i] is not None:
                k_ &= F[k][i][U]
        keeps[a] = k_
    return keeps


def s6(mu_C, bad_C, sC, mu_U, sU, lam, a, prune, seed):
    s_C, L_C, _ = site_scores(mu_C, bad_C, sC, sC.max() + 1, lam, M)
    s_U, _, _ = site_scores(mu_U, None, sU, sU.max() + 1, lam, M)
    eC, eU = np.isfinite(s_C), np.isfinite(s_U)
    keep = np.zeros(len(mu_U), bool)
    if eC.any() and eU.any():
        sel = score_sdr_gpu(L_C[eC], s_C[eC], s_U[eU], a, a, prune=prune, random_state=seed)
        cert = np.zeros(sU.max() + 1, bool); cert[np.nonzero(eU)[0][sel]] = True
        keep = (mu_U >= lam) & cert[sU]
    return keep


def run_split(Xg, y, bad, unit, site, qflag, F, U_units, rng, seed, tag, alphas, baselines=True, m3=True):
    U = np.isin(unit, U_units)
    gg = rng.permutation(np.unique(unit[~U])); half = len(gg) // 2
    iT, iC, iU = np.nonzero(np.isin(unit, gg[:half]))[0], np.nonzero(np.isin(unit, gg[half:]))[0], np.nonzero(U)[0]
    mu_C, mu_U = mu_good(Xg, y, iT, [iC, iU], seed)
    mu_T = oof(Xg, y, iT, unit[iT], seed, rng)
    sT, sC, sU = fact(site[iT]), fact(site[iC]), fact(site[iU]); ns = sU.max() + 1
    rows = []
    add = lambda a, meth, keep: rows.append(dict(tag, alpha=a, method=meth, **evaluate(keep, bad[iU], sU, ns)))
    if baselines:
        add(np.nan, "B0 no screening", np.ones(len(iU), bool))
        for k in (0, 1):
            add(np.nan, f"B1 summary flag <= {k}", (qflag[iU] <= k))
        keeps = gpu_grid(F, bad, ~U, U, alphas)
        for a in alphas:
            add(a, "M1 USGS-style grid (in-sample)", keeps[a])
        uC = fact(unit[iC])
        grid = np.concatenate([[np.inf], np.unique(np.quantile(mu_C, np.linspace(0, 1, 401)))[::-1]])
        L = unit_losses(mu_C, bad[iC], uC, uC.max() + 1, grid)
        for a in alphas:
            k1, _ = crc_index(L, a)
            add(a, "S1 site CRC (abstain = 0)", mu_U >= grid[k1])
    if m3:
        for a in alphas:
            add(a, "M3 pooled conformal", conf_select(mu_C, bad[iC], mu_U, a))
    lams = np.unique(np.quantile(mu_T, np.linspace(0.02, 0.98, 30)))
    for a in alphas:
        lam = choose_lam(mu_T, bad[iT], sT, sT.max() + 1, lams, a, M)
        for prune in (None, "homo"):
            add(a, f"S6 SCoRE-SDR ({prune})", s6(mu_C, bad[iC], sC, mu_U, sU, lam, a, prune, seed))
    return rows


def summarise(res, keys, name):
    res["none"] = res.retention == 0
    res["exceed"] = res.site_fdp_kept > res.alpha
    res.to_csv(OUT / f"{name}_per_split.csv", index=False)
    g = res.groupby(keys, dropna=False)
    s = g.agg(retention=("retention", "mean"), site_coverage=("site_coverage", "mean"), pooled_fdp=("fdp", "mean"),
              site_fdp_abstain0=("site_fdp_all", "mean"), sdr=("site_fdp_kept", "mean"), sdr_sd=("site_fdp_kept", "std"),
              n=("site_fdp_kept", "size"), share_gt_alpha=("exceed", "mean"), share_none=("none", "mean"))
    s["excess_over_2se"] = (s.sdr - s.index.get_level_values("alpha")) - 2 * s.sdr_sd / np.sqrt(s.n)
    s.to_csv(OUT / f"{name}_summary.csv")
    return s


def main():
    t0 = time.time()
    out(f"model settings: { {k: v for k, v in XGB.items() if k not in ('device', 'tree_method', 'objective')} }")
    df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
    twin = twin_map(df, pd.read_parquet(ROOT / "data" / "processed" / "cd_crosswalk_v1.parquet"))
    rows = []
    for t in TYPES:
        for v in VERSIONS:
            d = subset(df, t, v)
            X = features_v2(d, t, twin.loc[d.index]).replace([np.inf, -np.inf], np.nan)
            Xg = cp.asarray(X.to_numpy(np.float32)); F = grid_masks(d, X, t)
            unit, site, qflag = d.unit.to_numpy(), d.groupby(SITE).ngroup().to_numpy(), d.qflag.to_numpy()
            units = np.unique(unit)
            for tau in TAUS:
                bad = (d.eB.abs() > tau).to_numpy(); y = cp.asarray(bad.astype(np.int32))
                for q in range(REPEATS):
                    folds = np.array_split(np.random.default_rng(5000 * q + 17 * (t == "node") + 3 * (v == "C")).permutation(units), FOLDS)
                    for f in range(NF):
                        rows += run_split(Xg, y, bad, unit, site, qflag, F, folds[f], np.random.default_rng(100 * q + f),
                                          1000 * q + 10 * f, dict(type=t, v=v, tau=tau, rep=q, fold=f), ALPHAS)
                out(f"{t} {v} tau {tau}: bad rate {bad.mean():.3f}; total {time.time() - t0:.0f} s")
                pd.DataFrame(rows).to_csv(OUT / "main_per_split.csv", index=False)
    s = summarise(pd.DataFrame(rows), ["type", "v", "tau", "alpha", "method"], "main")
    s6r = s[s.index.get_level_values("method").str.startswith("S6")]
    out(f"stop rule 1 (S6, excess > 2 SE): {int((s6r.excess_over_2se > 0).sum())} of {len(s6r)} configurations flagged")
    rowsA = []
    for t in TYPES:
        d = subset(df, t, "D")
        d = d[d.eA.notna()].copy(); d["n"] = d.groupby(SITE).wse.transform("size"); d = d[d.n >= 5].copy()
        X = features_v2(d, t, twin.loc[d.index]).replace([np.inf, -np.inf], np.nan)
        Xg = cp.asarray(X.to_numpy(np.float32)); unit, site = d.unit.to_numpy(), d.groupby(SITE).ngroup().to_numpy()
        units = np.unique(unit)
        for tauA in (0.25, 0.50, 1.00):
            bad = (d.eA.abs() > tauA).to_numpy(); y = cp.asarray(bad.astype(np.int32))
            out(f"label A {t} tau {tauA}: {len(d)} pairs, {site.max() + 1} sites, bad rate {bad.mean():.3f}")
            for q in range(REPEATS):
                folds = np.array_split(np.random.default_rng(5000 * q + 17 * (t == "node")).permutation(units), FOLDS)
                for f in range(NF):
                    rowsA += run_split(Xg, y, bad, unit, site, d.qflag.to_numpy(), None, folds[f], np.random.default_rng(100 * q + f),
                                       1000 * q + 10 * f, dict(type=t, v="D", tau=tauA, rep=q, fold=f), (0.10, 0.20), baselines=False)
    sA = summarise(pd.DataFrame(rowsA), ["type", "tau", "alpha", "method"], "labelA")
    out("\nlabel A summary:\n" + sA.round(3).to_string())
    out(f"\ndone in {time.time() - t0:.0f} s")
    (OUT / "main_report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
