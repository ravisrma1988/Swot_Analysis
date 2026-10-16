"""Prototype (13 Oct 2026): what does a SITE-level guarantee cost in retention?

Target: for a new site, expected share of bad (|eB| > TAU) among its kept observations <= alpha
(share = 0 if a site keeps nothing). Same data, score model and unit-level 50/50 G/U splits as
check_failure_mode.py; G units split 50/50 into score-training and calibration units.
  M3  pooled conformal selection (BH) at alpha                         [reference, pooled target]
  S1  site-level conformal risk control on per-unit FDP, monotone envelope, single threshold
  S2  site-level Learn-then-Test, Hoeffding-Bentkus p-values, fixed sequence, delta = 0.1 (high-probability)
  S3  site-aware: gate on the site's median score (label-free) + CRC threshold; gate tuned on 40% of
      calibration units, CRC on the other 60%
Metrics on U: retention, pooled FDP, site-averaged FDP (zeros for sites keeping nothing),
site-averaged FDP among sites keeping >= 1, share of sites keeping >= 1.
"""
import io, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import binom
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE, label_B
from check_failure_mode import features, p_good, conf_select

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
OUT = ROOT / "results" / "site_level_proto_v1"; OUT.mkdir(parents=True, exist_ok=True)
TAU, ALPHAS, REPEATS, DELTA = 0.25, (0.10, 0.20), 20, 0.10
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def unit_losses(mu, bad, uidx, nu, grid, gate=None):
    """Per-unit FDP for each threshold in grid (strict -> loose). Row k, column unit."""
    L = np.zeros((len(grid), nu))
    for k, lam in enumerate(grid):
        keep = mu >= lam if gate is None else (mu >= lam) & gate
        kept = np.bincount(uidx, weights=keep, minlength=nu)
        b = np.bincount(uidx, weights=keep & bad, minlength=nu)
        L[k] = np.where(kept > 0, b / np.maximum(kept, 1), 0.0)
    return L


def crc_index(L, alpha):
    env = np.maximum.accumulate(L, axis=0)          # monotone upper envelope, nondecreasing as threshold loosens
    n = L.shape[1]
    risk = env.mean(axis=1) * n / (n + 1) + 1 / (n + 1)
    bad_k = np.nonzero(risk > alpha)[0]
    return (bad_k[0] - 1 if len(bad_k) else len(risk) - 1), risk


def hb_p(r, n, alpha):
    a = min(r, alpha)
    h = (a * np.log(a / alpha) if a > 0 else 0.0) + (1 - a) * np.log((1 - a) / (1 - alpha))
    return min(np.exp(-n * h), np.e * binom.cdf(np.ceil(n * r), n, alpha))


def ltt_index(L, alpha, delta):
    n, R, k_sel = L.shape[1], L.mean(axis=1), 0
    for k in range(len(R)):
        if hb_p(R[k], n, alpha) <= delta:
            k_sel = k
        else:
            break
    return k_sel


def evaluate(keep, bad, sidx, ns):
    kept = np.bincount(sidx, weights=keep, minlength=ns)
    b = np.bincount(sidx, weights=keep & bad, minlength=ns)
    f = np.where(kept > 0, b / np.maximum(kept, 1), 0.0)
    return dict(retention=keep.mean(), fdp=bad[keep].mean() if keep.any() else 0.0, site_fdp_all=f.mean(),
                site_fdp_kept=f[kept > 0].mean() if (kept > 0).any() else 0.0, site_coverage=(kept > 0).mean())


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
            Xn = features(d, t).to_numpy(float)
            units = np.unique(unit)
            out(f"{t} {v}: rows {len(d)}, units {len(units)}, bad rate {bad.mean():.3f}")
            for r in range(REPEATS):
                rng = np.random.default_rng(10007 * r + (t == "node") * 7 + (v == "C") * 3)   # same G/U as the failure check
                gu = rng.permutation(units); g_units = gu[: len(gu) // 2]
                U = ~np.isin(unit, g_units)
                half = len(g_units) // 2
                tr, ca = np.isin(unit, g_units[:half]), np.isin(unit, g_units[half:])
                mu_ca, mu_U = p_good(Xn[tr], bad[tr], [Xn[ca], Xn[U]], r)
                bad_ca, bad_U = bad[ca], bad[U]
                u_ca = pd.factorize(unit[ca])[0]
                s_U = pd.factorize(site[U])[0]; ns = s_U.max() + 1
                grid = np.concatenate([[np.inf], np.unique(np.quantile(mu_ca, np.linspace(0, 1, 401)))[::-1]])
                L = unit_losses(mu_ca, bad_ca, u_ca, u_ca.max() + 1, grid)
                med_ca = pd.Series(mu_ca).groupby(site[ca]).transform("median").to_numpy()
                med_U = pd.Series(mu_U).groupby(site[U]).transform("median").to_numpy()
                inA = np.isin(u_ca, rng.permutation(np.unique(u_ca))[: int(0.4 * (u_ca.max() + 1))])
                uA, uB = pd.factorize(u_ca[inA])[0], pd.factorize(u_ca[~inA])[0]
                base = dict(type=t, v=v, rep=r)
                for a in ALPHAS:
                    rows.append(dict(base, alpha=a, method="M3 pooled conformal selection",
                                     **evaluate(conf_select(mu_ca, bad_ca, mu_U, a), bad_U, s_U, ns)))
                    k1, _ = crc_index(L, a)
                    rows.append(dict(base, alpha=a, method="S1 site-level CRC", **evaluate(mu_U >= grid[k1], bad_U, s_U, ns)))
                    k2 = ltt_index(L, a, DELTA)
                    rows.append(dict(base, alpha=a, method="S2 site-level LTT (delta 0.1)", **evaluate(mu_U >= grid[k2], bad_U, s_U, ns)))
                    best = (-1.0, -np.inf, None)
                    for q in (None, 0.1, 0.2, 0.3, 0.4, 0.5):
                        thr = -np.inf if q is None else np.quantile(med_ca[inA], q)
                        gA = med_ca[inA] >= thr
                        kA, _ = crc_index(unit_losses(mu_ca[inA], bad_ca[inA], uA, uA.max() + 1, grid, gA), a)
                        retA = ((mu_ca[inA] >= grid[kA]) & gA).mean()
                        if retA > best[0]:
                            best = (retA, thr, q)
                    gB = med_ca[~inA] >= best[1]
                    kB, _ = crc_index(unit_losses(mu_ca[~inA], bad_ca[~inA], uB, uB.max() + 1, grid, gB), a)
                    rows.append(dict(base, alpha=a, method="S3 site-aware gate + CRC", gate_q=best[2],
                                     **evaluate((mu_U >= grid[kB]) & (med_U >= best[1]), bad_U, s_U, ns)))
            out(f"   done {t} {v} at {time.time() - t0:.0f} s")
    res = pd.DataFrame(rows)
    res["site_exceed"] = res.site_fdp_all > res.alpha
    res.to_csv(OUT / "per_split.csv", index=False)
    s = res.groupby(["type", "v", "alpha", "method"]).agg(
        retention=("retention", "mean"), pooled_fdp=("fdp", "mean"), site_fdp=("site_fdp_all", "mean"),
        site_fdp_sd=("site_fdp_all", "std"), share_splits_site_fdp_gt_alpha=("site_exceed", "mean"),
        site_fdp_if_kept=("site_fdp_kept", "mean"), site_coverage=("site_coverage", "mean"))
    s.to_csv(OUT / "summary.csv")
    out("\nsite_fdp = mean over held-out sites of the share bad among kept (0 if none kept); %d repeats, tau %.2f m" % (REPEATS, TAU))
    out(s.round(3).to_string())
    g = res[res.method.str.startswith("S3")].groupby(["type", "v", "alpha"]).gate_q.value_counts(dropna=False)
    out("\nS3 gate quantile chosen (count of splits):"); out(g.to_string())
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
