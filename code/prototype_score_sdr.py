"""Prototype v3 (13 Oct 2026): site-level selective deployment risk (SDR) via conformal risk-adjusted e-values + e-BH.

Framework: Bai & Jin (2026, arXiv 2603.24704), Definition 3.1 (E >= 0, E[L E] <= 1) and the e-BH theorem for SDR.
This is OUR construction inside that framework (fixed-threshold e-values, fixed-weight mixture), not a port of their
adaptive e-values. Units are sites; L_site = FDP among the site's kept observations (bias-relative labels, |eB| > TAU).
  site score  s_j(lam) = mean (1 - mu) over the site's observations with mu >= lam  (label-free; inf if fewer than m kept)
  e-value     E_j = sum_k w_k 1{s_j <= t_k} (n_B + 1) / (1 + sum_{i in B} L_i 1{s_i <= t_k})     (valid for fixed t_k, w_k)
  lam*, thresholds t_k and weights w_k (uniform over thresholds whose plug-in selective risk on A is <= alpha) come from
  calibration part A (40% of calibration units); e-values use part B (60%); e-BH at alpha over the held-out sites U.
Target: E[ average within-site FDP over certified sites ] <= alpha, i.e. the SDR with sites as units.
References recomputed on the same splits: M3, S1, S4 (m = 3).
"""
import io, sys, time, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE, label_B
from check_failure_mode import features, p_good, conf_select
from prototype_site_level import unit_losses, crc_index, evaluate
from prototype_selective import pair_arrays, selective_ltt, certified_keep, GATE_Q

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
OUT = ROOT / "results" / "site_level_proto_v3"; OUT.mkdir(parents=True, exist_ok=True)
TAU, ALPHAS, REPEATS, DELTA = 0.25, (0.10, 0.20), 20, 0.10
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def site_scores(mu, bad, sidx, ns, lam, m):
    keep = mu >= lam
    kept = np.bincount(sidx, weights=keep, minlength=ns)
    ok = kept >= m
    s = np.where(ok, np.bincount(sidx, weights=keep * (1 - mu), minlength=ns) / np.maximum(kept, 1), np.inf)
    L = None if bad is None else np.where(ok, np.bincount(sidx, weights=keep & bad, minlength=ns) / np.maximum(kept, 1), 0.0)
    return s, L, kept


def plugin_curve(s, L, kept, n_rows):
    """Sites ordered by score: plug-in selective risk and retained-row share if the first k sites are certified."""
    o = np.argsort(s, kind="stable")
    fin = np.isfinite(s[o])
    risk = np.cumsum(L[o]) / np.arange(1, len(s) + 1)
    ret = np.cumsum(kept[o]) / n_rows
    return s[o], fin, risk, ret


def sdr_ebh(mu_A, bad_A, sA, nsA, mu_B, bad_B, sB, nsB, mu_U, sU, nsU, lams, m, alpha):
    best = None
    for lam in lams:                                     # choose lam on A only
        s, L, kept = site_scores(mu_A, bad_A, sA, nsA, lam, m)
        so, fin, risk, ret = plugin_curve(s, L, kept, len(mu_A))
        feas = fin & (risk <= alpha)
        r = ret[np.nonzero(feas)[0].max()] if feas.any() else 0.0
        if best is None or r > best[0]:
            best = (r, lam)
    lam = best[1]
    s, L, kept = site_scores(mu_A, bad_A, sA, nsA, lam, m)
    so, fin, risk, _ = plugin_curve(s, L, kept, len(mu_A))
    cand = np.unique(np.quantile(so[fin], np.linspace(0.05, 1, 20))) if fin.any() else np.array([])
    feas_t = [t for t in cand if (risk[np.searchsorted(so, t, side="right") - 1] <= alpha)]
    if not feas_t:
        return np.zeros(len(mu_U), bool), lam, 0
    s_B, L_B, _ = site_scores(mu_B, bad_B, sB, nsB, lam, m)
    s_U, _, _ = site_scores(mu_U, None, sU, nsU, lam, m)
    E = np.zeros(nsU)
    for t in feas_t:
        E += (s_U <= t) * (nsB + 1) / (1 + np.sum(L_B * (s_B <= t))) / len(feas_t)
    o = np.argsort(-E, kind="stable")
    ok = np.nonzero(E[o] >= nsU / (alpha * np.arange(1, nsU + 1)))[0]
    if not len(ok):
        return np.zeros(len(mu_U), bool), lam, len(feas_t)
    cert = E >= nsU / (alpha * (ok.max() + 1))
    return (mu_U >= lam) & cert[sU], lam, len(feas_t)


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
                rng = np.random.default_rng(10007 * r + (t == "node") * 7 + (v == "C") * 3)   # same splits as v1/v2
                gu = rng.permutation(units); g_units = gu[: len(gu) // 2]
                U = ~np.isin(unit, g_units)
                half = len(g_units) // 2
                tr, ca = np.isin(unit, g_units[:half]), np.isin(unit, g_units[half:])
                mu_ca, mu_U = p_good(Xn[tr], bad[tr], [Xn[ca], Xn[U]], r)
                bad_ca, bad_U = bad[ca], bad[U]
                u_ca = pd.factorize(unit[ca])[0]
                s_U = pd.factorize(site[U])[0]; ns = s_U.max() + 1
                inA = np.isin(u_ca, rng.permutation(np.unique(u_ca))[: int(0.4 * (u_ca.max() + 1))])
                uA, uB = pd.factorize(u_ca[inA])[0], pd.factorize(u_ca[~inA])[0]
                sA, sB = pd.factorize(site[ca][inA])[0], pd.factorize(site[ca][~inA])[0]
                grid = np.concatenate([[np.inf], np.unique(np.quantile(mu_ca, np.linspace(0, 1, 401)))[::-1]])
                L = unit_losses(mu_ca, bad_ca, u_ca, u_ca.max() + 1, grid)
                med_ca = pd.Series(mu_ca).groupby(site[ca]).transform("median").to_numpy()
                med_U = pd.Series(mu_U).groupby(site[U]).transform("median").to_numpy()
                gates = [-np.inf if q is None else np.quantile(med_ca[inA], q) for q in GATE_Q]
                lams80 = np.unique(np.quantile(mu_ca, np.linspace(0.02, 1, 80)))[::-1]
                FA, CA, _ = pair_arrays(mu_ca[inA], bad_ca[inA], uA, uA.max() + 1, med_ca[inA], gates, lams80, 3)
                FB, CB, RB = pair_arrays(mu_ca[~inA], bad_ca[~inA], uB, uB.max() + 1, med_ca[~inA], gates, lams80, 3)
                lams30 = np.unique(np.quantile(mu_ca, np.linspace(0.02, 0.98, 30)))
                base = dict(type=t, v=v, rep=r)
                for a in ALPHAS:
                    rows.append(dict(base, alpha=a, method="M3 pooled conformal selection",
                                     **evaluate(conf_select(mu_ca, bad_ca, mu_U, a), bad_U, s_U, ns)))
                    k1, _ = crc_index(L, a)
                    rows.append(dict(base, alpha=a, method="S1 site-level CRC (abstain = 0)", **evaluate(mu_U >= grid[k1], bad_U, s_U, ns)))
                    p = selective_ltt(FA, CA, FB, CB, RB, a, DELTA)
                    keep = np.zeros(len(mu_U), bool) if p is None else certified_keep(
                        mu_U, med_U, s_U, ns, gates[p // len(lams80)], lams80[p % len(lams80)], 3)
                    rows.append(dict(base, alpha=a, method="S4 selective LTT (m=3)", **evaluate(keep, bad_U, s_U, ns)))
                    for m in (1, 3):
                        keep, lam, nt = sdr_ebh(mu_ca[inA], bad_ca[inA], sA, sA.max() + 1, mu_ca[~inA], bad_ca[~inA], sB, sB.max() + 1,
                                                mu_U, s_U, ns, lams30, m, a)
                        rows.append(dict(base, alpha=a, method=f"S5 site SDR e-BH (m={m})", n_thresholds=nt, **evaluate(keep, bad_U, s_U, ns)))
            out(f"   done {t} {v} at {time.time() - t0:.0f} s")
    res = pd.DataFrame(rows)
    res["sel_exceed"] = res.site_fdp_kept > res.alpha
    res["none"] = res.retention == 0
    res.to_csv(OUT / "per_split.csv", index=False)
    s = res.groupby(["type", "v", "alpha", "method"]).agg(
        retention=("retention", "mean"), site_coverage=("site_coverage", "mean"), pooled_fdp=("fdp", "mean"),
        site_fdp_where_kept=("site_fdp_kept", "mean"), sd=("site_fdp_kept", "std"),
        share_splits_gt_alpha=("sel_exceed", "mean"), share_splits_none=("none", "mean"))
    s.to_csv(OUT / "summary.csv")
    out("\nsite_fdp_where_kept = mean over held-out sites that keep data of their bad share; %d repeats, tau %.2f m" % (REPEATS, TAU))
    out(s.round(3).to_string())
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
