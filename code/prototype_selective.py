"""Prototype v2 (13 Oct 2026): SELECTIVE site-level risk control.

Target: E[FDP_site | site keeps data] <= alpha, reported together with site coverage.
Per-site loss  l = FDP * 1{certified} + alpha * 1{abstains}, so E[l] <= alpha  <=>  E[FDP | certified] <= alpha
(abstention counts as at-target risk instead of zero). A site is certified under rule (g, lam, m) if its median
score (label-free) >= g and at least m of its observations have score >= lam; non-certified sites keep nothing.
  S4(m)  Learn-then-Test over the (g, lam) grid: 40% of calibration units order the pairs by estimated risk,
         fixed-sequence Hoeffding-Bentkus tests on the other 60% at delta = 0.1, stop at the first failure,
         choose the certified pair with the highest retention. m in {1, 3}.
Reference rows M3, S1 and S3 are recomputed with the same splits and seeds as prototype v1.
"""
import io, sys, time, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE, label_B
from check_failure_mode import features, p_good, conf_select
from prototype_site_level import unit_losses, crc_index, hb_p, evaluate

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
OUT = ROOT / "results" / "site_level_proto_v2"; OUT.mkdir(parents=True, exist_ok=True)
TAU, ALPHAS, REPEATS, DELTA = 0.25, (0.10, 0.20), 20, 0.10
GATE_Q = (None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def pair_arrays(mu, bad, uidx, nu, med, gates, lams, m):
    """For every (gate, lam) pair: per-unit FDP and certification, and the retained share of rows."""
    P = len(gates) * len(lams)
    F, C, RET = np.zeros((P, nu)), np.zeros((P, nu), bool), np.zeros(P)
    p = 0
    for g in gates:
        gm = med >= g
        for lam in lams:
            keep = gm & (mu >= lam)
            kept = np.bincount(uidx, weights=keep, minlength=nu)
            b = np.bincount(uidx, weights=keep & bad, minlength=nu)
            C[p] = kept >= m
            F[p] = np.where(C[p], b / np.maximum(kept, 1), 0.0)
            RET[p] = (keep & C[p][uidx]).mean()
            p += 1
    return F, C, RET


def selective_ltt(FA, CA, FB, CB, RETB, alpha, delta):
    riskA = np.where(CA, FA, alpha).mean(axis=1)
    riskB = np.where(CB, FB, alpha).mean(axis=1)
    nB, chosen = FB.shape[1], []
    for p in np.argsort(riskA, kind="stable"):
        if riskA[p] >= alpha or hb_p(riskB[p], nB, alpha) > delta:
            break
        chosen.append(p)
    return max(chosen, key=lambda p: RETB[p]) if chosen else None


def certified_keep(mu, med, sidx, ns, g, lam, m):
    keep = (med >= g) & (mu >= lam)
    kept = np.bincount(sidx, weights=keep, minlength=ns)
    return keep & (kept >= m)[sidx]


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
                rng = np.random.default_rng(10007 * r + (t == "node") * 7 + (v == "C") * 3)
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
                gates = [-np.inf if q is None else np.quantile(med_ca[inA], q) for q in GATE_Q]
                lams = np.unique(np.quantile(mu_ca, np.linspace(0.02, 1, 80)))[::-1]
                arrays = {}
                for m in (1, 3):
                    FA, CA, _ = pair_arrays(mu_ca[inA], bad_ca[inA], uA, uA.max() + 1, med_ca[inA], gates, lams, m)
                    FB, CB, RB = pair_arrays(mu_ca[~inA], bad_ca[~inA], uB, uB.max() + 1, med_ca[~inA], gates, lams, m)
                    arrays[m] = (FA, CA, FB, CB, RB)
                base = dict(type=t, v=v, rep=r)
                for a in ALPHAS:
                    rows.append(dict(base, alpha=a, method="M3 pooled conformal selection",
                                     **evaluate(conf_select(mu_ca, bad_ca, mu_U, a), bad_U, s_U, ns)))
                    k1, _ = crc_index(L, a)
                    rows.append(dict(base, alpha=a, method="S1 site-level CRC (abstain = 0)", **evaluate(mu_U >= grid[k1], bad_U, s_U, ns)))
                    best = (-1.0, -np.inf)
                    for g in gates[:6]:
                        gA = med_ca[inA] >= g
                        kA, _ = crc_index(unit_losses(mu_ca[inA], bad_ca[inA], uA, uA.max() + 1, grid, gA), a)
                        retA = ((mu_ca[inA] >= grid[kA]) & gA).mean()
                        if retA > best[0]:
                            best = (retA, g)
                    gB = med_ca[~inA] >= best[1]
                    kB, _ = crc_index(unit_losses(mu_ca[~inA], bad_ca[~inA], uB, uB.max() + 1, grid, gB), a)
                    rows.append(dict(base, alpha=a, method="S3 gate + CRC (abstain = 0)",
                                     **evaluate((mu_U >= grid[kB]) & (med_U >= best[1]), bad_U, s_U, ns)))
                    for m in (1, 3):
                        p = selective_ltt(*arrays[m], a, DELTA)
                        if p is None:
                            keep = np.zeros(len(mu_U), bool); gq, lam = None, None
                        else:
                            g, lam = gates[p // len(lams)], lams[p % len(lams)]; gq = GATE_Q[p // len(lams)]
                            keep = certified_keep(mu_U, med_U, s_U, ns, g, lam, m)
                        rows.append(dict(base, alpha=a, method=f"S4 selective LTT (m={m})", gate_q=gq, **evaluate(keep, bad_U, s_U, ns)))
            out(f"   done {t} {v} at {time.time() - t0:.0f} s")
    res = pd.DataFrame(rows)
    res["sel_exceed"] = res.site_fdp_kept > res.alpha
    res.to_csv(OUT / "per_split.csv", index=False)
    s = res.groupby(["type", "v", "alpha", "method"]).agg(
        retention=("retention", "mean"), site_coverage=("site_coverage", "mean"), pooled_fdp=("fdp", "mean"),
        site_fdp_abstain0=("site_fdp_all", "mean"), site_fdp_where_kept=("site_fdp_kept", "mean"),
        sd_where_kept=("site_fdp_kept", "std"), share_splits_where_kept_gt_alpha=("sel_exceed", "mean"))
    s.to_csv(OUT / "summary.csv")
    out("\nsite_fdp_where_kept = mean over held-out sites that keep data of their bad share; %d repeats, tau %.2f m" % (REPEATS, TAU))
    out(s.round(3).to_string())
    out("\nS4 gate quantile chosen (count of splits):")
    out(res[res.method.str.startswith("S4")].groupby(["type", "v", "alpha", "method"]).gate_q.value_counts(dropna=False).to_string())
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
