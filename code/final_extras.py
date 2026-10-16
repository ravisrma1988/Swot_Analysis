"""Final extras (14 Oct 2026), GPU-bound, tuned model, same splits/seeds as final_main.py. Version D, tau 0.25.
 E1 named community filters (reach): Confluence 'Permissive Relaxed' approximation (prior width >= 60 m, length >= 5 km,
    dark fraction <= 0.6, valid-node fraction >= 0.4, Tukey outliers on the reach's WSE series) and a composite strict
    practitioner filter (reach_q <= 1, xovr_cal_q <= 1, dark <= 0.5, Hampel flag 0, cross-track 10-60 km). Label-free.
 E2 temporal holdout: T and C use observations before 2025-01-01 at gauged units; U uses 2025 observations at held-out
    units (2025 also contains the switch from reprocessed PGD0 to forward PID0 products).
 E3 conditional validity of S6 homo (alpha 0.10, 0.20) by width class, latitude band, agency, site flag regime, and,
    at observation level, gauge-flow percentile within site (evaluation only); also S6 'hete' pruning on all splits.
 E4 seed stability of S6 homo: 50 seeds on the 5 folds of repeat 0; per-reach certification frequency.
 E5 runtime: reference SCoRE_SDR vs GPU port on a realistic split; XGBoost fit time.
Outputs: results/final/extras_*.csv and extras_report.txt
"""
import io, sys, time, warnings
import numpy as np
import pandas as pd
import cupy as cp
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE
from features_v2 import features_v2, twin_map
from prototype_site_level import evaluate
from prototype_score_sdr import site_scores
from check_failure_mode import conf_select
from score_sdr_gpu import score_sdr_gpu, SCoRE_SDR_ref
from run_frontier_v5 import mu_good, XGB
from run_robustness_v6 import subset, oof, choose_lam, fact

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "final"
TAU, ALPHAS, REPEATS, FOLDS, M = 0.25, (0.10, 0.20), 4, 5, 3
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def certify(mu_C, bad_C, sC, mu_U, sU, lam, a, prune, seed):
    s_C, L_C, _ = site_scores(mu_C, bad_C, sC, sC.max() + 1, lam, M)
    s_U, _, _ = site_scores(mu_U, None, sU, sU.max() + 1, lam, M)
    eC, eU = np.isfinite(s_C), np.isfinite(s_U)
    cert = np.zeros(sU.max() + 1, bool)
    if eC.any() and eU.any():
        sel = score_sdr_gpu(L_C[eC], s_C[eC], s_U[eU], a, a, prune=prune, random_state=seed)
        cert[np.nonzero(eU)[0][sel]] = True
    return cert, (mu_U >= lam) & cert[sU]


def named_filters(d, X):
    w = d.wse
    g = w.groupby([d[c] for c in SITE])
    q1, q3 = g.transform(lambda x: x.quantile(0.25)), g.transform(lambda x: x.quantile(0.75))
    tukey = (w >= q1 - 1.5 * (q3 - q1)) & (w <= q3 + 1.5 * (q3 - q1))
    confluence = ((d.p_width >= 60) & (d.p_length >= 5000) & (d.dark_frac <= 0.6)
                  & (d.n_good_nod / d.p_n_nodes >= 0.4) & tukey)
    ax = d.xtrk_dist.abs()
    strict = ((d.qflag <= 1) & (d.xovr_cal_q <= 1) & (d.dark_frac <= 0.5) & (X.hampel_flag == 0)
              & (ax >= 10000) & (ax <= 60000))
    return {"Confluence Permissive Relaxed (approx.)": confluence.to_numpy(bool),
            "Composite strict practitioner filter": strict.to_numpy(bool)}


def site_fdp(keep, bad, sU):
    ns = sU.max() + 1
    kept = np.bincount(sU, weights=keep, minlength=ns)
    b = np.bincount(sU, weights=keep & bad, minlength=ns)
    return kept, np.where(kept > 0, b / np.maximum(kept, 1), np.nan)


def group_rows(tag, keep, bad, sU, site_groups, flow_cls):
    """Site-level realised SDR by site attribute group, and observation-level bad share by flow class."""
    rows = []
    kept, f = site_fdp(keep, bad, sU)
    for attr, g in site_groups.items():
        for cls in pd.unique(g):
            m = g == cls
            cert = m & (kept > 0)
            rows.append(dict(tag, level="site", attribute=attr, group=str(cls), n_sites=int(m.sum()),
                             coverage=float(cert.sum() / max(m.sum(), 1)), sdr=float(np.nanmean(f[cert])) if cert.any() else np.nan))
    for cls in pd.unique(flow_cls):
        m = flow_cls == cls
        rows.append(dict(tag, level="obs", attribute="flow_percentile", group=str(cls), n_obs=int(m.sum()),
                         retention=float(keep[m].mean()) if m.any() else np.nan,
                         bad_share_kept=float(bad[m & keep].mean()) if (m & keep).any() else np.nan))
    return rows


def seed_stability(tag, mu_C, bad_C, sC, mu_U, bad_U, sU, lam, a, n_seeds=50):
    s_U, _, _ = site_scores(mu_U, None, sU, sU.max() + 1, lam, M)
    elig = np.isfinite(s_U)
    certs, rets, sdrs = [], [], []
    for s in range(n_seeds):
        cert, keep = certify(mu_C, bad_C, sC, mu_U, sU, lam, a, "homo", s)
        certs.append(cert); ev = evaluate(keep, bad_U, sU, sU.max() + 1)
        rets.append(ev["retention"]); sdrs.append(ev["site_fdp_kept"])
    C = np.array(certs)[:, elig]
    freq = C.mean(axis=0)
    rng = np.random.default_rng(0); jac = []
    for _ in range(200):
        i, j = rng.choice(n_seeds, 2, replace=False)
        u = (C[i] | C[j]).sum(); jac.append((C[i] & C[j]).sum() / u if u else 1.0)
    return dict(tag, alpha=a, eligible_sites=int(elig.sum()), always=float((freq == 1).mean()), never=float((freq == 0).mean()),
                sometimes=float(((freq > 0) & (freq < 1)).mean()), mean_jaccard=float(np.mean(jac)),
                retention_mean=float(np.mean(rets)), retention_sd=float(np.std(rets)),
                sdr_mean=float(np.mean(sdrs)), sdr_sd=float(np.std(sdrs)))


def main():
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
    twin = twin_map(df, pd.read_parquet(ROOT / "data" / "processed" / "cd_crosswalk_v1.parquet"))
    R1, R2, R3, R4, R5 = [], [], [], [], []
    for t in ("reach", "node"):
        d = subset(df, t, "D")
        X = features_v2(d, t, twin.loc[d.index]).replace([np.inf, -np.inf], np.nan)
        Xg = cp.asarray(X.to_numpy(np.float32))
        bad = (d.eB.abs() > TAU).to_numpy(); y = cp.asarray(bad.astype(np.int32))
        unit, site = d.unit.to_numpy(), d.groupby(SITE).ngroup().to_numpy(); units = np.unique(unit)
        sa = d.groupby(SITE).agg(w=("p_width", "first"), lat=("gage_latitude", "first"), ag=("gage_agency", "first"),
                                 q=("qflag", "median")).reset_index(drop=True)
        width_c = pd.cut(sa.w, [0, 100, 300, 1000, np.inf], labels=["<100 m", "100-300 m", "300-1000 m", ">1000 m"]).astype(str).to_numpy()
        lat_c = pd.cut(sa.lat, [-np.inf, 35, 45, 55, np.inf], labels=["<35N", "35-45N", "45-55N", ">=55N"]).astype(str).to_numpy()
        ag_c = np.where(sa.ag == "USGS", "US (USGS)", "Canada (ECCC/Quebec)")
        q_c = np.where(sa.q <= 1, "site median flag <= 1", "site median flag >= 2")
        fp = d.groupby(SITE).gage_flow_m3s.rank(pct=True).to_numpy()
        flow_c = np.select([np.isnan(fp), fp >= 0.9, fp <= 0.1], ["no flow data", "high flow (>= p90)", "low flow (<= p10)"], "mid flow")
        filters = named_filters(d, X) if t == "reach" else {}
        pre = (d.t_swot < pd.Timestamp("2025-01-01", tz="UTC")).to_numpy()
        out(f"{t}: {len(d)} pairs; 2025 share {1 - pre.mean():.3f}; flow data share {np.mean(~np.isnan(fp)):.3f}")
        for q in range(REPEATS):
            folds = np.array_split(np.random.default_rng(5000 * q + 17 * (t == "node")).permutation(units), FOLDS)
            for f in range(FOLDS):
                rng = np.random.default_rng(100 * q + f); seed = 1000 * q + 10 * f
                U = np.isin(unit, folds[f]); gg = rng.permutation(np.unique(unit[~U])); half = len(gg) // 2
                iT, iC, iU = np.nonzero(np.isin(unit, gg[:half]))[0], np.nonzero(np.isin(unit, gg[half:]))[0], np.nonzero(U)[0]
                tag = dict(type=t, rep=q, fold=f)
                t1 = time.time(); mu_C, mu_U = mu_good(Xg, y, iT, [iC, iU], seed); fit_s = time.time() - t1
                mu_T = oof(Xg, y, iT, unit[iT], seed, rng)
                sT, sC, sU = fact(site[iT]), fact(site[iC]), fact(site[iU])
                lams = np.unique(np.quantile(mu_T, np.linspace(0.02, 0.98, 30)))
                uU = pd.unique(site[iU])
                groups = {"width": width_c[uU], "latitude": lat_c[uU], "agency": ag_c[uU], "flag_regime": q_c[uU]}
                for name, fk in filters.items():
                    R1.append(dict(tag, method=name, **evaluate(fk[iU], bad[iU], sU, sU.max() + 1)))
                for a in ALPHAS:
                    lam = choose_lam(mu_T, bad[iT], sT, sT.max() + 1, lams, a, M)
                    for prune in ("homo", "hete"):
                        cert, keep = certify(mu_C, bad[iC], sC, mu_U, sU, lam, a, prune, seed)
                        R3.append(dict(tag, alpha=a, method=f"S6 ({prune})", level="all", **evaluate(keep, bad[iU], sU, sU.max() + 1)))
                        if prune == "homo":
                            R3 += group_rows(dict(tag, alpha=a, method="S6 (homo)"), keep, bad[iU], sU, groups, flow_c[iU])
                    if q == 0:
                        R4.append(seed_stability(tag, mu_C, bad[iC], sC, mu_U, bad[iU], sU, lam, a))
                    if q == 0 and f == 0 and a == 0.20:
                        s_C, L_C, _ = site_scores(mu_C, bad[iC], sC, sC.max() + 1, lam, M)
                        s_U, _, _ = site_scores(mu_U, None, sU, sU.max() + 1, lam, M)
                        eC, eU = np.isfinite(s_C), np.isfinite(s_U)
                        t1 = time.time(); SCoRE_SDR_ref((L_C[eC], s_C[eC]), s_U[eU], a, a); ref_s = time.time() - t1
                        t1 = time.time(); score_sdr_gpu(L_C[eC], s_C[eC], s_U[eU], a, a); gpu_s = time.time() - t1
                        R5.append(dict(type=t, n_cal_sites=int(eC.sum()), n_test_sites=int(eU.sum()), scored_ref_s=ref_s,
                                       scored_gpu_s=gpu_s, speedup=ref_s / max(gpu_s, 1e-9), xgb_fit_plus_predict_s=fit_s))
                # E2 temporal holdout
                iT2, iC2, iU2 = iT[pre[iT]], iC[pre[iC]], iU[~pre[iU]]
                if len(iU2) > 0:
                    mu_C2, mu_U2 = mu_good(Xg, y, iT2, [iC2, iU2], seed); mu_T2 = oof(Xg, y, iT2, unit[iT2], seed, rng)
                    sT2, sC2, sU2 = fact(site[iT2]), fact(site[iC2]), fact(site[iU2])
                    lams2 = np.unique(np.quantile(mu_T2, np.linspace(0.02, 0.98, 30)))
                    for a in ALPHAS:
                        lam2 = choose_lam(mu_T2, bad[iT2], sT2, sT2.max() + 1, lams2, a, M)
                        for prune in ("homo", None):
                            _, keep2 = certify(mu_C2, bad[iC2], sC2, mu_U2, sU2, lam2, a, prune, seed)
                            R2.append(dict(tag, alpha=a, method=f"S6 ({prune})", **evaluate(keep2, bad[iU2], sU2, sU2.max() + 1)))
                        R2.append(dict(tag, alpha=a, method="M3 pooled", **evaluate(conf_select(mu_C2, bad[iC2], mu_U2, a), bad[iU2], sU2, sU2.max() + 1)))
            out(f"   {t} repeat {q + 1}/{REPEATS} at {time.time() - t0:.0f} s")
    for name, R in (("filters", R1), ("temporal", R2), ("conditional", R3), ("seeds", R4), ("runtime", R5)):
        pd.DataFrame(R).to_csv(OUT / f"extras_{name}.csv", index=False)
    summarise_all()
    out(f"done in {time.time() - t0:.0f} s")
    (OUT / "extras_report.txt").write_text(buf.getvalue(), encoding="utf-8")


def summarise_all():
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    L = []
    f1 = pd.read_csv(OUT / "extras_filters.csv")
    L.append("== E1 named filters (Version D reach, tau 0.25; site_fdp_kept = realised SDR)")
    L.append(f1.groupby(["type", "method"])[["retention", "site_coverage", "fdp", "site_fdp_kept"]].mean().round(3).to_string())
    f2 = pd.read_csv(OUT / "extras_temporal.csv")
    g2 = f2.groupby(["type", "alpha", "method"]).agg(retention=("retention", "mean"), coverage=("site_coverage", "mean"),
                                                      sdr=("site_fdp_kept", "mean"), sd=("site_fdp_kept", "std"), n=("site_fdp_kept", "size"))
    g2["excess_over_2se"] = g2.sdr - g2.index.get_level_values("alpha") - 2 * g2.sd / np.sqrt(g2.n)
    L.append("\n== E2 temporal holdout (calibrate < 2025, test 2025 at held-out units)\n" + g2.round(3).to_string())
    f3 = pd.read_csv(OUT / "extras_conditional.csv")
    a3 = f3[f3.level == "all"].groupby(["type", "alpha", "method"]).agg(retention=("retention", "mean"),
          coverage=("site_coverage", "mean"), sdr=("site_fdp_kept", "mean"), sd=("site_fdp_kept", "std"))
    L.append("\n== E3a S6 homo vs hete (all splits)\n" + a3.round(3).to_string())
    s3 = f3[f3.level == "site"].groupby(["type", "alpha", "attribute", "group"]).agg(
          n_sites=("n_sites", "mean"), coverage=("coverage", "mean"), sdr=("sdr", "mean"), sdr_sd=("sdr", "std"))
    L.append("\n== E3b site-level realised SDR by group (S6 homo)\n" + s3.round(3).to_string())
    o3 = f3[f3.level == "obs"].groupby(["type", "alpha", "group"]).agg(n_obs=("n_obs", "mean"), retention=("retention", "mean"),
          bad_share_kept=("bad_share_kept", "mean"))
    L.append("\n== E3c observation-level bad share among kept, by gauge-flow percentile within site\n" + o3.round(3).to_string())
    f4 = pd.read_csv(OUT / "extras_seeds.csv")
    L.append("\n== E4 seed stability (50 seeds, 5 folds of repeat 0; shares of eligible reaches)\n"
             + f4.groupby(["type", "alpha"])[["eligible_sites", "always", "never", "sometimes", "mean_jaccard", "retention_mean",
                                               "retention_sd", "sdr_mean", "sdr_sd"]].mean().round(3).to_string())
    L.append("\n== E5 runtime\n" + pd.read_csv(OUT / "extras_runtime.csv").round(2).to_string(index=False))
    (OUT / "extras_summary.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
