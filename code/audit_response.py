"""Audit response (14 Oct 2026): checks requested by the external experimental audit. GPU-bound where models are fitted.
 A leakage matrix: identity overlaps between T / C / U partitions (main design) and across versions (cross-version design)
 B prospective-offset sensitivity: label B with offsets from pre-2025 records only, temporal design (T, C < 2025; U 2025)
 C weight diagnostics: AUC (calibration vs evaluation sites), weight quantiles, ESS, dominance share (real shifts, Brazil reach)
 D Brazil QC flow table, match-distance and time-gap quantiles, final counts per stratum
 E support map relabelled as covariate support (not certification), denominator stated
 G main-result table with mean +- SD over the 20 splits and paired retention differences
Outputs: results/audit_response/*.csv, *.md, support_map_physical.png, report.txt
"""
import io, os, sys, time, warnings
import numpy as np
import pandas as pd
import cupy as cp
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE
from features_v2 import features_v2, twin_map
from prototype_site_level import evaluate
from run_frontier_v5 import mu_good
from run_robustness_v6 import subset, oof, choose_lam, fact
from final_extras import certify

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "audit_response"; OUT.mkdir(parents=True, exist_ok=True)
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def splits(unit, t):
    units = np.unique(unit)
    for q in range(4):
        folds = np.array_split(np.random.default_rng(5000 * q + 17 * (t == "node")).permutation(units), 5)
        for f in range(5):
            rng = np.random.default_rng(100 * q + f)
            U = np.isin(unit, folds[f]); gg = rng.permutation(np.unique(unit[~U])); h = len(gg) // 2
            yield q, f, rng, 1000 * q + 10 * f, (np.nonzero(np.isin(unit, gg[:h]))[0], np.nonzero(np.isin(unit, gg[h:]))[0],
                                                np.nonzero(U)[0])


def section_A(df):
    rows = []
    for t in ("reach", "node"):
        d = subset(df, t, "D")
        keys = {"gauge": d.gauge.to_numpy(), "unit": d.unit.to_numpy(), "feature": d.fid.to_numpy(),
                "SWOT observation (feature, time)": (d.fid.astype(str) + "@" + d.SWOT_time.astype(str)).to_numpy(),
                "overpass (cycle, pass)": (d.cycle_id.astype(str) + "/" + d.pass_id.astype(str)).to_numpy()}
        for q, f, rng, seed, (iT, iC, iU) in splits(d.unit.to_numpy(), t):
            for k, v in keys.items():
                sT, sC, sU = set(v[iT]), set(v[iC]), set(v[iU])
                rows.append(dict(type=t, rep=q, fold=f, identity=k, T_C=len(sT & sC), T_U=len(sT & sU), C_U=len(sC & sU),
                                 n_U=len(sU)))
    r = pd.DataFrame(rows)
    s = r.groupby(["type", "identity"]).agg(max_T_C=("T_C", "max"), max_T_U=("T_U", "max"), max_C_U=("C_U", "max"),
                                            mean_C_U=("C_U", "mean"), mean_n_U=("n_U", "mean"))
    s.to_csv(OUT / "A_leakage_matrix.csv")
    out("== A leakage matrix (max overlap over 20 splits; overpasses are expected to be shared)\n" + s.round(1).to_string())
    cw = pd.read_parquet(ROOT / "data" / "processed" / "cd_crosswalk_v1.parquet")
    dC, dD = subset(df, "reach", "C"), subset(df, "reach", "D")
    gmap = dD.groupby("gauge").unit.first().combine_first(dC.groupby("gauge").unit.first())
    out(f"cross-version design: gauges map to one group each across versions: "
        f"{int(gmap.index.duplicated().sum()) == 0}; crosswalk rows {len(cw)} (used only as a label-free feature)")


def section_B(df, twin):
    rows = []
    for t in ("reach", "node"):
        d = subset(df, t, "D")
        pre = (d.t_swot < pd.Timestamp("2025-01-01", tz="UTC")).to_numpy()
        dd = (d.wse - d.gage_stage_m)
        off_pre = dd.where(pre).groupby([d[c] for c in SITE]).transform("median")
        n_pre = pd.Series(pre.astype(int), index=d.index).groupby([d[c] for c in SITE]).transform("sum")
        keep_sites = (n_pre >= 5).to_numpy()
        d = d[keep_sites].copy(); pre = pre[keep_sites]
        eB_full = d.eB.to_numpy(); eB_pre = (dd[keep_sites] - off_pre[keep_sites]).to_numpy()
        X = features_v2(d, t, twin.loc[d.index]).replace([np.inf, -np.inf], np.nan)
        Xg = cp.asarray(X.to_numpy(np.float32))
        unit, site = d.unit.to_numpy(), d.groupby(SITE).ngroup().to_numpy()
        out(f"B {t}: {site.max() + 1} sites with >= 5 pre-2025 pairs; median |offset_full - offset_pre| = "
            f"{np.nanmedian(np.abs(eB_full - eB_pre)):.3f} m")
        for lab, e in (("offset from full record", eB_full), ("offset from pre-2025 only", eB_pre)):
            bad = np.abs(e) > 0.25; y = cp.asarray(bad.astype(np.int32))
            for q, f, rng, seed, (iT, iC, iU) in splits(unit, t):
                iT, iC, iU = iT[pre[iT]], iC[pre[iC]], iU[~pre[iU]]
                mu_C, mu_U = mu_good(Xg, y, iT, [iC, iU], seed); mu_T = oof(Xg, y, iT, unit[iT], seed, rng)
                sT, sC, sU = fact(site[iT]), fact(site[iC]), fact(site[iU])
                lams = np.unique(np.quantile(mu_T, np.linspace(0.02, 0.98, 30)))
                for a in (0.10, 0.20):
                    lam = choose_lam(mu_T, bad[iT], sT, sT.max() + 1, lams, a, 3)
                    _, keep = certify(mu_C, bad[iC], sC, mu_U, sU, lam, a, "homo", seed)
                    rows.append(dict(type=t, label=lab, alpha=a, rep=q, fold=f, **evaluate(keep, bad[iU], sU, sU.max() + 1)))
    r = pd.DataFrame(rows); r.to_csv(OUT / "B_prospective_offset_per_split.csv", index=False)
    s = r.groupby(["type", "label", "alpha"]).agg(retention=("retention", "mean"), coverage=("site_coverage", "mean"),
                                                  sdr=("site_fdp_kept", "mean"), sd=("site_fdp_kept", "std"))
    s["se"] = s.sd / np.sqrt(20); s.to_csv(OUT / "B_prospective_offset.csv")
    out("\n== B prospective-offset sensitivity (temporal design, S6 homo, tau 0.25)\n" + s.round(4).to_string())


def wdiag(tag, wC, wE):
    a = np.r_[wC, wE]; lab = np.r_[np.zeros(len(wC)), np.ones(len(wE))]
    W = wC.sum(); qC, qE = np.quantile(wC, [.5, .95, .99]), np.quantile(wE, [.5, .95, .99])
    return dict(tag, auc_cal_vs_eval=roc_auc_score(lab, a), ess_C=W ** 2 / (wC ** 2).sum(), n_C=len(wC),
                wC_q50=qC[0], wC_q95=qC[1], wC_q99=qC[2], wC_max=wC.max(), wE_q50=qE[0], wE_q95=qE[1], wE_q99=qE[2],
                wE_max=wE.max(), dominance_gt_0p1=float(np.mean(wE / W > 0.1)), dominance_gt_0p01=float(np.mean(wE / W > 0.01)))


def section_C(df):
    from run_shift_v7 import ratio_weights
    from real_shift_v9 import physical_table
    from real_shift_v8 import site_table, DESIGNS
    from external_test_v1 import physical, site_cov
    from external_test_v2 import prep
    rows = []
    sw, COV = physical_table()
    for t in ("reach", "node"):
        d = subset(df, t, "D"); site, unit = d.groupby(SITE).ngroup().to_numpy(), d.unit.to_numpy()
        st = site_table(d, t, sw); Zs = st[COV].to_numpy(float)
        ut = st.groupby("unit").agg(agency=("agency", "first"), x=("x", "first"), y=("y", "first"))
        for di, (name, (fs, ft)) in enumerate(DESIGNS.items()):
            src_u, tgt_u = ut.index[fs(ut)].to_numpy(), ut.index[ft(ut)].to_numpy()
            for rep in range(5):
                rng = np.random.default_rng(11000 * rep + 101 * di + 7 * (t == "node"))
                su, tu = rng.permutation(src_u), rng.permutation(tgt_u)
                T_u, C_u, F_u, E_u = su[: len(su) // 2], su[len(su) // 2:], tu[: len(tu) // 2], tu[len(tu) // 2:]
                uC, uE = pd.unique(site[np.isin(unit, C_u)]), pd.unique(site[np.isin(unit, E_u)])
                w = ratio_weights(Zs[np.unique(site[np.isin(unit, T_u)])], Zs[np.unique(site[np.isin(unit, F_u)])],
                                  np.vstack([Zs[uC], Zs[uE]]), 100 * rep + 13 * di + 1)
                rows.append(wdiag(dict(experiment="real shift (v9)", type=t, setting=name, rep=rep), w[: len(uC)], w[len(uC):]))
    raw = ROOT / "data" / "raw" / "sword_v17b"
    na_phys, sa_phys = physical(raw / "na_sword_v17b.nc"), physical(raw / "sa_sword_v17b.nc")
    C2 = [c for c in na_phys.columns if c != "reach_id"]
    d = subset(df, "reach", "D"); site, unit = d.groupby(SITE).ngroup().to_numpy(), d.unit.to_numpy()
    Z_na = site_cov(d, na_phys)[C2].to_numpy(float)
    b, masks = prep(); Zb = site_cov(b, sa_phys)[C2].to_numpy(float)
    site_b, unit_b = b.groupby(SITE).ngroup().to_numpy(), b.unit.to_numpy()
    order = ["main", "strict_names", "strict_qc", "hydro_only", "non_hydro", "amazon_basin", "optimistic"]
    for k in ("main", "strict_qc", "non_hydro", "amazon_basin"):
        sel = masks[k]
        for r in range(5):
            rng = np.random.default_rng(31000 + r); seed = 500 + r
            uu = rng.permutation(np.unique(unit)); T_u, C_u = uu[: len(uu) // 2], uu[len(uu) // 2:]
            bu = np.random.default_rng(seed * 100 + order.index(k)).permutation(np.unique(unit_b[sel]))
            F_u, E_u = bu[: len(bu) // 2], bu[len(bu) // 2:]
            uC = pd.unique(site[np.isin(unit, C_u)]); uE = pd.unique(site_b[sel & np.isin(unit_b, E_u)])
            w = ratio_weights(Z_na[np.unique(site[np.isin(unit, T_u)])], Zb[np.unique(site_b[sel & np.isin(unit_b, F_u)])],
                              np.vstack([Z_na[uC], Zb[uE]]), seed)
            rows.append(wdiag(dict(experiment="Brazil reach tau 0.25", type="reach", setting=k, rep=r), w[: len(uC)], w[len(uC):]))
    r = pd.DataFrame(rows); r.to_csv(OUT / "C_weight_diagnostics_per_rep.csv", index=False)
    s = r.groupby(["experiment", "type", "setting"]).mean(numeric_only=True).drop(columns="rep")
    s.to_csv(OUT / "C_weight_diagnostics.csv")
    out("\n== C weight diagnostics (mean over 5 repeats)\n" + s[["auc_cal_vs_eval", "ess_C", "n_C", "wC_q99", "wE_q50", "wE_q99",
                                                              "dominance_gt_0p01", "dominance_gt_0p1"]].round(3).to_string())


def section_D():
    A = ROOT / "results" / "amazon_v1"
    st = pd.read_csv(A / "ana_stations.csv", dtype=str)
    code = pd.to_numeric(st.CodEstacao, errors="coerce")
    lat = pd.to_numeric(st.Latitude.str.replace(",", "."), errors="coerce"); lon = pd.to_numeric(st.Longitude.str.replace(",", "."), errors="coerce")
    fl = (code >= 1e7) & (code <= 99999999) & lat.notna() & lon.notna()
    m = pd.read_csv(A / "matches_qc.csv"); q = pd.read_csv(A / "gauge_qc.csv")
    b = pd.read_parquet(ROOT / "data" / "processed" / "brazil_pairs_v1.parquet"); bb = b[b.tz_mode == "brasilia"]
    flow = [("ANA telemetric stations listed", len(st)), ("fluviometric codes with coordinates", int(fl.sum())),
            ("within 1 km of a SWORD v17b river node", len(m)), ("with >= 10 valid SWOT D passes", int((m.n_swot_D >= 10).sum())),
            ("with >= 100 gauge records 2023-07-26..2025-12-31", int((q.records >= 100).sum())),
            ("passing gauge-only QC", int(q.gauge_qc_pass.sum())),
            ("paired gauges (Brasilia time)", bb.gauge.nunique()), ("paired reaches", bb.fid.nunique()), ("pairs", len(bb))]
    L = ["| Step | Count |", "|---|---|"] + [f"| {a} | {b_} |" for a, b_ in flow]
    dq = np.quantile(m.dist_km, [.5, .95, 1.0]); tq = np.quantile(bb.dt_s / 60, [.5, .9, .95, 1.0])
    L.append(f"\nMatch distance km: median {dq[0]:.3f}, p95 {dq[1]:.3f}, max {dq[2]:.3f}. "
             f"Pairing gap min: median {tq[0]:.1f}, p90 {tq[1]:.1f}, p95 {tq[2]:.1f}, max {tq[3]:.1f}.")
    from external_test_v2 import prep
    pb, masks = prep()
    L += ["\n| Stratum (after dam/reservoir, tidal, >= 5 pairs) | Gauges | Sites | Pairs | Bad rate (tau 0.25) |", "|---|---|---|---|---|"]
    for k, s in masks.items():
        x = pb[s]
        L.append(f"| {k} | {x.gauge.nunique()} | {x.groupby(SITE).ngroup().max() + 1 if len(x) else 0} | {len(x)} | {np.mean(np.abs(x.eB) > 0.25):.3f} |")
    txt = "\n".join(L); (OUT / "D_brazil_qc_flow.md").write_text(txt, encoding="utf-8"); out("\n== D Brazil QC flow\n" + txt)


def section_E():
    u = pd.read_csv(ROOT / "results" / "sword_domain_v2" / "reach_support_physical.csv")
    lab = {"likely certifiable": "within estimated covariate support (ratio <= 0.01)",
           "partial": "weak covariate support (0.01 < ratio <= 0.1)", "abstention likely": "outside covariate support (ratio > 0.1)"}
    u["support"] = u.cls.map(lab); u.to_csv(OUT / "E_support_classes.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7))
    for k, c in zip(lab.values(), ("#1b9e77", "#e6ab02", "#d95f02")):
        s = u[u.support == k]; ax.scatter(s.x, s.y, s=0.3, c=c, label=f"{k}: {len(s)} ({len(s) / len(u):.1%})")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend(loc="lower left", markerscale=8, fontsize=8)
    ax.set_title(f"Physical-covariate support of {len(u)} ungauged North American river reaches relative to gauged reaches\n"
                 "(diagnostic of feature support; not a certification of water-level accuracy)", fontsize=10)
    fig.savefig(OUT / "support_map_physical.png", dpi=150, bbox_inches="tight")
    out(f"\n== E support classes (denominator: {len(u)} ungauged SWORD v17b river reaches, type 1, lakeflag 0)\n"
        + u.support.value_counts().to_string())


def section_G():
    r = pd.read_csv(ROOT / "results" / "final" / "main_per_split.csv")
    r = r[(r.v == "D") & (r.tau == 0.25) & (r.alpha.isin([0.10, 0.20]))]
    meths = ["S6 SCoRE-SDR (homo)", "S1 site CRC (abstain = 0)", "M3 pooled conformal", "M1 USGS-style grid (in-sample)"]
    r = r[r.method.isin(meths)]
    g = r.groupby(["type", "alpha", "method"]).agg(ret=("retention", "mean"), ret_sd=("retention", "std"),
         cov=("site_coverage", "mean"), sdr=("site_fdp_kept", "mean"), sdr_sd=("site_fdp_kept", "std"), n=("retention", "size"))
    p = r.pivot_table(index=["type", "alpha", "rep", "fold"], columns="method", values="retention")
    L = ["| Type | alpha | Method | Retained obs (mean +- SD) | Site coverage | Realised SDR (mean +- SD) |", "|---|---|---|---|---|---|"]
    for (t, a, m), x in g.iterrows():
        L.append(f"| {t} | {a:.2f} | {m} | {x.ret:.3f} +- {x.ret_sd:.3f} | {x['cov']:.3f} | {x.sdr:.3f} +- {x.sdr_sd:.3f} |")
    L += ["\n| Type | alpha | Paired retention difference (per split, n = 20) | mean +- SD |", "|---|---|---|---|"]
    for (t, a), x in p.groupby(level=[0, 1]):
        for other in meths[1:]:
            dlt = x[meths[0]] - x[other]
            L.append(f"| {t} | {a:.2f} | S6 - {other} | {dlt.mean():.3f} +- {dlt.std():.3f} |")
    L.append("\nSD is across 20 splits (4 repeats x 5 folds); splits share units across repeats, so SDs understate "
             "fully independent replication. Paired differences use identical splits.")
    txt = "\n".join(L); (OUT / "G_main_table.md").write_text(txt, encoding="utf-8"); out("\n== G main table\n" + txt)


def main():
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
    twin = twin_map(df, pd.read_parquet(ROOT / "data" / "processed" / "cd_crosswalk_v1.parquet"))
    for name, fn in (("A", lambda: section_A(df)), ("D", section_D), ("E", section_E), ("G", section_G),
                     ("C", lambda: section_C(df)), ("B", lambda: section_B(df, twin))):
        try:
            fn()
        except Exception as e:
            out(f"SECTION {name} FAILED: {e.__class__.__name__}: {e}")
        out(f"[section {name} done at {time.time() - t0:.0f} s]")
        (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
