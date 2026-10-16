"""Extended label-free features (v2) for the screening score. Everything is computed from SWOT data only
(the site's own SWOT series, its prior-database attributes, and the other product version of the same overpass),
so it is available at ungauged reaches in a retrospective screening setting. No gauge data enter any feature.
"""
import numpy as np
import pandas as pd
from swotio import SITE
from check_failure_mode import features as features_base


def _site(d, col, how):
    return d.groupby(SITE)[col].transform(how)


def features_v2(d, t, twin_dwse):
    """d: one type x version subset of the cleaned table; twin_dwse: Series indexed like d (NaN where no twin)."""
    X = features_base(d, t)
    s = d.sort_values(SITE + ["t_swot"])
    g = s.groupby(SITE)
    w = s.wse
    med = g.wse.transform("median")
    mad = (w - med).abs().groupby([s[c] for c in SITE]).transform("median")
    X["robust_z"] = ((w - med).abs() / (1.4826 * mad + 1e-3)).reindex(d.index)
    roll = g.wse.transform(lambda x: x.rolling(5, center=True, min_periods=3).median())
    X["dev_med5"] = (w - roll).abs().reindex(d.index)
    tdays = s.t_swot.astype("int64") / 86400e9
    X["jump_prev"] = (w - g.wse.shift(1)).abs().reindex(d.index)
    X["jump_next"] = (w - g.wse.shift(-1)).abs().reindex(d.index)
    X["dt_prev"] = (tdays - tdays.groupby([s[c] for c in SITE]).shift(1)).reindex(d.index)
    X["dt_next"] = (tdays.groupby([s[c] for c in SITE]).shift(-1) - tdays).reindex(d.index)
    wid = d.width.astype(float)
    mw, mh = _site(d.assign(_w=wid), "_w", "mean"), _site(d, "wse", "mean")
    cov = _site(d.assign(_c=(wid - mw) * (d.wse - mh)), "_c", "mean")
    var = _site(d.assign(_v=(wid - mw) ** 2), "_v", "mean")
    resid = d.wse - (mh + (cov / var.replace(0, np.nan)) * (wid - mw))
    rmad = resid.abs().groupby([d[c] for c in SITE]).transform("median")
    X["width_wse_resid"] = (resid.abs() / (rmad + 1e-3)).astype(float)
    X["site_med_wse_u"] = _site(d, "wse_u", "median")
    X["site_frac_q2plus"] = _site(d.assign(_q=(d.qflag >= 2).astype(float)), "_q", "mean")
    X["site_mad_wse"] = mad.reindex(d.index)
    X["site_nobs"] = _site(d, "wse", "size").astype(float)
    X["site_med_dark"] = _site(d, "dark_frac", "median")
    X["twin_dwse"] = twin_dwse.reindex(d.index).astype(float)
    X["has_twin"] = X.twin_dwse.notna().astype(float)
    return X


def twin_map(df, crosswalk):
    """|wse_D - wse_C| for the same overpass (gauge, cycle, pass, |dt| <= 60 s), keyed by row_id of either version."""
    cw = crosswalk.drop_duplicates("row_id_C").drop_duplicates("row_id_D")
    dw = (cw.wse_D - cw.wse_C).abs()
    m = pd.concat([pd.Series(dw.to_numpy(), index=cw.row_id_C.to_numpy()),
                   pd.Series(dw.to_numpy(), index=cw.row_id_D.to_numpy())])
    m = m[~m.index.duplicated()]
    return pd.Series(df.row_id.map(m).to_numpy(), index=df.index)


def features_causal(d, t):
    """Near-real-time variant: every feature uses only the current pass and earlier passes of the same site."""
    X = features_base(d, t).drop(columns=["hampel_r", "hampel_flag"])
    s = d.sort_values(SITE + ["t_swot"])
    keys = [s[c] for c in SITE]
    prev = s.groupby(SITE).wse.shift(1)
    roll = prev.groupby(keys).rolling(10, min_periods=3)
    past_med = roll.median().reset_index(level=list(range(len(SITE))), drop=True)
    past_mad = roll.apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True).reset_index(
        level=list(range(len(SITE))), drop=True)
    dev = (s.wse - past_med).abs()
    X["dev_past_med10"] = dev.reindex(d.index)
    X["robust_z_past"] = (dev / (1.4826 * past_mad + 1e-3)).reindex(d.index)
    tdays = s.t_swot.astype("int64") / 86400e9
    X["jump_prev"] = (s.wse - prev).abs().reindex(d.index)
    X["dt_prev"] = (tdays - tdays.groupby(keys).shift(1)).reindex(d.index)
    q2 = (s.qflag >= 2).astype(float)
    X["past_frac_q2plus"] = q2.groupby(keys).transform(lambda x: x.shift(1).expanding().mean()).reindex(d.index)
    X["past_med_wse_u"] = s.wse_u.groupby(keys).transform(lambda x: x.shift(1).expanding().median()).reindex(d.index)
    X["n_past"] = s.groupby(SITE).cumcount().astype(float).reindex(d.index)
    return X
