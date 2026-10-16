"""Inspect SWORD v17b reach variables needed for physical (non-geographic) weight covariates."""
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "pylib"))
import h5py

with h5py.File(ROOT / "data" / "raw" / "sword_v17b" / "na_sword_v17b.nc", "r") as f:
    g = f["reaches"]
    for v in ["iceflag", "julian_day", "stream_order", "dist_out", "max_width", "path_freq", "trib_flag", "low_slope_flag"]:
        d = g[v]
        a = np.asarray(d[...])
        attrs = {k: (d.attrs[k].decode() if isinstance(d.attrs[k], bytes) else d.attrs[k]) for k in d.attrs.keys()
                 if k not in ("DIMENSION_LIST", "REFERENCE_LIST")}
        vals = np.unique(a[np.isfinite(a)] if a.dtype.kind == "f" else a)
        print(f"{v}: shape {a.shape} dtype {a.dtype}; unique (first 10) {vals[:10]}; attrs {attrs}")
