"""Table I counts: North American analysis sets (science phase, non-tidal, >= 5 pairs per site), tau 0.25 m."""
import pathlib, sys
import numpy as np
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from swotio import SITE
from run_robustness_v6 import subset

df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
print("cleaned rows", len(df), "| gauges", df.gauge.nunique())
for v in ("D", "C"):
    for t in ("reach", "node"):
        d = subset(df, t, v)
        print(f"{t} {v}: pairs {len(d)}, sites {d.groupby(SITE).ngroup().max() + 1}, gauges {d.gauge.nunique()}, "
              f"features {d.fid.nunique()}, bad(0.25) {np.mean(np.abs(d.eB) > 0.25):.3f}, "
              f"dates {d.t_swot.min():%Y-%m-%d} to {d.t_swot.max():%Y-%m-%d}")
