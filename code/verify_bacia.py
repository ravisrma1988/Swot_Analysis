"""Verify ANA basin codes: most frequent rivers and median location per Bacia code among matched stations."""
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT / "results" / "amazon_v1" / "matches_qc.csv")
m = m[m.n_swot_D >= 10]
for code, g in m.groupby(pd.to_numeric(m.Bacia, errors="coerce")):
    rivers = g.ana_name_fixed.value_counts().head(8).to_dict()
    print(f"Bacia {code:.0f}: {len(g)} stations, median lat {g.lat.median():.1f}, lon {g.lon.median():.1f}; top rivers {rivers}")
