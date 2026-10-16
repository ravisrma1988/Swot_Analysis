"""Hydrocron SWOT D reach series for every matched Brazilian reach (skips reaches already saved)."""
import concurrent.futures as cf, pathlib, sys
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from amazon_download import get_reach, FULL, log

ROOT = pathlib.Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT / "results" / "amazon_v1" / "matches_qc.csv")
rids = m[m.n_swot_D >= 10].reach_id.unique()
log(f"SWOT reach series for {len(rids)} reaches (existing files skipped)")
with cf.ThreadPoolExecutor(4) as ex:
    ok = list(ex.map(lambda r: get_reach(r, FULL), rids))
log(f"done: {sum(ok)}/{len(rids)} reaches available")
