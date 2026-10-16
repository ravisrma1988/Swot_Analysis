"""Read-only checks for the external pre-submission audit. Nothing frozen is modified."""
import pathlib, subprocess, sys
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; R = ROOT / "results"; D = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "external" / "pylib"))
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()
from swotio import SITE                 # same helpers as the frozen pipeline and fig1 script
from run_robustness_v6 import subset
df = pd.read_parquet(D / "paired_clean_v1.parquet")
print("SITE =", SITE, "| has unit column:", "unit" in df.columns)
print("== A2/C1 river units versus sites (Version D analysis sets)")
for ty in ("reach", "node"):
    d = subset(df, ty, "D").copy()
    if "unit" not in d.columns:
        d["unit"] = d.fid // (10 if ty == "reach" else 10000)
    s = d.groupby(SITE).agg(unit=("unit", "first")).reset_index()
    per = s.groupby("unit").size()
    multi = per[per > 1]
    print(f"  {ty}: sites {len(s)}, units {len(per)}, units with >1 site {len(multi)} ({len(multi)/len(per):.1%}),"
          f" sites in such units {int(multi.sum())} ({multi.sum()/len(s):.1%}), max sites per unit {int(per.max())}")
print("== C2 out-of-fold folds and gamma")
print(sh("grep -n -i 'oof\\|GroupKFold\\|n_splits\\|n_folds' code/final_main.py code/run_robustness_v6.py code/prototype_score_sdr.py 2>/dev/null | head -10"))
print(sh("grep -rn -i 'gamma' code/*.py | grep -v '^code/external/' | head -8"))
print("== C2 version-difference feature")
print(sh("grep -rn -i 'twin' code/swotio.py code/features*.py 2>/dev/null | head -6"))
print("== C5 compute context")
print(sh("lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread\\(s\\) per core'"))
print(sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1"))
print(sh("python3 -c 'import torch, cupy, xgboost, numpy; print(\"torch\", torch.__version__, \"cupy\", cupy.__version__, \"xgboost\", xgboost.__version__, \"numpy\", numpy.__version__)' 2>&1 | tail -1"))
print("== C6 support-class sensitivity to thresholds")
u = pd.read_csv(R / "audit_response" / "E_support_classes.csv"); print("  columns:", list(u.columns))
rc = [c for c in u.columns if "ratio" in c.lower() or c.lower() in ("rho",)]
if rc:
    r = u[rc[0]].to_numpy(dtype=float)
    print(f"  ratio quantiles q50 {np.nanquantile(r, .5):.4f} q90 {np.nanquantile(r, .9):.4f} q99 {np.nanquantile(r, .99):.4f}")
    for lo, hi in ((0.005, 0.05), (0.01, 0.1), (0.02, 0.2)):
        print(f"  thresholds {lo}/{hi}: within {np.mean(r <= lo):.3f} weak {np.mean((r > lo) & (r <= hi)):.3f} outside {np.mean(r > hi):.3f}")
