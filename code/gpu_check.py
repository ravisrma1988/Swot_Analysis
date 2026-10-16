"""Report GPUs and GPU-capable Python libraries on the DGX."""
import importlib.util, subprocess

print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                      "--format=csv"], capture_output=True, text=True).stdout)
for m in ["torch", "xgboost", "cupy", "cudf", "cuml", "lightgbm", "catboost"]:
    if importlib.util.find_spec(m) is None:
        print(f"{m}: MISSING")
        continue
    try:
        mod = importlib.import_module(m)
        print(f"{m}: {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"{m}: import error {e.__class__.__name__}: {e}")
try:
    import torch
    print("torch cuda:", torch.cuda.is_available(), torch.cuda.device_count())
except Exception as e:
    print("torch check failed:", e)
