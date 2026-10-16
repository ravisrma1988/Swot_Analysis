"""Check readers needed for SWORD netCDF and for maps."""
import importlib.util
for m in ["netCDF4", "h5py", "xarray", "h5netcdf", "matplotlib", "geopandas", "shapely"]:
    spec = importlib.util.find_spec(m)
    if spec is None:
        print(f"{m}: MISSING")
    else:
        mod = __import__(m)
        print(f"{m}: {getattr(mod, '__version__', 'ok')}")
