#!/bin/sh
# Verified range download of SWORD v17b (netCDF), then the support analysis; each step runs only if the previous passed.
cd /workspace/ravi/swot || exit 1
rm -f data/raw/sword_v17b/SWORD_v17b_netcdf.zip data/raw/sword_v17b/expected.md5
python3 code/download_range.py 15299138 SWORD_v17b_netcdf.zip data/raw/sword_v17b 6 || { echo DOWNLOAD_FAILED; exit 1; }
python3 code/sword_support.py || { echo SUPPORT_FAILED; exit 1; }
echo PIPELINE_DONE
