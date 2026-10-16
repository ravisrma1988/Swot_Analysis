#!/bin/sh
# External-set chain v3 (13 Oct 2026): SWOT reach series for all Brazilian matches -> full-record ANA download
# (yearly requests, Amazon basin first) -> time-zone check 2. Each step logs separately and is resumable.
cd /workspace/ravi/swot || exit 1
python3 code/swot_reach_download.py > logs/swot_reach_download.log 2>&1 || echo SWOT_DOWNLOAD_FAILED
echo "SWOT reach download done $(date -u '+%H:%M UTC')"
python3 code/amazon_download_full.py > logs/amazon_download_full.log 2>&1 || echo ANA_FULL_FAILED
echo "ANA full download done $(date -u '+%H:%M UTC')"
python3 code/amazon_tz_check2.py > logs/amazon_tz_check2.log 2>&1 || echo TZ_CHECK2_FAILED
echo "TZ check 2 done $(date -u '+%H:%M UTC')"
echo CHAIN_DONE
