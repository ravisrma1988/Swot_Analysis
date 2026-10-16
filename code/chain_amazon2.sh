#!/bin/sh
# External-set chain (13 Oct 2026): match QC -> Amazon download -> time-zone check 2 -> rest-of-Brazil download.
# Each step logs separately; the downloads are resumable (re-running skips finished stations).
cd /workspace/ravi/swot || exit 1
python3 code/amazon_match_qc.py > logs/amazon_match_qc.log 2>&1 || { echo QC_FAILED; exit 1; }
echo "QC done $(date -u '+%H:%M UTC')"
python3 code/amazon_download.py amazon > logs/amazon_download_amazon.log 2>&1 || echo DOWNLOAD_AMAZON_FAILED
echo "Amazon download done $(date -u '+%H:%M UTC')"
python3 code/amazon_tz_check2.py > logs/amazon_tz_check2.log 2>&1 || echo TZ_CHECK2_FAILED
echo "TZ check 2 done $(date -u '+%H:%M UTC')"
python3 code/amazon_download.py brazil > logs/amazon_download_brazil.log 2>&1 || echo DOWNLOAD_BRAZIL_FAILED
echo "Brazil download done $(date -u '+%H:%M UTC')"
echo CHAIN_DONE
