#!/bin/sh
# Overnight chain (13-14 Oct 2026): waits for chain_amazon3 (ANA download + time-zone check 2), then
# basin-code check -> full Brazil pairing (both time readings) -> external test North America -> Brazil (GPU).
cd /workspace/ravi/swot || exit 1
while pgrep -f chain_amazon3.sh >/dev/null; do sleep 60; done
echo "download chain finished $(date -u '+%H:%M UTC')"
python3 code/verify_bacia.py > logs/verify_bacia.log 2>&1 || echo VERIFY_BACIA_FAILED
python3 code/pair_brazil.py > logs/pair_brazil.log 2>&1 || { echo PAIR_FAILED; exit 1; }
echo "pairing done $(date -u '+%H:%M UTC')"
python3 code/external_test_v1.py > logs/external_test_v1.log 2>&1 || echo EXTERNAL_TEST_FAILED
echo "external test done $(date -u '+%H:%M UTC')"
echo OVERNIGHT_DONE
