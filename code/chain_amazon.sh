#!/bin/sh
# Wait for amazon_step1 (station matching + Hydrocron counts) to finish, then run the ANA time-zone check.
cd /workspace/ravi/swot || exit 1
while pgrep -f 'code/amazon_step1.py' >/dev/null; do sleep 30; done
if [ ! -f results/amazon_v1/matches.csv ]; then echo STEP1_MISSING_OUTPUT; exit 1; fi
python3 code/amazon_tz_check.py > logs/amazon_tz_check.log 2>&1 || echo TZ_CHECK_FAILED
echo CHAIN_DONE
