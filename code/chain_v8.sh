#!/bin/sh
# Certifiable domain (fast), then the real-shift experiment; logs per step.
cd /workspace/ravi/swot || exit 1
python3 code/sword_domain_v1.py > logs/sword_domain_v1.log 2>&1 || echo DOMAIN_FAILED
python3 code/real_shift_v8.py > logs/real_shift_v8.log 2>&1 || echo REAL_SHIFT_FAILED
echo CHAIN_DONE
