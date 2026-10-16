#!/bin/sh
# Final experiment chain (14 Oct 2026). Pre-final results are archived; the node download runs in parallel (network);
# GPU steps run in order with the tuned model (results/final/model_config.json, read by run_frontier_v5.XGB).
cd /workspace/ravi/swot || exit 1
mkdir -p results/archive_prefinal
for d in frontier_v5 robustness_v6 shift_v7 real_shift_v9 external_v1 external_v2; do
  [ -d results/$d ] && [ ! -d results/archive_prefinal/$d ] && cp -r results/$d results/archive_prefinal/
done
echo "archived $(date -u '+%H:%M UTC')"
python3 code/brazil_nodes.py > logs/brazil_nodes.log 2>&1 &
NODE_PID=$!
python3 code/tune_score.py > logs/tune_score.log 2>&1 || { echo TUNE_FAILED; exit 1; }
echo "tuning done $(date -u '+%H:%M UTC')"
python3 code/final_main.py > logs/final_main.log 2>&1 || echo FINAL_MAIN_FAILED
echo "final main done $(date -u '+%H:%M UTC')"
python3 code/run_robustness_v6.py > logs/run_robustness_v6_final.log 2>&1 || echo ROBUSTNESS_FAILED
python3 code/run_shift_v7.py > logs/run_shift_v7_final.log 2>&1 || echo SHIFT_FAILED
python3 code/real_shift_v9.py > logs/real_shift_v9_final.log 2>&1 || echo REAL_SHIFT_FAILED
echo "robustness and shifts done $(date -u '+%H:%M UTC')"
python3 code/external_test_v2.py > logs/external_v2_final.log 2>&1 || echo EXTERNAL_FAILED
TAU=0.5 python3 code/external_test_v2.py > logs/external_v2_tau05.log 2>&1 || echo EXTERNAL_TAU05_FAILED
TAU=0.1 python3 code/external_test_v2.py > logs/external_v2_tau01.log 2>&1 || echo EXTERNAL_TAU01_FAILED
python3 code/external_diverse.py > logs/external_diverse.log 2>&1 || echo DIVERSE_FAILED
echo "Brazil reach tests done $(date -u '+%H:%M UTC')"
wait $NODE_PID
echo "node download/pairing done $(date -u '+%H:%M UTC')"
TYPE=node python3 code/external_test_v2.py > logs/external_v2_node.log 2>&1 || echo EXTERNAL_NODE_FAILED
echo FINAL_CHAIN_DONE
