#!/bin/bash
set -u
LOG=/root/Hi-OnTop/outputs/runs/watch_fix7.log
for i in $(seq 1 60); do
  n=$(/root/Hi-OnTop/.venv/bin/python /root/Hi-OnTop/scripts/eacl_phase1/qc_axis_completeness.py \
        /root/Hi-OnTop/data/eacl_phase1/personkey_cap20_combined_master.jsonl \
        /root/Hi-OnTop/outputs/eacl_phase0/phase1_all_v31 2>/dev/null | head -1)
  alive=$(ps -eo args | grep -c "s5_state[s].py")
  echo "$(TZ=Asia/Seoul date -Is) tick=$i alive=$alive | $n" >> "$LOG"
  grep -q FIX7_DONE /root/Hi-OnTop/outputs/runs/fix7.log 2>/dev/null && { echo "DONE" >> "$LOG"; exit 0; }
  sleep 180
done
echo "TIMEOUT" >> "$LOG"; exit 3
