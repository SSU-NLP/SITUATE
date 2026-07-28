#!/bin/bash
# 한 (doc,axis,arm) 추출. arm=on|off. v3 프롬프트.
set -e
d="$1"; a="$2"; arm="$3"
master="outputs/eacl_phase0/${d}_michelmann_master.jsonl"
if [ "$arm" = "off" ]; then
  out="outputs/eacl_phase0/${d}_michelmann_s5_eismv3_noreason"; flag="--no-reasoning"
else
  out="outputs/eacl_phase0/${d}_michelmann_s5_eismv3"; flag=""
fi
log="outputs/experiments/2026-07-23_s5_redesign/logs/${d}_${a}_${arm}.log"
.venv/bin/python scripts/eacl_phase0/llm_s5_states.py "$master" "$out" \
   --axis "$a" --prompt-version v3 $flag > "$log" 2>&1
echo "DONE ${d}/${a}/${arm}"
