#!/bin/bash
set -e
d="$1"; a="$2"; model="$3"; variant="$4"; fixes="$5"; master="$6"
[ "$fixes" = "time_NONE" ] && fixes=""
out="outputs/eacl_phase0/${d}_s5_${variant}"
log="outputs/experiments/2026-07-23_s5_redesign/logs/${variant}_${d}.log"
.venv/bin/python scripts/eacl_phase0/llm_s5_states.py "$master" "$out" \
  --axis "$a" --fixes "$fixes" --no-reasoning --engine vllm \
  --base-url http://localhost:8001/v1 --model "$model" > "$log" 2>&1
echo "DONE ${variant}/${d}"
