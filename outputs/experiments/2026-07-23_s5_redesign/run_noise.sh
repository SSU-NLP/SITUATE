#!/bin/bash
set -e
d="$1"; a="$2"; k="$3"
out="outputs/eacl_phase0/${d}_michelmann_s5_q36_35b_v2_r${k}"
master="outputs/eacl_phase0/${d}_michelmann_master.jsonl"
log="outputs/experiments/2026-07-23_s5_redesign/logs/noise_r${k}_${d}_${a}.log"
.venv/bin/python scripts/eacl_phase0/llm_s5_states.py "$master" "$out" \
  --axis "$a" --fixes "" --no-reasoning --engine vllm \
  --base-url http://localhost:8001/v1 --model qwen36-35b > "$log" 2>&1
echo "DONE r${k}/${d}/${a}"
