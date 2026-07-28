#!/bin/bash
# $1=doc $2=axis $3=model-slug $4=out-suffix $5=prompt-version
set -e
d="$1"; a="$2"; model="$3"; suf="$4"; pv="$5"
M="outputs/eacl_phase0/${d}_michelmann_master.jsonl"
out="outputs/eacl_phase0/${d}_michelmann_s5_${suf}"
log="outputs/experiments/2026-07-23_s5_redesign/logs/${suf}_${d}_${a}.log"
.venv/bin/python scripts/eacl_phase0/llm_s5_states.py "$M" "$out" \
  --axis "$a" --prompt-version "$pv" --model "$model" > "$log" 2>&1
echo "DONE ${suf}/${d}/${a}"
