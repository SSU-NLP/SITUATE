#!/bin/bash
# 축이 잘린 7편을 끝까지 채운다. llm_s5_states 는 resume-safe(done 부터 이어감)라
# 잘린 JSON 으로 죽어도 재실행하면 전진한다 — 그래서 문서별로 반복 재시도한다.
set -u
cd /root/Hi-OnTop
PY=.venv/bin/python
OUT=outputs/eacl_phase0/phase1_all_v31
LOG=outputs/runs/fix7.log
COMMON="--model qwen36-35b --engine vllm --no-reasoning --prompt-version v3.1 --stage1-max-tokens 8000"
i=0
for f in data/eacl_phase1/fix7_shards/*.jsonl; do
  d=$(basename "$f" .jsonl); port=$((8001 + i % 4)); i=$((i+1))
  ( for try in $(seq 1 25); do
      $PY scripts/eacl_phase0/llm_s5_states.py "$f" "$OUT" $COMMON \
        --base-url "http://127.0.0.1:$port/v1" --axis protagonist --axis intentionality \
        >> "$OUT/log_fix7_${d}.log" 2>&1 && { echo "$(date -Is) [ok] $d (try=$try)" >> "$LOG"; break; }
      echo "$(date -Is) [retry] $d $try" >> "$LOG"
    done ) &
done
wait
echo "$(date -Is) FIX7_DONE" >> "$LOG"
