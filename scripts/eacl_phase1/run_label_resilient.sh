#!/bin/bash
# Phase-1 앵커 라벨링 — resilient 드라이버 (문제 문서 skip + 계속).
#
# `run_label.sh` 는 샤드 하나가 Stage-1 JSON 절단으로 죽으면 그 샤드의 나머지 문서를 통째로 잃는다
# (shard35: 120중 2개, shard12: 동일 원인). 무인 운영이 안 되므로 전 샤드를 `resilient_shard.sh`
# 로 돌린다 — 실패 시 문제 문서만 빼고 재시도한다.
#
# 전부 resume-safe: {axis}_states.jsonl 캐시가 남아 이미 끝난 문서는 즉시 건너뛴다.
set -u
cd /root/Hi-OnTop
MASTER_DIR="${1:?usage: run_label_resilient.sh <shard dir> <out> <combined master> [P]}"
OUT="${2:?}"
COMBINED="${3:?}"
P="${4:-64}"
PY=/root/Hi-OnTop/.venv/bin/python
mkdir -p "$OUT"
SHARDS=$(cd "$MASTER_DIR" && ls *.jsonl)

echo "=== RESILIENT LABEL start $(date +%H:%M:%S) · shards=$(echo "$SHARDS" | wc -w) · P=$P ==="
echo "--- Phase A: protagonist (Stage-1 선생성 포함) ---"
for s in $SHARDS; do echo "$MASTER_DIR/$s protagonist"; done \
  | xargs -P"$P" -n2 bash -c 'bash scripts/eacl_phase1/resilient_shard.sh "$1" "$2" "'"$OUT"'"' _

echo "--- Phase B: space/time/causality/intentionality ---"
for a in space time causality intentionality; do
  for s in $SHARDS; do echo "$MASTER_DIR/$s $a"; done
done | xargs -P"$P" -n2 bash -c 'bash scripts/eacl_phase1/resilient_shard.sh "$1" "$2" "'"$OUT"'"' _

echo "--- Phase C: full pass (통합 master 1회, 캐시라 LLM 호출 0) ---"
$PY scripts/eacl_phase0/llm_s5_states.py "$COMBINED" "$OUT" \
  --model qwen36-35b --engine vllm --no-reasoning --base-url http://localhost:8001/v1 \
  --prompt-version v3.1 --stage1-max-tokens 2000 >> "$OUT/log_full.log" 2>&1 \
  && echo "[ok] full pass" || echo "[FAIL] full pass"

echo "=== RESILIENT LABEL done $(date +%H:%M:%S) · 건너뛴 문서 $(cat "$OUT/_skipped.log" 2>/dev/null | wc -l) ==="
echo "LABEL_ALLDONE"
