#!/bin/bash
# Phase-1 앵커 라벨링 — **2복제(TP=2 × 2) 분배** + resilient(문제 문서 skip).
#
# 실측(2026-07-26, 배치20 stage2, conc 200): 1장 6.01 콜/s · TP=2 7.88 · TP=4 9.45.
# TP 는 GPU 4배에 1.57배뿐(MoE 활성 3B → 통신 지배)이라, TP=2 두 복제(7.88×2=15.76)가
# TP=4 하나(9.45)보다 1.67배 빠르다. + 장애 격리(한 서버 죽어도 절반 계속).
#
# 분배: 짝수 샤드 → :8001 (GPU0,1) · 홀수 샤드 → :8002 (GPU2,3).
# max-model-len 32768 이라 Stage-1 컨텍스트 초과 0편(16384 에선 278편 제외였다).
set -u
cd /root/Hi-OnTop
MASTER_DIR="${1:?usage: run_label_2rep.sh <shard dir> <out> <combined master> [P_per_replica]}"
OUT="${2:?}"
COMBINED="${3:?}"
P="${4:-100}"          # 복제당 동시성. 합계 = 2P
U1=http://localhost:8001/v1
U2=http://localhost:8002/v1
PY=/root/Hi-OnTop/.venv/bin/python
mkdir -p "$OUT"
SHARDS=$(cd "$MASTER_DIR" && ls *.jsonl)

# "<shard> <axis> <url>" 목록을 만들어 샤드 번호 홀짝으로 서버를 가른다
gen() {
  local axis="$1"
  for s in $SHARDS; do
    n=$(echo "$s" | tr -dc '0-9'); n=$((10#$n))
    if [ $((n % 2)) -eq 0 ]; then u=$U1; else u=$U2; fi
    echo "$MASTER_DIR/$s $axis $u"
  done
}
run() { xargs -P"$P" -n3 bash -c \
  'bash scripts/eacl_phase1/resilient_shard.sh "$1" "$2" "'"$OUT"'" 20 "$3"' _ ; }

echo "=== 2REP LABEL start $(date +%H:%M:%S) · shards=$(echo "$SHARDS" | wc -w) · P/복제=$P ==="
echo "--- Phase A: protagonist (Stage-1 선생성 포함) ---"
gen protagonist | run
echo "--- Phase B: space/time/causality/intentionality ---"
for a in space time causality intentionality; do gen "$a"; done | run
echo "--- Phase C: full pass (통합 master 1회, 캐시라 LLM 호출 0) ---"
$PY scripts/eacl_phase0/llm_s5_states.py "$COMBINED" "$OUT" \
  --model qwen36-35b --engine vllm --no-reasoning --base-url $U1 \
  --prompt-version v3.1 --stage1-max-tokens 8000 >> "$OUT/log_full.log" 2>&1 \
  && echo "[ok] full pass" || echo "[FAIL] full pass"
echo "=== 2REP LABEL done $(date +%H:%M:%S) · 건너뜀 $(cat "$OUT/_skipped.log" 2>/dev/null | wc -l) ==="
echo "LABEL_ALLDONE"
