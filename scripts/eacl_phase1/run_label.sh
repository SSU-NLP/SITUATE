#!/bin/bash
# Phase-1 앵커 라벨링 드라이버 — (문서 × 축) 병렬.
#
# 동결 llm_s5_states.py 는 문서·축·배치를 전부 순차 처리한다. 8,000편을 그대로 돌리면 동시성 1
# (실측 0.29 콜/s)이라 수백 시간이 된다. run_repcorpus_stage0.sh 와 같은 구조로 병렬화한다:
#   Phase A: 문서별 protagonist (= Stage-1 story_lists 를 문서별로 선생성, 파일 분리라 무충돌)
#   Phase B: 나머지 4축 (캐시된 story_lists 를 읽음)
#   Phase C: 문서별 full 패스 (전부 캐시 → 즉시) = carry_forward + reset_matrix 조립
#
# resume-safe: {axis}_states.jsonl 의 기존 item_id 를 읽어 이어간다. kill 해도 손실 없음.
# ⚠️ --stage1-max-tokens 2000: 기본 8000 은 ami 에서 컨텍스트 초과로 런을 죽였다(D-01).
set -u
cd /root/Hi-OnTop

MASTER_DIR="${1:?usage: run_label.sh <per-doc master dir> <out dir> [parallel]}"
OUT="${2:?}"
P="${3:-32}"

export PY=/root/Hi-OnTop/.venv/bin/python
export OUT
export COMMON="--model qwen36-35b --engine vllm --no-reasoning \
--base-url http://localhost:8001/v1 --prompt-version v3.1 --stage1-max-tokens 2000"
mkdir -p "$OUT"
DOCS=$(cd "$MASTER_DIR" && ls *.jsonl | sed 's/\.jsonl$//')
export MASTER_DIR

job() {
  local d="$1" a="$2"
  $PY scripts/eacl_phase0/llm_s5_states.py "$MASTER_DIR/$d.jsonl" "$OUT" $COMMON --axis "$a" \
    >> "$OUT/log_${d}_${a}.log" 2>&1 && echo "[ok] $d $a" || echo "[FAIL] $d $a"
}
export -f job

echo "=== PHASE1 LABEL start $(date +%H:%M:%S) · docs=$(echo "$DOCS" | wc -w) · P=$P -> $OUT ==="
echo "--- Phase A: stage1 prewarm + protagonist ---"
for d in $DOCS; do echo "$d protagonist"; done | xargs -P"$P" -n2 bash -c 'job "$@"' _
echo "--- Phase B: space/time/causality/intentionality ---"
for d in $DOCS; do for a in space time causality intentionality; do echo "$d $a"; done; done \
  | xargs -P"$P" -n2 bash -c 'job "$@"' _
echo "--- Phase C: full pass (carry_forward + reset_matrix) — 통합 master 1회 ---"
# 🔴 문서별 병렬 금지: llm_s5_states 는 reset_matrix.jsonl 을 out_dir 에 **하나만** 쓴다.
# 문서별로 돌리면 서로 덮어써 마지막 문서 것만 남는다(카나리아에서 10,840 → 315 행으로 실측).
# 전 축이 캐시돼 있으므로 통합 master 1회 패스는 LLM 호출 0, 수초.
COMBINED="${COMBINED:-}"
if [ -z "$COMBINED" ]; then
  COMBINED="$OUT/_combined_master.jsonl"
  cat "$MASTER_DIR"/*.jsonl > "$COMBINED"
fi
$PY scripts/eacl_phase0/llm_s5_states.py "$COMBINED" "$OUT" $COMMON \
  >> "$OUT/log_full.log" 2>&1 && echo "[ok] full pass" || echo "[FAIL] full pass"
echo "=== PHASE1 LABEL done $(date +%H:%M:%S) ==="
echo "LABEL_ALLDONE"
