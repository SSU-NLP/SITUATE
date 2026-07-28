#!/bin/bash
# Stage-0 정제본 재추출 — 동결 라벨러(Qwen3.6-35B-A3B no-thinking v2)를 repcorpus_master_stage0 위에서 재실행.
# 설정 동결·입력만 교체(동결 위반 아님, decision-log 2026-07-25). 새 out_dir = 정제본 전용(Stage-1 캐시 신규).
# race-safe: Phase A(protagonist)로 doc별 story_lists.json 선생성(문서별 파일=무충돌) → Phase B(나머지 4축,
# 캐시된 story_lists 읽음) → Phase C(full pass = carry_forward + reset_matrix). resume-safe(캐시 이어감).
set -u
cd /root/Hi-OnTop
export PY=/root/Hi-OnTop/.venv/bin/python
export MAS=outputs/eacl_phase0/repcorpus_masters_stage0
export COMBINED=outputs/eacl_phase0/repcorpus_master_stage0.jsonl
export OUT=outputs/eacl_phase0/repcorpus_s5_q36_35b_v2_stage0
export COMMON="--model qwen36-35b --engine vllm --no-reasoning --base-url http://localhost:8001/v1 --prompt-version v2"
mkdir -p "$OUT"
DOCS=$(cd "$MAS" && ls *.jsonl | sed 's/.jsonl//')

job() {  # $1=doc $2=axis
  local d="$1" a="$2"
  $PY scripts/eacl_phase0/llm_s5_states.py "$MAS/$d.jsonl" "$OUT" $COMMON --axis "$a" \
    >> "$OUT/par_${d}_${a}.log" 2>&1 && echo "[ok] $d $a $(date +%H:%M:%S)" || echo "[FAIL] $d $a"
}
export -f job

echo "=== STAGE0 RELABEL start $(date +%H:%M:%S) -> $OUT ==="
echo "--- Phase A: stage1 prewarm + protagonist (-P8) $(date +%H:%M:%S) ---"
for d in $DOCS; do echo "$d protagonist"; done | xargs -P8 -n2 bash -c 'job "$@"' _

echo "--- Phase B: space/time/causality/intentionality (-P8) $(date +%H:%M:%S) ---"
for d in $DOCS; do for a in space time causality intentionality; do echo "$d $a"; done; done \
  | xargs -P8 -n2 bash -c 'job "$@"' _

echo "--- Phase C: full pass (carry_forward + reset_matrix) $(date +%H:%M:%S) ---"
$PY scripts/eacl_phase0/llm_s5_states.py "$COMBINED" "$OUT" $COMMON >> "$OUT/full.log" 2>&1
echo "=== STAGE0 RELABEL done $(date +%H:%M:%S): reset_matrix rows = $(wc -l < "$OUT/reset_matrix.jsonl" 2>/dev/null) ==="