#!/bin/bash
# ①B 대표코퍼스 추출 — 2회(결정성 검사, D-B11). (문서×축) xargs -P4 병렬 후 full pass로 reset_matrix.
# resume-safe: {axis}_states.jsonl 캐시 이어감. 재실행 안전.
set -u
cd /root/Hi-OnTop
export PY=/root/Hi-OnTop/.venv/bin/python
export MAS=outputs/eacl_phase0/repcorpus_masters
export COMBINED=outputs/eacl_phase0/repcorpus_master.jsonl
DOCS=$(cd "$MAS" && ls *.jsonl | sed 's/.jsonl//')
AXES="protagonist space time causality intentionality"

run() {
  local N=$1
  export OUT=outputs/eacl_phase0/repcorpus_s5_eismv2_run$N
  mkdir -p "$OUT"
  echo "=== RUN $N start $(date +%H:%M:%S) -> $OUT ==="
  # parallel (doc x axis)
  for d in $DOCS; do for a in $AXES; do echo "$d $a"; done; done \
    | xargs -P4 -n2 bash -c '
        $PY scripts/eacl_phase0/llm_s5_states.py "$MAS/$0.jsonl" "$OUT" --axis "$1" \
          >> "$OUT/parallel_$0_$1.log" 2>&1 && echo "[ok] $0 $1 $(date +%H:%M:%S)" || echo "[FAIL] $0 $1"'
  echo "--- RUN $N full pass (carry_forward + reset_matrix) $(date +%H:%M:%S) ---"
  $PY scripts/eacl_phase0/llm_s5_states.py "$COMBINED" "$OUT" >> "$OUT/full.log" 2>&1
  echo "=== RUN $N done $(date +%H:%M:%S): reset_matrix rows = $(wc -l < "$OUT/reset_matrix.jsonl" 2>/dev/null) ==="
}

run 1
run 2
echo "ALL EXTRACTION DONE $(date +%H:%M:%S)"
