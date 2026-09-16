#!/usr/bin/env bash
# D-B62 ψₑ 재개봉 러너 — 조건 목록(TSV)을 로컬 GPU 4장에 워크큐로 분배한다.
#
#   bash scripts/eacl_phase0/psi_reopen_launch.sh outputs/experiments/psi_reopen_2026-08-01/runs.tsv
#
# TSV 각 줄 = "<조건태그>\t<train_anchor_m4m5.py 추가 인자>"
# - 🔴 `--dev3` 를 절대 넣지 않는다 (선별 단계 · dev-3 미개봉).
# - db29_launch.sh 를 고치지 않고 새로 둔 이유: 그쪽은 `--split phase1` 이 BASE 에 하드코딩돼
#   있어 과거 배치의 재현성이 걸려 있다. 실행 중/완료된 스크립트는 건드리지 않는다.
set -uo pipefail
cd "$(dirname "$0")/../.."
LIST="${1:?run 목록 TSV 경로를 달라}"
OUT=${OUT:-outputs/experiments/psi_reopen_2026-08-01}
mkdir -p "$OUT/logs"
NGPU=${NGPU:-4}
PER_GPU=${PER_GPU:-2}
# 두 split 기준선과 동일한 공통 설정. split·psi·head_feat 만 줄마다 다르다.
BASE="--models M5 --dynamics per_slot --pred-space slot --pos-weight-mode one --batch 48"

grep -v '^\s*$' "$LIST" | nl -ba -w1 -s$'\t' > "$OUT/.queue"
total=$(wc -l < "$OUT/.queue")
echo "[psi-reopen] $total run · GPU $NGPU 장 × $PER_GPU 워커 · dev-3 미개봉 · seed 0"

worker() {
  local gpu=$1
  while true; do
    line=$(flock "$OUT/.lock" -c "head -1 $OUT/.queue; sed -i 1d $OUT/.queue") || break
    [ -z "$line" ] && break
    n=$(cut -f1 <<<"$line"); tag=$(cut -f2 <<<"$line"); rest=$(cut -f3- <<<"$line")
    log="$OUT/logs/${tag}.log"
    echo "[gpu$gpu] #$n $tag $rest" | tee "$log"
    start=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/eacl_phase0/train_anchor_m4m5.py \
      $BASE $rest >>"$log" 2>&1
    code=$?
    j=$(grep -oP 'J_anchor = \K[0-9.]*' "$log" | tail -1)
    printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$code" "${j:-FAIL}" "$(( $(date +%s) - start ))s" "$rest" \
      >> "$OUT/results.tsv"
    echo "[gpu$gpu] #$n done rc=$code J_anchor=${j:-FAIL}"
  done
}

: > "$OUT/.lock"
for w in $(seq 0 $((NGPU * PER_GPU - 1))); do worker "$((w % NGPU))" & done
wait
echo "[psi-reopen] 전부 완료"
sort "$OUT/results.tsv" | column -t -s$'\t'
