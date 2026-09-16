#!/usr/bin/env bash
# D-B29 배치 러너 — run 목록(TSV)을 GPU 4장에 워크큐로 분배한다.
#
# 사용: bash scripts/eacl_phase0/db29_launch.sh outputs/experiments/db29_batch/wave1.tsv
#
# TSV 각 줄 = "<묶음태그>\t<train_anchor_m4m5.py 인자들>"
# - 🔴 dev-3 는 열지 않는다 (--dev3 를 절대 넣지 않는다). D-B29 프로토콜.
# - 로그는 outputs/experiments/db29_batch/logs/<tag>_<n>.log
# - 산출 파일명 충돌은 train 스크립트의 args.ns 네임스페이스가 막는다.
set -uo pipefail
cd "$(dirname "$0")/../.."
LIST="${1:?run 목록 TSV 경로를 달라}"
# 산출 디렉토리는 배치마다 다르다 (D-B29 = db29_batch · D-B33 = inputpath_dynamics_ablation).
# 기본값은 D-B29 그대로라 기존 호출은 동작이 바뀌지 않는다.
OUT=${OUT:-outputs/experiments/db29_batch}
mkdir -p "$OUT/logs"
NGPU=${NGPU:-4}
BASE="--split phase1 --batch 48 --pred-space slot --dynamics per_slot"

grep -v '^\s*$' "$LIST" | nl -ba -w1 -s$'\t' > "$OUT/.queue"
total=$(wc -l < "$OUT/.queue")
echo "[db29] $total run · GPU $NGPU 장 × ${PER_GPU:-3} 워커 · dev-3 미개봉"

worker() {
  local gpu=$1
  while true; do
    # 원자적으로 한 줄 꺼내기 (flock)
    line=$(flock "$OUT/.lock" -c "head -1 $OUT/.queue; sed -i 1d $OUT/.queue") || break
    [ -z "$line" ] && break
    n=$(cut -f1 <<<"$line"); tag=$(cut -f2 <<<"$line"); rest=$(cut -f3- <<<"$line")
    log="$OUT/logs/${tag}_${n}.log"
    echo "[gpu$gpu] #$n $tag $rest" | tee "$log"
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/eacl_phase0/train_anchor_m4m5.py \
      $BASE $rest >>"$log" 2>&1
    code=$?
    j=$(grep -oP 'J_anchor = \K.*' "$log" | tail -1)
    echo -e "${tag}\t${n}\t${code}\t${j:-FAIL}\t${rest}" >> "$OUT/results.tsv"
    echo "[gpu$gpu] #$n done rc=$code  ${j:-FAIL}"
  done
}

: > "$OUT/.lock"
# GPU 사용률이 ~22% (시간축 순차 루프가 병목, 메모리 3.5GB) 이므로 GPU 당 여러 워커를 올린다.
PER_GPU=${PER_GPU:-3}
for w in $(seq 0 $((NGPU * PER_GPU - 1))); do worker "$((w % NGPU))" & done
wait
echo "[db29] 전부 완료"
sort -k1,1 -k2,2n "$OUT/results.tsv" | column -t -s$'\t'
