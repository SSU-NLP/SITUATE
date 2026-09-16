#!/usr/bin/env bash
# GutenQA 검색 평가 배치 러너 — 셀 목록을 GPU 여러 장에 **작업큐**로 분배한다.
#
# 사용:
#   bash scripts/gutenqa/09_eval_launch.sh <cells.txt> ["<gpu:slots> ..."]
#   GPU_SLOTS="0:2 2:2 1:1 3:1" bash scripts/gutenqa/09_eval_launch.sh cells.txt
#
# `cells.txt` 각 줄 = `data/chunks/<셀>.jsonl` 의 셀 이름 하나 (빈 줄·`#` 주석 무시).
#
# ## 왜 db29_launch.sh 를 그대로 쓰지 않는가
# `db29_launch.sh` 는 워커에 `CUDA_VISIBLE_DEVICES=$((w % NGPU))` 를 박아 **물리 GPU 0..NGPU-1 을
# 균등하게** 쓴다. 여기서는 그게 안 된다 — **타 실험이 GPU 를 점유하고 있다**(실측 2026-07-30:
# `emllm/benchmark/pred.py` 2건이 GPU1·GPU3 에서 각 29GB, `train.py` 4건이 각 5~6GB).
# 남은 여유가 GPU 별로 다르므로 **GPU 마다 슬롯 수를 따로** 줘야 한다. `db33_launch_gpu.sh` 가
# "특정 GPU 지정" 을 위해 갈라진 것과 같은 이유다.
#
# 🔴 `db33_launch_gpu.sh` 의 교훈 승계: **실행 중인 러너 스크립트를 고치지 않는다**(D-B17 사고 —
# 실행 중 수정으로 조건이 조용히 바뀜). 슬롯 배치는 인자/환경변수로만 바꾼다.
#
# ## 메모리
# 셀당 피크는 batch 에 좌우된다. 타 작업 여유를 남기려고 `BATCH` 기본값을 하네스 기본(256)보다
# 낮춘 **128** 로 둔다. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 로 단편화를 줄인다.
set -uo pipefail
cd "$(dirname "$0")/../.."

LIST="${1:?셀 목록 파일 경로를 달라}"
OUT=${OUT:-outputs/experiments/2026-07-29_gutenqa_chunking}
# "gpu:slots" 공백구분. 인자 2 > 환경변수 > 기본값(4장 균등 1슬롯)
GPU_SLOTS="${2:-${GPU_SLOTS:-0:1 1:1 2:1 3:1}}"
BATCH=${BATCH:-128}
SKIP_DONE=${SKIP_DONE:-1}          # 이미 retrieval/<셀>.json 이 있으면 건너뛴다

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs"
Q="$OUT/.eval_queue"; LK="$OUT/.eval_lock"

grep -vE '^\s*(#|$)' "$LIST" | while read -r c; do
  if [ "$SKIP_DONE" = 1 ] && [ -f "$OUT/retrieval/$c.json" ]; then
    echo "[skip] $c (결과 있음)" >&2
  else
    echo "$c"
  fi
done > "$Q"
: > "$LK"

total=$(wc -l < "$Q")
nslots=0; for gs in $GPU_SLOTS; do nslots=$((nslots + ${gs##*:})); done
echo "[gutenqa-eval] 셀 $total개 · 슬롯 $nslots개 ($GPU_SLOTS) · batch $BATCH"
[ "$total" -eq 0 ] && { echo "[gutenqa-eval] 할 일 없음"; exit 0; }

worker() {
  local gpu=$1 wid=$2 line log code
  while true; do
    line=$(flock "$LK" -c "head -1 '$Q'; sed -i 1d '$Q'") || break
    [ -z "$line" ] && break
    log="$OUT/logs/eval_${line}.log"
    echo "[gpu$gpu/$wid] $line 시작 $(date +%T)"
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/gutenqa/02_eval_harness.py \
      --chunks "$OUT/data/chunks/${line}.jsonl" --name "$line" \
      --device cuda:0 --batch "$BATCH" > "$log" 2>&1
    code=$?
    d20=$(grep -oP 'DCG@20\s+\K[0-9.]+' "$log" | tail -1)
    if grep -q "CUDA out of memory" "$log"; then d20="OOM"; fi
    printf '%s\t%s\t%s\t%s\n' "$line" "$code" "${d20:-CHECK_LOG}" "$(date +%T)" >> "$OUT/eval_results.tsv"
    echo "[gpu$gpu/$wid] $line rc=$code DCG@20=${d20:-CHECK_LOG}"
  done
}

w=0
for gs in $GPU_SLOTS; do
  gpu=${gs%%:*}; slots=${gs##*:}
  for _ in $(seq 1 "$slots"); do w=$((w+1)); worker "$gpu" "$w" & done
done
wait
echo "[gutenqa-eval] 전부 완료"
[ -f "$OUT/eval_results.tsv" ] && sort -u "$OUT/eval_results.tsv" | column -t -s$'\t'
