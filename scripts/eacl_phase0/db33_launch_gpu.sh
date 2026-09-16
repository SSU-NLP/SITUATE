#!/usr/bin/env bash
# D-B33 보조 러너 — 대기열 일부를 **지정한 한 장의 GPU** 에서 순차/동시 실행한다.
#
# 사용: bash scripts/eacl_phase0/db33_launch_gpu.sh <queue.tsv> <gpu_id>
#
# 왜 별도 러너인가: `db29_launch.sh` 는 워커에 `CUDA_VISIBLE_DEVICES=$((w % NGPU))` 를 박아 넣어
# **물리 GPU 0..NGPU-1 만** 쓴다. 놀고 있는 특정 GPU(예: 2번)로 보낼 방법이 없고, 실행 중인
# 러너 스크립트를 고치는 것은 금지(D-B17 사고 — 실행 중 수정으로 조건이 조용히 바뀜).
#
# queue.tsv 각 줄 = "<번호>\t<태그>\t<train_anchor_m4m5.py 인자들>"  (db29_launch.sh 의 .queue 형식)
# - 🔴 dev-3 는 열지 않는다(--dev3 없음). D-B29/D-B33 프로토콜.
# - BASE 는 db29_launch.sh 와 **동일**해야 한다 — 조건이 갈리면 셀 비교가 무효가 된다.
set -uo pipefail
cd "$(dirname "$0")/../.."
LIST="${1:?대기열 TSV 경로}"
GPU="${2:?GPU 번호}"
OUT=${OUT:-outputs/experiments/inputpath_dynamics_ablation}   # 배치마다 다르다 (D-B35 = tagger_d1_ablation)
BASE="--split phase1 --batch 48 --pred-space slot --dynamics per_slot"
mkdir -p "$OUT/logs"

while IFS=$'\t' read -r n tag rest; do
  [ -z "${n:-}" ] && continue
  log="$OUT/logs/${tag}_${n}.log"
  echo "[gpu$GPU] #$n $tag $rest" | tee "$log"
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/eacl_phase0/train_anchor_m4m5.py \
    $BASE $rest >>"$log" 2>&1 &
done < "$LIST"
wait

for line in $(cut -f1 "$LIST"); do :; done
while IFS=$'\t' read -r n tag rest; do
  log="$OUT/logs/${tag}_${n}.log"
  j=$(grep -oP 'J_anchor = \K.*' "$log" | tail -1)
  echo -e "${tag}\t${n}\t0\t${j:-CHECK_LOG}\t${rest}" >> "$OUT/results.tsv"
done < "$LIST"
echo "[gpu$GPU] 전부 완료"
