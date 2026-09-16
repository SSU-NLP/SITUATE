#!/usr/bin/env bash
# D-B44 — 손실 가중치 2×2 (ALPHA × pos_weight) 재학습 런처.
#
# 🔒 codex 판정(2026-07-30, outputs/experiments/2026-07-30_m5_retrain_design/CODEX_ANSWER_2.md):
#   - 확률을 왜곡하는 것은 pos_weight 뿐. ALPHA 는 축별 경사 배분.
#   - ALPHA 균등화는 **합 보존**(0.63)이어야 총 손실 스케일·gamma 대비 비중이 안 변한다.
#   - beta/gamma/lr/scheduler/clipping 은 **건드리지 않는다**.
#   - W2(exact odds)는 여기 없다 — time 붕괴가 관측될 때만 발동하는 조건부 rescue.
#
# 🔴 dev-3 를 열지 않는다(--dev3 없음). D-B29 프로토콜 — 학습 중 선택셋 미개봉.
# 🔴 기존 동결 산출을 덮어쓰지 않는다: 네임스페이스에 alpha/pw 조건이 들어가고 --models M5 라
#    mtag=_M5 가 붙어 `M5_per_slot_slot_seed*.pt`(W0 원본)와 파일명이 절대 겹치지 않는다.
#
# 셀:
#   W0int (무결성 1 seed) : alpha current · pw clip   <- 기존 W0 재현되는지 대조용
#   W3    (3 seed)        : alpha current · pw one
#   W1    (3 seed, 주안)  : alpha uniform · pw one
#   W4    (3 seed)        : alpha uniform · pw clip
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
LOG=outputs/experiments/2026-07-30_m5_retrain_design/logs
mkdir -p "$LOG"

# tag|alpha|pw|seed
JOBS=(
  "W0int|current|clip|0"
  "W3|current|one|0"   "W3|current|one|1"   "W3|current|one|2"
  "W1|uniform|one|0"   "W1|uniform|one|1"   "W1|uniform|one|2"
  "W4|uniform|clip|0"  "W4|uniform|clip|1"  "W4|uniform|clip|2"
)
NGPU=${NGPU:-4}

for g in $(seq 0 $((NGPU - 1))); do
(
  i=$g
  while [ $i -lt ${#JOBS[@]} ]; do
    IFS='|' read -r tag am pm sd <<< "${JOBS[$i]}"
    echo "[gpu$g] start $tag alpha=$am pw=$pm seed=$sd  $(date -u +%H:%M:%S)UTC"
    CUDA_VISIBLE_DEVICES=$g $PY -u scripts/eacl_phase0/train_anchor_m4m5.py \
      --split phase1 --models M5 --batch 48 --pred-space slot --dynamics per_slot \
      --alpha-mode "$am" --pos-weight-mode "$pm" --seed "$sd" \
      > "$LOG/${tag}_seed${sd}.log" 2>&1
    echo "[gpu$g] done  $tag seed$sd rc=$? $(date -u +%H:%M:%S)UTC"
    i=$((i + NGPU))
  done
) &
done
wait
echo "ALL DONE $(date -u +%H:%M:%S)UTC"
