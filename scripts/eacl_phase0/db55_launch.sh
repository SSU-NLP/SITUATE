#!/usr/bin/env bash
# D-B55 — Gemma 앵커 M5 재학습 + dev-3 비교. 사전등록 = ledger D-B55.
#
# 🔒 학습 조건은 D-B44 W3 명령줄 **그대로**. 바뀌는 것은 --split(=앵커 코퍼스) 하나뿐.
#    ns 는 split 이 phase1 이 아니면 자동으로 `_phase1_gemma_...` 가 되므로 기존 W3
#    체크포인트와 파일명이 겹치지 않는다(train_anchor_m4m5.py:447).
# ⚠️ 실행 중 이 파일을 수정하지 말 것.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
LOG=outputs/experiments/2026-07-31_db55_anchor_swap/logs
mkdir -p "$LOG"
IDX=outputs/experiments/m4m5_phase1_anchor/data/phase1_gemma_index.npz

# ── 데이터셋 빌드 완료 대기 (인코딩이 별도 프로세스에서 돌고 있다) ──────────
echo "[wait] $IDX 대기 $(date -u +%H:%M:%S)UTC"
for _ in $(seq 1 720); do [ -f "$IDX" ] && break; sleep 10; done
[ -f "$IDX" ] || { echo "🔴 데이터셋 빌드 실패/지연 — 중단"; exit 1; }
sleep 5
echo "[ok] 데이터셋 준비됨 $(date -u +%H:%M:%S)UTC"

# ── W3 3 seed 병렬 (GPU 0/1/2) ────────────────────────────────────────────
for sd in 0 1 2; do
(
  echo "[gpu$sd] start seed=$sd $(date -u +%H:%M:%S)UTC"
  CUDA_VISIBLE_DEVICES=$sd $PY -u scripts/eacl_phase0/train_anchor_m4m5.py \
    --split phase1_gemma --models M5 --batch 48 --pred-space slot --dynamics per_slot \
    --alpha-mode current --pos-weight-mode one --seed "$sd" \
    > "$LOG/gemma_W3_seed${sd}.log" 2>&1
  echo "[gpu$sd] done seed=$sd rc=$? $(date -u +%H:%M:%S)UTC"
) &
done
wait
echo "[train] ALL DONE $(date -u +%H:%M:%S)UTC"

# ── dev-3 평가 (기준선 W3 와 나란히) ──────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/eacl_phase0/db55_anchor_swap_eval.py \
  2>&1 | tee "$LOG/eval.log"
echo "DB55_DONE"
