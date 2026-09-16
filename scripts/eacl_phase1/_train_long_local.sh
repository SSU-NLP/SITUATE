#!/bin/bash
# M4/M5 장기 학습 — 로컬 4장 = seed 0~3. epochs 40 · patience 3.
# epochs 8 에서는 4 seed × 2 모델 전부 마지막 epoch 가 최적이라 조기종료가 한 번도 안 걸렸다
# (= 아직 과적합 전, 8이 부족). 상한을 늘리고 조기종료에 맡긴다.
set -u
cd /root/Hi-OnTop
CORPUS=data/eacl_phase1/pg_shorts_personkey_cap20_scant.jsonl
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s nohup .venv/bin/python scripts/eacl_phase0/train_anchor_m4m5.py \
    --split "phase1__personkey" --models M5,M4 --seed "$s" --dev3 \
    --epochs 40 --patience 3 --group-split author --author-map "$CORPUS" --ns-suffix e40 \
    > "outputs/runs/train_e40_seed${s}.log" 2>&1 &
  sleep 3
done
wait
echo "$(TZ=Asia/Seoul date -Is) LOCAL_E40_DONE" >> outputs/runs/train_e40.log
