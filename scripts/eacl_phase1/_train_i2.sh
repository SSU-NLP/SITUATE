#!/bin/bash
# M5 재학습(로컬). M4 는 VESSL 4장으로 분리 — 모델을 쪼개면 완료 시각이 절반 — 의도 축 **I2** · epoch 100 · 10 epoch 마다 완결 스냅숏.
# 조기종료 유지(--patience 3). 상한만 100 으로 올려 40 에서 잘리던 것을 푼다.
# 최종 저장본은 여전히 **val 손실 최적 epoch** 가중치로 복원된다.
set -u
cd /root/Hi-OnTop
CORPUS=data/eacl_phase1/pg_shorts_personkey_cap20_scant.jsonl
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s nohup .venv/bin/python scripts/eacl_phase0/train_anchor_m4m5.py \
    --split "phase1__personkey_i2" --models M5 --seed "$s" --dev3 \
    --epochs 100 --patience 3 --ckpt-every 10 \
    --group-split author --author-map "$CORPUS" --ns-suffix i2e100 \
    > "outputs/runs/train_i2e100_seed${s}.log" 2>&1 &
  sleep 3
done
wait
echo "$(TZ=Asia/Seoul date -Is) I2_E100_DONE" >> outputs/runs/train_i2.log
