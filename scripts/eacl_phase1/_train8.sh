#!/bin/bash
# M4/M5 인물키 앵커 학습 — seed 병렬(로컬 4장 = seed 0~3).
# 분할 = **작가 단위**(--group-split author). 기본 none 은 val 의 96.1% 가 train 에
# 같은 작가를 두고 있어 held-out 이 아니다(2026-08-03 실측). book 도 55.7% 남는다.
set -u
cd /root/Hi-OnTop
LOG=outputs/runs/train8_es_personkey.log
CORPUS=data/eacl_phase1/pg_shorts_personkey_cap20_scant.jsonl
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s nohup .venv/bin/python scripts/eacl_phase0/train_anchor_m4m5.py \
    --split "phase1__personkey" --models M5,M4 --seed "$s" --dev3 \
    --group-split author --author-map "$CORPUS" --ns-suffix es \
    > "outputs/runs/train_es_personkey_seed${s}.log" 2>&1 &
  echo "$(TZ=Asia/Seoul date -Is) seed=$s GPU=$s 기동 (group-split=author)" >> "$LOG"
  sleep 3
done
wait
echo "$(TZ=Asia/Seoul date -Is) TRAIN8_LOCAL_DONE" >> "$LOG"
