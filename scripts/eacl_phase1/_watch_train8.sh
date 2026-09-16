#!/bin/bash
set -u
LOG=/root/Hi-OnTop/outputs/runs/watch_train8.log
for i in $(seq 1 120); do
  alive=$(ps -eo args | grep -c "train_anchor_m4m[5].py")
  line=$(tail -1 /root/Hi-OnTop/outputs/runs/train_personkey_seed0.log | cut -c1-120)
  echo "$(TZ=Asia/Seoul date -Is) tick=$i alive=$alive | $line" >> "$LOG"
  [ "$alive" -eq 0 ] && { echo "END: 4 seed 종료" >> "$LOG"; exit 0; }
  sleep 180
done
echo "TIMEOUT: 6시간" >> "$LOG"; exit 3
