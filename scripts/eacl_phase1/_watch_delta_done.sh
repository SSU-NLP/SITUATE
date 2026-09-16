#!/bin/bash
# 델타 라벨링 **전체** 완주 감시 — VESSL Job 종료 + 로컬 미완 0 을 둘 다 본다.
# 종료: 완주(0) / 30분 정체(4) / Job 전부 종료됐는데 로컬만 남음은 계속 대기 / 3시간 초과(3).
set -u
LOG=/root/Hi-OnTop/outputs/runs/watch_delta_done.log
last=999999; stall=0
for i in $(seq 1 90); do
  line=$(tail -1 /root/Hi-OnTop/outputs/runs/supervise_delta.log)
  rl=$(echo "$line" | grep -oE 'rem_l=[0-9]+' | cut -d= -f2)
  rv=$(echo "$line" | grep -oE 'rem_v=[0-9]+' | cut -d= -f2)
  nrun=$(timeout 60 vesslctl job list 2>/dev/null | awk -F'\t' '$2 ~ /delta-cap20e/ && ($3=="running"||$3=="created"||$3=="pending")' | wc -l)
  rem=$(( ${rl:-9999} + ${rv:-9999} ))
  echo "$(TZ=Asia/Seoul date -Is) tick=$i rem_l=${rl:-?} rem_v=${rv:-?} vessl_running=$nrun" >> "$LOG"
  if [ "$rem" -eq 0 ]; then echo "DONE: 미완 0 · VESSL running $nrun" >> "$LOG"; exit 0; fi
  if [ "$rem" -ge "$last" ]; then stall=$((stall+1)); else stall=0; fi
  last=$rem
  if [ "$stall" -ge 15 ]; then echo "STALL: 30분간 미완 ${rem}편 고정" >> "$LOG"; exit 4; fi
  sleep 120
done
echo "TIMEOUT: 3시간" >> "$LOG"; exit 3
