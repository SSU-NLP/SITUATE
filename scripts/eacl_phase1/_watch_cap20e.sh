#!/bin/bash
# delta-cap20e 완주 감시 — Job 이 다 끝나거나(성공/실패) 3시간 초과하면 나간다.
set -u
OUT=/shared/hiontop/out/delta-cap20e
LOG=/root/Hi-OnTop/outputs/runs/watch_cap20e.log
for i in $(seq 1 60); do
  st=$(timeout 60 vesslctl job list 2>/dev/null | awk -F'\t' '$2 ~ /delta-cap20e/ {print $3}' | sort | uniq -c | tr '\n' ' ')
  nrun=$(timeout 60 vesslctl job list 2>/dev/null | awk -F'\t' '$2 ~ /delta-cap20e/ && ($3=="running"||$3=="created"||$3=="pending")' | wc -l)
  na=$(find "$OUT" -maxdepth 2 -name '*_states.jsonl' 2>/dev/null | wc -l)
  rem=$(tail -1 /root/Hi-OnTop/outputs/runs/supervise_delta.log | grep -oE 'rem_l=[0-9]+ rem_v=[0-9]+')
  echo "$(TZ=Asia/Seoul date -Is) tick=$i states=[$st] cap20e_axis=$na $rem" >> "$LOG"
  [ "$nrun" -eq 0 ] && { echo "END: Job 전부 종료 [$st] · cap20e 축파일 $na개" >> "$LOG"; exit 0; }
  sleep 180
done
echo "TIMEOUT: 3시간" >> "$LOG"; exit 3
