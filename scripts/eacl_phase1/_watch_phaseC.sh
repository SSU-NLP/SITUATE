#!/bin/bash
# Phase C(통합 master full 패스) 감시 — reset_matrix.jsonl 재생성 완료까지.
set -u
LOG=/root/Hi-OnTop/outputs/runs/watch_phaseC.log
RM=/root/Hi-OnTop/outputs/eacl_phase0/phase1_all_v31/reset_matrix.jsonl
DRV=/root/Hi-OnTop/outputs/runs/phaseC_personkey.log
for i in $(seq 1 120); do
  n=$(grep -c "^\[pg.*\] stage1 " "$DRV" 2>/dev/null)
  alive=$(ps -eo args | grep -c "personkey_cap20_combined_maste[r]")
  sz=$(stat -c%s "$RM" 2>/dev/null)
  ts=$(stat -c%Y "$RM" 2>/dev/null)
  echo "$(TZ=Asia/Seoul date -Is) tick=$i docs=$n alive=$alive rm_size=$sz" >> "$LOG"
  if [ "$alive" -eq 0 ]; then echo "END: 프로세스 종료 · docs=$n · rm_size=$sz" >> "$LOG"; exit 0; fi
  sleep 60
done
echo "TIMEOUT: 2시간" >> "$LOG"; exit 3
