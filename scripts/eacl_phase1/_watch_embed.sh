#!/bin/bash
# mpnet 임베딩 빌드 감시 — 302만 문장, 가장 무거운 단계.
set -u
LOG=/root/Hi-OnTop/outputs/runs/watch_embed.log
DRV=/root/Hi-OnTop/outputs/runs/build_anchor_personkey.log
D=/root/Hi-OnTop/outputs/experiments/m4m5_phase1_anchor/data
for i in $(seq 1 120); do
  alive=$(ps -eo args | grep -c "build_anchor_datase[t].py")
  last=$(tail -1 "$DRV" | cut -c1-110)
  sz=$(ls -la $D/phase1_personkey_emb.npy 2>/dev/null | awk '{print $5}')
  echo "$(TZ=Asia/Seoul date -Is) tick=$i alive=$alive emb=${sz:-0} | $last" >> "$LOG"
  [ "$alive" -eq 0 ] && { echo "END: 종료 · emb=${sz:-0}" >> "$LOG"; exit 0; }
  sleep 120
done
echo "TIMEOUT: 4시간" >> "$LOG"; exit 3
