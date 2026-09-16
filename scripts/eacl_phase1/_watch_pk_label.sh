#!/bin/bash
# 인물키 재빌드 신규 43편 라벨링 완주 감시.
set -u
LOG=/root/Hi-OnTop/outputs/runs/watch_pk_label.log
DRV=/root/Hi-OnTop/outputs/eacl_phase0/delta_personkey_label.log
for i in $(seq 1 40); do
  n=$(/root/Hi-OnTop/.venv/bin/python - <<'PY'
import json
from pathlib import Path
out=Path("/root/Hi-OnTop/outputs/eacl_phase0/phase1_all_v31"); AX=("protagonist","space","time","causality","intentionality")
docs={json.loads(l)["doc_id"] for l in open("/root/Hi-OnTop/data/eacl_phase1/delta_personkey_transitions.jsonl")}
print(sum(1 for d in docs if all((out/d/f"{a}_states.jsonl").exists() for a in AX)))
PY
)
  echo "$(TZ=Asia/Seoul date -Is) tick=$i done=$n/43" >> "$LOG"
  grep -q LABEL_ALLDONE "$DRV" && { echo "DONE: 드라이버 종료 · 완료 $n/43" >> "$LOG"; exit 0; }
  [ "$n" -ge 43 ] && { echo "DONE: 43/43" >> "$LOG"; exit 0; }
  sleep 90
done
echo "TIMEOUT: 60분" >> "$LOG"; exit 3
