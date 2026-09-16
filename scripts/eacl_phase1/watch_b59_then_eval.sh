#!/bin/bash
# 무인 워처 — F4×I2 학습(b59) 체크포인트 3개가 나오면 dev-3 2×2 표를 재산출해 저널에 적는다.
set -u
REPO=/root/Hi-OnTop
JOURNAL="$REPO/EACL/UNATTENDED_2026-08-01_chain_gptoss.md"
LOG="$REPO/outputs/runs/watch_b59.log"
CK="$REPO/outputs/experiments/m4m5_phase1_anchor"
SHARED_CK="/shared/hiontop/code/outputs/experiments/m4m5_phase1_anchor"
mkdir -p "$(dirname "$LOG")"
POLL="${POLL:-300}"; MAX_HOURS="${MAX_HOURS:-6}"
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
kst() { TZ=Asia/Seoul date '+%H:%M KST'; }
say() { echo "$(TZ=Asia/Seoul date -Is) $*" >> "$LOG"; }
journal() { printf '\n%s\n' "$1" >> "$JOURNAL"; }

NS=_phase1_nonong_i2_per_slot_slot_pwone
say "b59 워처 시작"
while :; do
  # job 이 /shared 에 쓰므로 로컬로 가져온다
  for sd in 0 1 2; do
    f="$SHARED_CK/M5${NS}_seed${sd}.pt"
    [ -f "$f" ] && [ ! -f "$CK/M5${NS}_seed${sd}.pt" ] && cp "$f" "$CK/" && say "체크포인트 회수 seed$sd"
  done
  n=$(ls "$CK"/M5${NS}_seed*.pt 2>/dev/null | wc -l)
  say "F4×I2 체크포인트 $n/3"
  if [ "$n" -ge 3 ]; then
    journal "### [$(kst)] ✅ F4×I2 학습 완료 — 2×2 빈칸 채움. dev-3 재평가 중."
    out=$(cd "$REPO" && CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eacl_phase0/full_comparison_dev3.py 2>&1 | tail -24)
    journal "### [$(kst)] dev-3 전체 비교 재산출 (F4×I2 포함)
\`\`\`
${out}
\`\`\`
🔴 판정은 하지 않는다 — 아침에 사용자 확인."
    say "평가 완료"; exit 0
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    journal "### [$(kst)] 🔴 b59 워처 타임아웃 — 체크포인트 $n/3 만 나왔다."
    say "타임아웃"; exit 1
  fi
  sleep "$POLL"
done
