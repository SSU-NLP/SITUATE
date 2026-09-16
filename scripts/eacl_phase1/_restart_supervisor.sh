#!/bin/bash
# 감독기를 **하나만** 남기고 재기동한다.
#
# ⚠️ 감독기가 둘 이상 돌면 각자 VESSL 재제출 카운터를 따로 세어 Job 이 8장으로 불어난다
# (2026-08-03 실측: delta-cap20c-r1/r2 가 중복 제출됨). 워크로드 토큰은 terminate 권한이
# 없어 사람이 웹 콘솔에서 내려야 하므로, 애초에 중복 기동을 막는 게 유일한 방어다.
set -u
REPO=/root/Hi-OnTop
cd "$REPO"

before=$(pgrep -fc 'supervise_delta.sh' || echo 0)
pkill -f 'supervise_delta.sh' 2>/dev/null
sleep 3
pkill -9 -f 'supervise_delta.sh' 2>/dev/null
sleep 2
echo "감독기 정리: ${before}개 → $(pgrep -fc 'supervise_delta.sh' || echo 0)개"

P="${P:-192}" VESSL_RUN="${VESSL_RUN:-delta-cap20c}" \
  setsid nohup bash "$REPO/scripts/eacl_phase1/supervise_delta.sh" \
  > "$REPO/outputs/runs/supervise_boot.log" 2>&1 < /dev/null &
disown
sleep 5
echo "기동 후 감독기 $(pgrep -fc 'supervise_delta.sh' || echo 0)개 (P=${P:-192})"
