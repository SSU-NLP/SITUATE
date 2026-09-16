#!/bin/bash
# 델타 라벨링 재시작 — 동시성 상향용 일회성 헬퍼.
#
# ⚠️ 이 파일로 뺀 이유: `pkill -f run_label_delta` 를 대화형 셸에서 치면 **그 셸 자신의 명령줄**에
# 패턴이 들어 있어 셸이 먼저 죽는다(오늘 3회 발생). 스크립트 파일 안에서는 cmdline 이
# "bash _restart_delta.sh" 라 패턴에 안 걸린다.
set -u
REPO=/root/Hi-OnTop
cd "$REPO"

pkill -f 'supervise_delta.sh' 2>/dev/null; sleep 2
export P="${P:-96}" VESSL_RUN="${VESSL_RUN:-delta-cap20c}"
echo "정리 전: 드라이버 $(pgrep -fc 'run_label_delta.sh' || echo 0) · 라벨러 $(pgrep -fc llm_s5_states || echo 0)"
pkill -f 'run_label_delta.sh' 2>/dev/null
sleep 3
pkill -f llm_s5_states 2>/dev/null
sleep 5
pkill -9 -f llm_s5_states 2>/dev/null
sleep 2
echo "정리 후: 드라이버 $(pgrep -fc 'run_label_delta.sh' || echo 0) · 라벨러 $(pgrep -fc llm_s5_states || echo 0)"

setsid nohup bash "$REPO/scripts/eacl_phase1/supervise_delta.sh" \
  > "$REPO/outputs/runs/supervise_boot.log" 2>&1 < /dev/null &
disown
echo "감독기 기동 (P=${P:-96})"
