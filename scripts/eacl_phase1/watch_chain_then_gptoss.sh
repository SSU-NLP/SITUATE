#!/bin/bash
# 무인 워처 — Qwen 인과 사슬(b58-chain) 이 전부 끝나면 gpt-oss 판(b60-chain-oss)을 제출한다.
# 저널: EACL/UNATTENDED_2026-08-01_chain_gptoss.md 에 시각·결과를 append.
# 실패해도 조용히 죽지 않는다 — 모든 종료 경로에서 저널에 사유를 남긴다.
set -u
REPO=/root/Hi-OnTop
JOURNAL="$REPO/EACL/UNATTENDED_2026-08-01_chain_gptoss.md"
LOG="$REPO/outputs/runs/watch_chain_gptoss.log"
mkdir -p "$(dirname "$LOG")"
MAX_HOURS="${MAX_HOURS:-12}"
POLL="${POLL:-300}"
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))

kst() { TZ=Asia/Seoul date '+%H:%M KST'; }
say() { echo "$(TZ=Asia/Seoul date -Is) $*" >> "$LOG"; }
journal() { printf '\n%s\n' "$1" >> "$JOURNAL"; }

say "워처 시작 (poll=${POLL}s, 최대 ${MAX_HOURS}h)"
while :; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    journal "### [$(kst)] 🔴 워처 타임아웃 — b58 이 ${MAX_HOURS}시간 안에 안 끝났다. gpt-oss 미제출."
    say "타임아웃 종료"; exit 1
  fi
  alive=$(vesslctl job list 2>/dev/null | awk '/b58-chain/ && ($3=="running"||$3=="created"||$3=="pending"){n++} END{print n+0}')
  if [ -z "$alive" ]; then
    say "🔴 job list 조회 실패 — 재시도"
    sleep "$POLL"; continue
  fi
  done_docs=$(ls /shared/hiontop/out/phase1_chain_qwen_v31/ 2>/dev/null | grep -c '^pg' || echo 0)
  say "b58 잔여 $alive · Qwen 사슬 산출 ${done_docs}편"
  if [ "$alive" -eq 0 ]; then
    journal "### [$(kst)] ✅ b58-chain(Qwen) 전량 종료 — 산출 **${done_docs}편**. gpt-oss 판 제출 시작."
    out=$(bash /shared/hiontop/submit_chain_gptoss.sh 60 2>&1 | tail -1)
    journal "### [$(kst)] gpt-oss 사슬 제출 결과: \`${out}\`
- job 이름 \`b60-chain-oss-s0..59\` · 모델 \`/shared/models/gpt-oss-120b\` · \`--reasoning-parser openai_gptoss\`
- 산출 \`/shared/hiontop/out/phase1_chain_gptoss_v31/\`
- 러너 \`job_chain_v2.sh\`(v1 은 b58 실행 중이라 수정하지 않고 신규 작성)
- 프롬프트는 Qwen 판과 **동일**(진단 러너에서 AST 로 로드) — 라벨러만 다르다."
    say "제출 완료: $out"
    exit 0
  fi
  sleep "$POLL"
done
