#!/bin/bash
# 무인 워처 v2 — 인과 사슬(Qwen) 완료 감시 → 자가 복구 재제출 → 완료 시 gpt-oss 판 제출.
#
# v1 의 결함(실측으로 확인):
#   🔴 `vesslctl job list` 는 기본 **20개만** 반환한다(--per-page 필요). 그걸 완료 판정에 썼다가
#      b58 이 절반만 끝난 상태에서 "전량 종료"로 오판하고 gpt-oss 를 조기 제출했다.
#   🔴 진행률을 문서 디렉토리 개수로 셌는데 산출은 `parts/*.tar` 안에 있어 항상 0 으로 보였다.
# ⇒ v2 는 **파일시스템 사실**만 본다: 샤드별 `_stage_N.log` 에 "40 추출 완료" 가 찍혔는가.
#
# 자가 복구: 정체(기본 45분) 가 감지되면 미완 샤드를 재제출한다(최대 3회).
set -u
REPO=/root/Hi-OnTop
OUTD=/shared/hiontop/out/phase1_chain_qwen_v31
JOURNAL="$REPO/EACL/UNATTENDED_2026-08-01_chain_gptoss.md"
LOG="$REPO/outputs/runs/watch_chain_v2.log"
mkdir -p "$(dirname "$LOG")"
N_SHARDS="${N_SHARDS:-60}"; POLL="${POLL:-300}"
STALL_MIN="${STALL_MIN:-45}"; MAX_RESUB="${MAX_RESUB:-3}"; MAX_HOURS="${MAX_HOURS:-10}"
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
kst(){ TZ=Asia/Seoul date '+%H:%M KST'; }
say(){ echo "$(TZ=Asia/Seoul date -Is) $*" >> "$LOG"; }
journal(){ printf '\n%s\n' "$1" >> "$JOURNAL"; }

done_shards(){ local n=0 s; for s in $(seq 0 $((N_SHARDS-1))); do
    grep -q "40 추출 완료" "$OUTD/_stage_${s}.log" 2>/dev/null && n=$((n+1)); done; echo "$n"; }
missing(){ local s; for s in $(seq 0 $((N_SHARDS-1))); do
    grep -q "40 추출 완료" "$OUTD/_stage_${s}.log" 2>/dev/null || echo -n "$s "; done; }

last=$(done_shards); last_t=$(date +%s); resub=0
say "v2 시작 — 완료 $last/$N_SHARDS"
while :; do
  now=$(date +%s); cur=$(done_shards)
  [ "$cur" -gt "$last" ] && { last=$cur; last_t=$now; }
  say "완료 $cur/$N_SHARDS (정체 $(( (now-last_t)/60 ))분, 재제출 $resub회)"

  if [ "$cur" -ge "$N_SHARDS" ]; then
    docs=$(ls "$OUTD"/parts/*.tar 2>/dev/null | wc -l)
    journal "### [$(kst)] ✅ Qwen 인과 사슬 **전 샤드 완료** ($cur/$N_SHARDS · parts ${docs}개). gpt-oss 판 제출."
    if [ "$(ls /shared/hiontop/out/phase1_chain_gptoss_v31/ 2>/dev/null | wc -l)" -gt 0 ]; then
      out="이미 진행 중 — 중복 제출 안 함(b60 은 02:16 조기 제출분이 그대로 돈다)"
    else
      out=$(bash /shared/hiontop/submit_chain_gptoss.sh 60 2>&1 | tail -1)
    fi
    journal "### [$(kst)] gpt-oss 사슬 제출: \`$out\`
- \`b60-chain-oss-s0..59\` · \`/shared/models/gpt-oss-120b\` · \`--reasoning-parser openai_gptoss\`
- 산출 \`/shared/hiontop/out/phase1_chain_gptoss_v31/\` · 러너 \`job_chain_v2.sh\`
- 프롬프트는 Qwen 판과 동일(진단 러너에서 AST 로드). **라벨러만 다르다.**"
    say "gpt-oss 제출: $out"; exit 0
  fi

  if [ $(( (now-last_t)/60 )) -ge "$STALL_MIN" ] && [ "$resub" -lt "$MAX_RESUB" ]; then
    m=$(missing); resub=$((resub+1))
    out=$(bash /shared/hiontop/submit_chain_shards.sh "$N_SHARDS" $m 2>&1 | tail -1)
    journal "### [$(kst)] ⚠️ 정체 ${STALL_MIN}분 — 미완 샤드 재제출 #${resub}: \`$out\`
미완 = $m"
    say "재제출 #$resub: $out"; last_t=$now
  fi

  if [ "$now" -ge "$deadline" ]; then
    journal "### [$(kst)] 🔴 워처 타임아웃 — 완료 $cur/$N_SHARDS. gpt-oss **미제출**. 미완 = $(missing)"
    say "타임아웃"; exit 1
  fi
  sleep "$POLL"
done
