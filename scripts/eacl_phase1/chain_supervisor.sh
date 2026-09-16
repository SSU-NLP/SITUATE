#!/bin/bash
# 인과 사슬 축 무인 완주 감독기 — OpenRouter 프록시. GPU·VESSL job 을 쓰지 않는다.
#
# 동시성 자동 감쇠: W=200 에서 시작해 **터지면 10씩 낮춘다**(하한 20). 패스마다 resume 이므로
# 중단 손실이 없다. 모든 패스의 진행·실패율·실비·W 를 저널에 append.
#
# 사용: bash chain_supervisor.sh <MODEL> <OUT_DIR> <TAG> [W0]
set -u
MODEL="${1:?}"; OUT="${2:?}"; TAG="${3:?}"; W="${4:-200}"
REPO=/root/Hi-OnTop
JOURNAL="$REPO/EACL/UNATTENDED_2026-08-01_chain_gptoss.md"
LOG="$REPO/outputs/runs/chain_sup_${TAG}.log"
SHARDS=(/shared/hiontop/data/shards_all/*.jsonl)
W_MIN=20; W_STEP=10; FAIL_MAX=5          # 실패율 5% 초과면 감쇠
COOL=30; MAX_PASS="${MAX_PASS:-60}"
mkdir -p "$(dirname "$LOG")" "$OUT"

kst(){ TZ=Asia/Seoul date '+%H:%M KST'; }
say(){ echo "$(TZ=Asia/Seoul date -Is) $*" | tee -a "$LOG"; }
journal(){ printf '\n%s\n' "$1" >> "$JOURNAL"; }

journal "### [$(kst)] 감독기 시작 — \`$TAG\` · 모델 \`$MODEL\` · W=$W (터지면 ${W_STEP}씩 감쇠, 하한 ${W_MIN})"
say "start model=$MODEL out=$OUT W=$W"

for pass in $(seq 1 "$MAX_PASS"); do
  PLOG="$REPO/outputs/runs/chain_${TAG}_p${pass}.log"
  say "pass $pass · W=$W 시작"
  "$REPO/.venv/bin/python" "$REPO/scripts/eacl_phase1/chain_shard.py" "${SHARDS[@]}" "$OUT" \
      --model "$MODEL" --proxy --reasoning-effort minimal \
      --workers "$W" --batch 20 --timeout 300 > "$PLOG" 2>&1
  rc=$?
  line=$(grep -E "^\[chain\] 완료" "$PLOG" | tail -1)
  ok=$(echo "$line" | sed -n 's/.*ok \([0-9]*\).*/\1/p'); ok=${ok:-0}
  fail=$(echo "$line" | sed -n 's/.*fail \([0-9]*\).*/\1/p'); fail=${fail:-0}
  cost=$(echo "$line" | sed -n 's/.*실비 \$\([0-9.]*\).*/\1/p'); cost=${cost:-0}
  ndocs=$(ls "$OUT" 2>/dev/null | wc -l)
  tot=$((ok+fail)); rate=0
  [ "$tot" -gt 0 ] && rate=$(( fail*100/tot ))
  say "pass $pass rc=$rc ok=$ok fail=$fail(${rate}%) 문서=$ndocs 실비=\$$cost"

  # 완주: 정상 종료 + 실패 0 + 신규 라벨 없음(전부 cached)
  cached=$(grep -c "cached" "$PLOG" 2>/dev/null || echo 0)
  if [ "$rc" -eq 0 ] && [ "$fail" -eq 0 ]; then
    journal "### [$(kst)] ✅ \`$TAG\` 완주 — 문서 ${ndocs}편 · pass ${pass} · W=$W · 이 패스 실비 \$${cost}"
    say "완주"; exit 0
  fi

  if [ "$rc" -ne 0 ] || [ "$rate" -gt "$FAIL_MAX" ]; then
    if [ "$W" -gt "$W_MIN" ]; then
      W=$(( W - W_STEP )); [ "$W" -lt "$W_MIN" ] && W=$W_MIN
      journal "### [$(kst)] ⚠️ \`$TAG\` pass ${pass} 실패율 ${rate}% (rc=$rc) → **W=${W} 로 감쇠**. 문서 ${ndocs}편 · 누적 실비 \$${cost}"
      say "감쇠 → W=$W"
    else
      journal "### [$(kst)] 🔴 \`$TAG\` W 하한(${W_MIN}) 에서도 실패율 ${rate}% — 계속 재시도한다. 문서 ${ndocs}편"
      say "하한에서 재시도"
    fi
  else
    journal "### [$(kst)] \`$TAG\` pass ${pass} 진행 — 문서 **${ndocs}편** · 실패율 ${rate}% · W=$W · 실비 \$${cost}"
  fi
  sleep "$COOL"
done
journal "### [$(kst)] 🔴 \`$TAG\` pass 상한(${MAX_PASS}) 도달 — 미완. 문서 $(ls "$OUT" | wc -l)편"
say "pass 상한 도달"; exit 1
