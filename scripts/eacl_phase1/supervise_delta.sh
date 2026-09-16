#!/bin/bash
# 델타 라벨링 무인 완주 감독기 — 로컬 vLLM 4복제 + VESSL Job 4개.
#
# 오늘 세 번 넘어졌고 전부 사람이 붙어야 복구됐다(ninja PATH · VESSL health 타임아웃 ·
# 편/전이 형식 혼동). 셋 다 고쳤지만 감시자가 없으면 밤중에 하나만 죽어도 아침까지 멈춘다.
# 이 스크립트가 그 상태를 없앤다.
#
# 3분 주기로 세 가지를 복구한다:
#   ① vLLM 복제 health — 죽은 복제만 재기동 (PATH 에 ninja, 모델은 로컬 복사본)
#   ② 로컬 드라이버 생존 — 죽었고 남은 일이 있으면 재기동 (캐시라 재개 안전)
#   ③ VESSL Job — 전부 종료됐는데 미완이면 재제출. ⚠️ 워크로드 토큰은 terminate 권한이 없어
#      과금 중단은 사람이 해야 한다 — 저널에 경고만 남긴다.
#
# 모든 결정은 저널에 즉시 append 한다 — 복귀 시 그 파일만 읽으면 밤새 일이 다 보이도록.
set -u
REPO=/root/Hi-OnTop
cd "$REPO"

JOURNAL="$REPO/EACL/UNATTENDED_2026-08-02_delta_cap20.md"
LOG="$REPO/outputs/runs/supervise_delta.log"
OUT="$REPO/outputs/eacl_phase0/phase1_all_v31"
LOCAL_SHARDS="$REPO/data/eacl_phase1/delta_local"
VESSL_SHARDS="$REPO/data/eacl_phase1/delta_vessl"
VESSL_RUN="${VESSL_RUN:-delta-cap20c}"
VESSL_OUT="/shared/hiontop/out/$VESSL_RUN"
MODEL=/root/models/Qwen3.6-35B-A3B
NREP=4
P="${P:-96}"
TICK=180                 # 감시 주기(초)
STALL_TICKS=10           # 30분 정체하면 드라이버 재기동
VESSL_RESUBMIT_MAX=3
REP_FAIL_MAX=5

mkdir -p "$(dirname "$LOG")" "$OUT"
export PATH="$REPO/.venv-vllm/bin:$PATH"

kst(){ TZ=Asia/Seoul date '+%m-%d %H:%M KST'; }
say(){ echo "$(TZ=Asia/Seoul date -Is) $*" | tee -a "$LOG"; }
journal(){ printf '\n- **[%s]** %s\n' "$(kst)" "$1" >> "$JOURNAL"; }

# 남은 일 = 5축이 다 없는 문서 수. 인자: 샤드 디렉터리
remaining(){
  "$REPO/.venv/bin/python" - "$1" "$OUT" <<'PY'
import json, sys
from pathlib import Path
shards, out = Path(sys.argv[1]), Path(sys.argv[2])
want = {json.loads(l)["doc_id"] for f in shards.glob("*.jsonl") for l in open(f)}
AX = ("protagonist", "space", "time", "causality", "intentionality")
print(sum(1 for d in want if not all((out / d / f"{a}_states.jsonl").exists() for a in AX)))
PY
}
total_docs(){
  "$REPO/.venv/bin/python" - "$1" <<'PY'
import json, sys
from pathlib import Path
print(len({json.loads(l)["doc_id"] for f in Path(sys.argv[1]).glob("*.jsonl") for l in open(f)}))
PY
}

rep_up(){ curl -sf -m 30 "http://127.0.0.1:$((8001+$1))/v1/models" >/dev/null 2>&1; }

start_rep(){
  local g=$1
  command -v ninja >/dev/null || { journal "🔴 ninja 를 PATH 에서 못 찾음 — 복제 $g 기동 포기"; return 1; }
  CUDA_VISIBLE_DEVICES=$g nohup "$REPO/.venv-vllm/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen36-35b --quantization fp8 \
    --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --max-num-seqs 300 --reasoning-parser qwen3 --trust-remote-code --enable-prefix-caching \
    --port $((8001+g)) > "$REPO/outputs/runs/vllm_delta_$g.log" 2>&1 &
  disown
  say "복제 $g 재기동 (GPU $g · port $((8001+g)))"
}

start_driver(){
  setsid nohup bash "$REPO/scripts/eacl_phase1/run_label_delta.sh" \
    "$LOCAL_SHARDS" "$OUT" "$NREP" "$P" \
    > "$REPO/outputs/eacl_phase0/delta_local_label.log" 2>&1 < /dev/null &
  disown
  say "드라이버 기동 P=$P"
}

driver_alive(){ pgrep -f "run_label_delta.sh" >/dev/null 2>&1; }

vessl_states(){ vesslctl job list 2>/dev/null | awk -F'\t' '$2 ~ /delta-cap20/ {print $3}'; }
vessl_slugs(){ vesslctl job list 2>/dev/null | awk -F'\t' '$2 ~ /delta-cap20/ && ($3=="running"||$3=="created") {print $1}'; }

LT=$(total_docs "$LOCAL_SHARDS"); VT=$(total_docs "$VESSL_SHARDS")
journal "감독기 시작 — 로컬 ${LT}편(P=$P·복제 $NREP) · VESSL ${VT}편(\`$VESSL_RUN\`) · 총 $((LT+VT))편"
say "start local=$LT vessl=$VT P=$P"

rep_fail=(0 0 0 0); vessl_resubmits=0; stall=0; last_rem=999999; tick=0
while :; do
  tick=$((tick+1))

  # ── ① vLLM 복제 ────────────────────────────────────────────────────
  up=0
  for g in $(seq 0 $((NREP-1))); do
    if rep_up "$g"; then up=$((up+1)); rep_fail[$g]=0
    else
      pgrep -f "port $((8001+g))" >/dev/null 2>&1 && continue   # 기동 중이면 기다린다
      rep_fail[$g]=$(( ${rep_fail[$g]} + 1 ))
      if [ "${rep_fail[$g]}" -le "$REP_FAIL_MAX" ]; then
        journal "복제 $g 무응답 → 재기동 (${rep_fail[$g]}/$REP_FAIL_MAX)"
        start_rep "$g"
      elif [ "${rep_fail[$g]}" -eq $((REP_FAIL_MAX+1)) ]; then
        journal "🔴 복제 $g 를 $REP_FAIL_MAX 회 재기동해도 안 뜬다 — 포기하고 남은 복제로 계속"
      fi
    fi
  done

  # VESSL 은 /shared 에 쓴다 — 병합해야 진행률에 반영된다. 이미 있는 파일은 건드리지 않는다.
  for d in /shared/hiontop/out/delta-cap20*/; do
    [ -d "$d" ] || continue
    cp -rn "$d"/pg*/ "$OUT"/ 2>/dev/null
  done

  rem_l=$(remaining "$LOCAL_SHARDS"); rem_v=$(remaining "$VESSL_SHARDS")
  rem=$((rem_l + rem_v))

  # ── ② 로컬 드라이버 ────────────────────────────────────────────────
  if [ "$rem_l" -gt 0 ] && [ "$up" -gt 0 ] && ! driver_alive; then
    journal "드라이버가 죽어 있고 로컬 미완 ${rem_l}편 남음 → 재기동"
    start_driver
  fi

  # ── ③ VESSL ────────────────────────────────────────────────────────
  states=$(vessl_states); nrun=$(echo "$states" | grep -c "running\|created\|pending" || true)
  if [ "$rem_v" -eq 0 ] && [ "$nrun" -gt 0 ]; then
    journal "⚠️ VESSL 담당분 완료인데 Job ${nrun}개가 아직 running — **워크로드 토큰은 terminate 권한이 없다.** 웹 콘솔이나 로그인된 셸에서 내려야 과금이 멈춘다"
  elif [ "$rem_v" -gt 0 ] && [ "$nrun" -eq 0 ] && [ -n "$states" ]; then
    if [ "$vessl_resubmits" -lt "$VESSL_RESUBMIT_MAX" ]; then
      vessl_resubmits=$((vessl_resubmits+1))
      VESSL_RUN="delta-cap20c-r$vessl_resubmits"; VESSL_OUT="/shared/hiontop/out/$VESSL_RUN"
      journal "VESSL Job 이 전부 종료됐는데 ${rem_v}편 미완 → 재제출 \`$VESSL_RUN\` ($vessl_resubmits/$VESSL_RESUBMIT_MAX)"
      N=4 RUN="$VESSL_RUN" HEALTH_MIN=75 bash "$REPO/scripts/vessl/submit_jobs.sh" >> "$LOG" 2>&1
    elif [ "$vessl_resubmits" -eq "$VESSL_RESUBMIT_MAX" ]; then
      vessl_resubmits=$((vessl_resubmits+1))
      journal "🔴 VESSL 재제출 $VESSL_RESUBMIT_MAX 회 실패 — VESSL 담당 ${rem_v}편을 로컬 샤드로 이관"
      cp "$VESSL_SHARDS"/*.jsonl "$LOCAL_SHARDS"/ 2>/dev/null
      pkill -f run_label_delta.sh 2>/dev/null; sleep 5; start_driver
    fi
  fi

  # ── 정체 감지 ──────────────────────────────────────────────────────
  if [ "$rem" -ge "$last_rem" ]; then stall=$((stall+1)); else stall=0; fi
  last_rem=$rem
  if [ "$stall" -ge "$STALL_TICKS" ]; then
    journal "⚠️ 30분간 진행 없음(미완 ${rem}편 고정) → 드라이버 재기동"
    pkill -f run_label_delta.sh 2>/dev/null; sleep 5; start_driver; stall=0
  fi

  # ── 30분마다 진행 기록 ─────────────────────────────────────────────
  if [ $((tick % 10)) -eq 0 ]; then
    journal "진행 — 로컬 $((LT-rem_l))/$LT · VESSL $((VT-rem_v))/$VT · 복제 $up/$NREP · 미완 ${rem}편"
  fi
  say "tick=$tick up=$up rem_l=$rem_l rem_v=$rem_v stall=$stall"

  # ── 완주 ───────────────────────────────────────────────────────────
  if [ "$rem" -eq 0 ]; then
    journal "✅ 완주 — 총 $((LT+VT))편 5축 주석 완료. VESSL Job 종료함"
    say "완주"; exit 0
  fi
  sleep "$TICK"
done
