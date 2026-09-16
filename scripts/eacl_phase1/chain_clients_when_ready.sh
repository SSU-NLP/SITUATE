#!/bin/bash
# 이미 떠 있는 vLLM 복제 4개가 ready 되면 사슬 클라이언트를 붙인다.
# (감시기의 ready 대기 15분이 모델 로딩 45분보다 짧아 중단됐다 — 로딩을 살리려 분리했다.)
set -u
REPO=/root/Hi-OnTop
OUT="$REPO/outputs/eacl_phase0/phase1_chain_qwen_v31"
SERVED=qwen36-35b
JOURNAL="$REPO/EACL/UNATTENDED_2026-08-01_chain_gptoss.md"
LOG="$REPO/outputs/runs/chain_clients.log"
NGPU=4; WORKERS="${WORKERS:-96}"; MAX_WAIT_MIN="${MAX_WAIT_MIN:-90}"
# 로컬 4 + VESSL job 2 = 워커 6개로 샤드를 6등분. 로컬은 그룹 0~3 을 맡는다.
NGROUPS="${NGROUPS:-6}"
kst(){ TZ=Asia/Seoul date '+%H:%M KST'; }
say(){ echo "$(TZ=Asia/Seoul date -Is) $*" | tee -a "$LOG"; }
journal(){ printf '\n%s\n' "$1" >> "$JOURNAL"; }

deadline=$(( $(date +%s) + MAX_WAIT_MIN*60 ))
say "ready 대기 시작 (최대 ${MAX_WAIT_MIN}분)"
while :; do
  n=0
  for g in $(seq 0 $((NGPU-1))); do
    curl -sf -m 5 "http://127.0.0.1:$((8001+g))/v1/models" >/dev/null && n=$((n+1))
  done
  say "ready $n/$NGPU"
  [ "$n" -ge "$NGPU" ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    journal "### [$(kst)] 🔴 vLLM ready $n/$NGPU — ${MAX_WAIT_MIN}분 내 미완, 클라이언트 미실행."
    say "타임아웃"; exit 1
  fi
  sleep 60
done
journal "### [$(kst)] ✅ vLLM 복제 ${NGPU}개 ready — 사슬 클라이언트 착수 (workers=${WORKERS})
기존 $(find "$OUT" -name causalchain_states.jsonl -size +0 | wc -l)편에 이어쓴다."

CP=()
for g in $(seq 0 $((NGPU-1))); do
  mapfile -t MY < <(ls /shared/hiontop/data/shards_all/*.jsonl | sort | awk -v s="$g" -v n="$NGROUPS" '((NR-1)%n)==s')
  say "복제 $g: 샤드 ${#MY[@]}개"
  "$REPO/.venv/bin/python" "$REPO/scripts/eacl_phase1/chain_shard.py" "${MY[@]}" "$OUT" \
    --model "$SERVED" --base-url "http://127.0.0.1:$((8001+g))/v1" --no-reasoning \
    --workers "$WORKERS" --batch 20 --timeout 900 \
    > "$REPO/outputs/runs/local_chain_g$g.log" 2>&1 &
  CP+=($!)
done
say "클라이언트 ${#CP[@]}개 실행"
for p in "${CP[@]}"; do wait "$p"; done
nd=$(find "$OUT" -name causalchain_states.jsonl -size +0 | wc -l)
fl=$(grep -ho "FAIL [A-Za-z]*" "$REPO"/outputs/runs/local_chain_g*.log 2>/dev/null | wc -l)
journal "### [$(kst)] ✅ Qwen3.6 사슬 완주 — 라벨 보유 **${nd}편** · 실패 ${fl}건"
say "완주 $nd편 실패 $fl"
for p in $(ps -eo pid,args | grep "vllm.entrypoints.openai.api_serve[r]" | awk '{print $1}'); do kill "$p" 2>/dev/null; done
say "vLLM 종료"
