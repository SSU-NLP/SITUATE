#!/bin/bash
# 로컬 vLLM 4복제 재기동 + ready 후 사슬 클라이언트 착수 (그룹 0~3 / 6등분).
# 1차 실패 원인: `.venv-vllm/bin` 이 PATH 에 없어 vLLM 이 JIT 컴파일용 `ninja` 를 못 찾았다
# (가중치 26/26 로딩은 끝났는데 엔진 초기화에서 죽음). ⇒ PATH 를 명시적으로 넣는다.
set -u
REPO=/root/Hi-OnTop
export PATH="$REPO/.venv-vllm/bin:$PATH"
OUT="$REPO/outputs/eacl_phase0/phase1_chain_qwen_v31"
MODEL=/shared/models/Qwen3.6-35B-A3B
SERVED=qwen36-35b
JOURNAL="$REPO/EACL/UNATTENDED_2026-08-01_chain_gptoss.md"
LOG="$REPO/outputs/runs/relaunch_local_chain.log"
NGPU=4; NGROUPS=6; WORKERS="${WORKERS:-96}"; READY_MIN="${READY_MIN:-75}"
mkdir -p "$(dirname "$LOG")"
kst(){ TZ=Asia/Seoul date '+%H:%M KST'; }
say(){ echo "$(TZ=Asia/Seoul date -Is) $*" | tee -a "$LOG"; }
journal(){ printf '\n%s\n' "$1" >> "$JOURNAL"; }

command -v ninja >/dev/null || { journal "### [$(kst)] 🔴 ninja 를 PATH 에서 못 찾음 — 재기동 중단"; exit 1; }
say "ninja $(command -v ninja) 확인 · vLLM 4복제 기동"

PIDS=()
for g in $(seq 0 $((NGPU-1))); do
  port=$((8001+g))
  CUDA_VISIBLE_DEVICES=$g "$REPO/.venv-vllm/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$SERVED" --quantization fp8 \
    --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --max-num-seqs 512 --trust-remote-code --enable-prefix-caching \
    --port "$port" > "$REPO/outputs/runs/vllm_chain_$g.log" 2>&1 &
  PIDS+=($!)
  say "복제 $g 기동 (GPU $g · port $port)"
  sleep 45
done

deadline=$(( $(date +%s) + READY_MIN*60 ))
while :; do
  n=0
  for g in $(seq 0 $((NGPU-1))); do
    curl -sf -m 5 "http://127.0.0.1:$((8001+g))/v1/models" >/dev/null && n=$((n+1))
  done
  say "ready $n/$NGPU"
  [ "$n" -ge "$NGPU" ] && break
  alive=$(ps -eo pid | grep -cE "^\s*($(IFS='|'; echo "${PIDS[*]}"))$" || true)
  if [ "${alive:-0}" -eq 0 ]; then
    journal "### [$(kst)] 🔴 vLLM 복제가 전부 죽었다 — 클라이언트 미실행. 로그 \`outputs/runs/vllm_chain_*.log\`"
    say "전 복제 사망"; exit 1
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    journal "### [$(kst)] 🔴 ready $n/$NGPU — ${READY_MIN}분 초과. 클라이언트 미실행."
    say "ready 타임아웃"; exit 1
  fi
  sleep 60
done

journal "### [$(kst)] ✅ 로컬 vLLM ${NGPU}복제 ready — 사슬 클라이언트 착수(그룹 0~3 / ${NGROUPS}등분)
job b61(그룹 4·5)과 분담. 기존 $(find "$OUT" -name causalchain_states.jsonl -size +0 | wc -l)편에 이어쓴다."
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
journal "### [$(kst)] ✅ 로컬 사슬 클라이언트 종료 — 라벨 보유 **${nd}편**"
say "종료 $nd편 · vLLM 정리"
for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
