#!/bin/bash
# emllm 다운스트림 실험이 끝나면 **로컬 GPU** 로 Qwen3.6 인과 사슬 잔여분을 완주한다.
# VESSL job 을 쓰지 않는다(사용자 지시). GPU 상한 = 로컬 4장.
#
# 판정: emllm 프로세스(`benchmark/pred.py`) 부재 + GPU 여유 메모리 확인. 10분 주기.
# 착수: GPU 당 vLLM 복제 1개(포트 8001..) → 샤드를 복제 수만큼 나눠 동시 처리 → resume.
# 산출은 기존 Qwen3.6 경로에 이어 쓴다(같은 모델·같은 프롬프트라 이어쓰기가 정당).
set -u
REPO=/root/Hi-OnTop
OUT="$REPO/outputs/eacl_phase0/phase1_chain_qwen_v31"
MODEL=/shared/models/Qwen3.6-35B-A3B
SERVED=qwen36-35b
JOURNAL="$REPO/EACL/UNATTENDED_2026-08-01_chain_gptoss.md"
LOG="$REPO/outputs/runs/local_chain_after_emllm.log"
NGPU="${NGPU:-4}"; POLL="${POLL:-600}"; MAX_WAIT_H="${MAX_WAIT_H:-14}"
WORKERS="${WORKERS:-96}"
mkdir -p "$(dirname "$LOG")" "$OUT"
kst(){ TZ=Asia/Seoul date '+%H:%M KST'; }
say(){ echo "$(TZ=Asia/Seoul date -Is) $*" | tee -a "$LOG"; }
journal(){ printf '\n%s\n' "$1" >> "$JOURNAL"; }

if ! "$REPO/.venv-vllm/bin/python" -c "import vllm" 2>/dev/null; then
  journal "### [$(kst)] 🔴 착수 중단 — \`.venv-vllm\` 에 vLLM 이 없다. 설치 후 재기동할 것."
  say "vLLM 없음 — 중단"; exit 1
fi
deadline=$(( $(date +%s) + MAX_WAIT_H*3600 ))
say "감시 시작 — emllm 종료 대기(10분 주기, 최대 ${MAX_WAIT_H}h)"
journal "### [$(kst)] emllm 종료 감시 시작 — 끝나면 **로컬 ${NGPU}장**으로 Qwen3.6 사슬 잔여분 착수(job 미사용)"

while :; do
  n=$(ps -eo args | grep -c "benchmark/pred\.p[y]")
  free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1<2000{c++} END{print c+0}')
  say "emllm 프로세스 $n · 여유 GPU $free/${NGPU}"
  if [ "$n" -eq 0 ] && [ "$free" -ge "$NGPU" ]; then
    journal "### [$(kst)] ✅ emllm 종료 확인 (여유 GPU ${free}장) — Qwen3.6 사슬 로컬 착수"
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    journal "### [$(kst)] 🔴 emllm 이 ${MAX_WAIT_H}시간 안에 안 끝났다 — 사슬 착수 안 함."
    say "대기 타임아웃"; exit 1
  fi
  sleep "$POLL"
done

# ── vLLM 복제 기동 (GPU 당 1개, 로드 I/O 경합 피하려 60초 간격) ──────────
PIDS=()
for g in $(seq 0 $((NGPU-1))); do
  port=$((8001+g))
  say "vLLM 복제 $g 기동 (GPU $g · port $port)"
  CUDA_VISIBLE_DEVICES=$g "$REPO/.venv-vllm/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$SERVED" --quantization fp8 \
    --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --max-num-seqs 512 --trust-remote-code --enable-prefix-caching \
    --port "$port" > "$REPO/outputs/runs/vllm_chain_$g.log" 2>&1 &
  PIDS+=($!)
  sleep 60
done
say "복제 ${#PIDS[@]}개 기동 — ready 대기"
bad=0
for g in $(seq 0 $((NGPU-1))); do
  port=$((8001+g)); ok=0
  for i in $(seq 1 90); do
    curl -sf -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null && { ok=1; break; }
    sleep 10
  done
  [ "$ok" -eq 1 ] || { bad=$((bad+1)); say "🔴 복제 $g (port $port) ready 실패"; }
done
if [ "$bad" -gt 0 ]; then
  journal "### [$(kst)] 🔴 착수 중단 — vLLM 복제 ${bad}/${NGPU} 개가 ready 되지 않았다. 클라이언트를 띄우지 않는다.
로그: \`outputs/runs/vllm_chain_*.log\`"
  say "ready 실패 $bad개 — 클라이언트 미실행, 종료"
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  exit 1
fi
say "전 복제 ready (검증됨)"

# ── 샤드를 복제 수만큼 분배해 동시 실행 (문서별 resume) ─────────────────
CPIDS=()
for g in $(seq 0 $((NGPU-1))); do
  port=$((8001+g))
  mapfile -t MY < <(ls /shared/hiontop/data/shards_all/*.jsonl | sort | awk -v s="$g" -v n="$NGPU" '((NR-1)%n)==s')
  say "복제 $g: 샤드 ${#MY[@]}개"
  "$REPO/.venv/bin/python" "$REPO/scripts/eacl_phase1/chain_shard.py" "${MY[@]}" "$OUT" \
    --model "$SERVED" --base-url "http://127.0.0.1:$port/v1" --no-reasoning \
    --workers "$WORKERS" --batch 20 --timeout 900 \
    > "$REPO/outputs/runs/local_chain_g$g.log" 2>&1 &
  CPIDS+=($!)
done
journal "### [$(kst)] 로컬 사슬 착수 — vLLM 복제 ${NGPU}개 · 클라이언트 ${#CPIDS[@]}개 · workers=${WORKERS}
기존 산출 $(ls "$OUT" | wc -l)편에 이어쓴다(같은 모델·프롬프트). 예상 2.5~3시간."
say "클라이언트 ${#CPIDS[@]}개 실행 — 완료 대기"
for p in "${CPIDS[@]}"; do wait "$p"; done

ndocs=$(ls "$OUT" | wc -l)
fails=$(grep -ho "🔴 [a-z0-9_]*: FAIL" "$REPO"/outputs/runs/local_chain_g*.log 2>/dev/null | wc -l)
journal "### [$(kst)] ✅ Qwen3.6 사슬 로컬 완주 — 문서 **${ndocs}편** · 실패 ${fails}건
산출 \`outputs/eacl_phase0/phase1_chain_qwen_v31/\`. 🔴 판정은 하지 않는다 — 아침에 확인."
say "완주 문서=$ndocs 실패=$fails · vLLM 종료"
for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
