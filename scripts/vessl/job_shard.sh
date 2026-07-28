#!/usr/bin/env bash
# VESSL Job 1개 = A100 1장 = vLLM 복제 1개 + 자기 몫 (샤드 × 축) 소비.
#
# Job 안에서 실행된다. 코드·데이터·모델은 전부 /shared (objvol-gsvyr0eu87wt) 에서 읽는다.
# private repo 인증을 피하려고 git clone 대신 스테이징된 사본을 쓴다 (capregion Job 관례).
#
# 실측 근거 (2026-07-27, 동일 A100 SXM 80GB):
#   TP=1 가중치 33.89 GiB · KV 29.99 GiB(1,283,125 토큰) · 기동 ~200s
#   MoE 활성 3B 라 TP 를 키우면 통신이 지배 → 복제를 늘리는 쪽이 선형
#
# 필수 env: SHARD_ID, NUM_SHARDS
set -euo pipefail

: "${SHARD_ID:?}"; : "${NUM_SHARDS:?}"

STAGE="${STAGE:-/shared/hiontop}"
MODEL="${MODEL:-/shared/models/Qwen3.6-35B-A3B}"   # bf16 원본. fp8 은 아래 실행 옵션으로.
SERVED="${SERVED:-qwen36-35b}"
PORT="${PORT:-8001}"
OUT_DIR="${OUT_DIR:?OUT_DIR 필요 — 예: /shared/hiontop/out/<run-name>}"
SHARD_DIR="${SHARD_DIR:-$STAGE/data/shards}"
WORKLIST="${WORKLIST:-$STAGE/worklist.txt}"
CONC="${CONC:-130}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"   # 내리지 말 것 — 16384 면 278편이 컨텍스트 초과 탈락
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
BATCH="${BATCH:-20}"

cd "$STAGE/code"
LOGD="$OUT_DIR/_logs/shard${SHARD_ID}"; mkdir -p "$LOGD"

echo "[shard $SHARD_ID/$NUM_SHARDS] model=$MODEL conc=$CONC out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── 0. 의존성 ────────────────────────────────────────────────────────
# flashinfer 샘플러가 JIT 빌드 때 ninja 를 exec 로 찾는다. PATH 에 없으면 기동이 죽는다.
python -c "import vllm" 2>/dev/null || pip install -q "vllm==0.25.1"
command -v ninja >/dev/null || pip install -q ninja
export PATH="$(python -c 'import sys,os;print(os.path.dirname(sys.executable))'):$PATH"
command -v ninja >/dev/null || { echo "[shard $SHARD_ID] ninja 여전히 없음 — 기동 실패 예상"; }

# ── 1. vLLM 기동 (로컬과 동일 설정) ──────────────────────────────────
VLLM_LOG="$LOGD/vllm.log"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$SERVED" \
  --quantization fp8 --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs 512 --reasoning-parser qwen3 \
  --no-enable-log-requests --enable-prefix-caching \
  --port "$PORT" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

# ── 2. health 대기. /shared 는 object storage FUSE 라 로컬(60s)보다 로드가 느릴 수 있다 ──
for _ in $(seq 1 360); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "[shard $SHARD_ID] vLLM 죽음"; tail -60 "$VLLM_LOG"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
  || { echo "[shard $SHARD_ID] health timeout (30분)"; tail -60 "$VLLM_LOG"; exit 1; }
grep -oE "Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds|GPU KV cache size: [0-9,]+ tokens" "$VLLM_LOG" | tail -2
echo "[shard $SHARD_ID] vLLM ready"

# ── 3. 자기 몫만 처리 ────────────────────────────────────────────────
# worklist: "<shard 파일명> <axis>" 2열. 배분 = 라인 round-robin.
# 20개 Job 이 같은 OUT_DIR 에 쓴다 — (문서×축) 파일이 서로 겹치지 않아 안전
# (로컬 run_label_2rep.sh 가 2복제로 같은 방식 검증됨).
SLICE="$LOGD/slice.txt"
grep -vE '^\s*(#|$)' "$WORKLIST" | awk -v s="$SHARD_ID" -v n="$NUM_SHARDS" '(NR-1) % n == s' > "$SLICE"
echo "[shard $SHARD_ID] task $(wc -l < "$SLICE")건"
[ -s "$SLICE" ] || { echo "[shard $SHARD_ID] 할당 없음 — 정상 종료"; exit 0; }

export URL="http://127.0.0.1:${PORT}/v1" OUT_DIR SHARD_DIR BATCH LOGD
xargs -a "$SLICE" -P "$CONC" -n 2 bash -c '
  set -uo pipefail
  bash scripts/eacl_phase1/resilient_shard.sh \
    "$SHARD_DIR/$0" "$1" "$OUT_DIR" "$BATCH" "$URL" >> "$LOGD/client.log" 2>&1 \
    || echo "[FAIL] $0:$1"
' || true

echo "[shard $SHARD_ID] 완료 $(date -u +%H:%M:%S)"
