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
HEALTH_MIN="${HEALTH_MIN:-75}"            # health 대기(분). /shared 경합 시 로딩이 30분을 넘긴다
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"   # 내리지 말 것 — 16384 면 278편이 컨텍스트 초과 탈락
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
BATCH="${BATCH:-20}"

cd "$STAGE/code"
LOGD="$OUT_DIR/_logs/shard${SHARD_ID}"; mkdir -p "$LOGD"

echo "[shard $SHARD_ID/$NUM_SHARDS] model=$MODEL conc=$CONC out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── 0-a. 자기 몫 확정 + preflight ────────────────────────────────────
# vLLM 기동(30~57분) **전에** 한다. 2026-08-03: worklist 1열이 확장자 없는 `shard032`
# 인데 `$SHARD_DIR/$0` 를 그대로 붙여 cp 가 전부 실패했고, llm_s5_states 의 read() 가
# 없는 파일에 [] 를 반환해 exit 0 → 전 태스크 `[ok]`, 산출물 0. 5회차 × A100 4장을
# 그렇게 태웠다. 파일이 없으면 여기서 즉시 죽는다.
# worklist: "<샤드> <axis>" 2열. 배분 = 라인 round-robin.
SLICE="$LOGD/slice.txt"
grep -vE '^\s*(#|$)' "$WORKLIST" | awk -v s="$SHARD_ID" -v n="$NUM_SHARDS" '(NR-1) % n == s' > "$SLICE"
echo "[shard $SHARD_ID] task $(wc -l < "$SLICE")건"
[ -s "$SLICE" ] || { echo "[shard $SHARD_ID] 할당 없음 — 정상 종료"; exit 0; }

# 샤드 이름 → 실제 파일 경로. 확장자 유무 양쪽을 받는다.
resolve_shard(){
  for p in "$SHARD_DIR/$1" "$SHARD_DIR/$1.jsonl"; do [ -s "$p" ] && { echo "$p"; return 0; }; done
  return 1
}
MISSING=0
while read -r name _axis; do
  resolve_shard "$name" >/dev/null || { echo "[shard $SHARD_ID] 샤드 없음: $name"; MISSING=$((MISSING+1)); }
done < <(sort -u -k1,1 "$SLICE")
[ "$MISSING" -eq 0 ] || {
  echo "[shard $SHARD_ID] preflight 실패 — 샤드 ${MISSING}종이 $SHARD_DIR 에 없다. GPU 안 켜고 중단"
  ls "$SHARD_DIR" | head -5; exit 1
}
echo "[shard $SHARD_ID] preflight ok — 샤드 $(sort -u -k1,1 "$SLICE" | wc -l)종 확인"

# ── 0. 의존성 ────────────────────────────────────────────────────────
# flashinfer 샘플러가 JIT 빌드 때 ninja 를 exec 로 찾는다. PATH 에 없으면 기동이 죽는다.
python -c "import vllm" 2>/dev/null || pip install -q "vllm==0.25.1"
command -v ninja >/dev/null || pip install -q ninja
python -c "import dotenv" 2>/dev/null || pip install -q python-dotenv   # llm_s5_states 가 임포트한다
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
for _ in $(seq 1 $((HEALTH_MIN*12))); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "[shard $SHARD_ID] vLLM 죽음"; tail -60 "$VLLM_LOG"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
  || { echo "[shard $SHARD_ID] health timeout (${HEALTH_MIN}분)"; tail -60 "$VLLM_LOG"; exit 1; }
grep -oE "Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds|GPU KV cache size: [0-9,]+ tokens" "$VLLM_LOG" | tail -2
echo "[shard $SHARD_ID] vLLM ready"

# ── 3. Stage-1 예열 ──────────────────────────────────────────────────
# story_lists.json 은 **축 간 공유 캐시**다. 예열 없이 5축을 동시에 던지면 같은 문서의
# 5개 축이 Stage-1 을 중복 호출한다(로컬은 2026-08-03 10:20 재구성 때 넣었고 Job 경로엔
# 빠져 있었다). 자기 슬라이스의 샤드만 모아 먼저 채운다.
export REPO_ROOT="$STAGE/code" PY="$(command -v python3 || command -v python)"
MASTER="$LOGD/prewarm_master.jsonl"; : > "$MASTER"
while read -r name; do cat "$(resolve_shard "$name")" >> "$MASTER"; done \
  < <(cut -d' ' -f1 "$SLICE" | sort -u)
"$PY" scripts/eacl_phase1/prewarm_stage1.py "$MASTER" "$OUT_DIR" \
    --ports "$PORT" --workers "$CONC" >> "$LOGD/prewarm.log" 2>&1 \
  && echo "[shard $SHARD_ID] prewarm ok" \
  || echo "[shard $SHARD_ID] prewarm 일부 실패 — 축 단계에서 개별 재시도된다"

# ── 4. 자기 몫만 처리 ────────────────────────────────────────────────
# 여러 Job 이 같은 OUT_DIR 에 쓴다 — (문서×축) 파일이 서로 겹치지 않아 안전
# (로컬 run_label_2rep.sh 가 2복제로 같은 방식 검증됨).
# resilient_shard.sh 는 기본이 /root/Hi-OnTop 이다 — 컨테이너 경로를 넘겨야 돈다
export URL="http://127.0.0.1:${PORT}/v1" OUT_DIR SHARD_DIR BATCH LOGD
while read -r name axis; do printf '%s\t%s\n' "$(resolve_shard "$name")" "$axis"; done < "$SLICE" \
| xargs -d '\n' -I{} -P "$CONC" bash -c '
  set -uo pipefail
  IFS=$'"'"'\t'"'"' read -r shard axis <<< "$1"
  bash scripts/eacl_phase1/resilient_shard.sh \
    "$shard" "$axis" "$OUT_DIR" "$BATCH" "$URL" >> "$LOGD/client.log" 2>&1 \
    || echo "[FAIL] $shard:$axis"
' _ {} || true

# ── 5. postflight — 빈 성공을 성공으로 보고하지 않는다 ───────────────
# 이게 없어서 Job 이 5회 연속 `succeeded` 로 뜨면서 산출물이 0이었다.
NDOC=$(find "$OUT_DIR" -maxdepth 1 -type d -name 'pg*' 2>/dev/null | wc -l)
echo "[shard $SHARD_ID] 완료 $(date -u +%H:%M:%S) · OUT_DIR 문서 디렉터리 ${NDOC}개"
[ "$NDOC" -gt 0 ] || { echo "[shard $SHARD_ID] 🔴 산출물 0 — 실패로 보고한다"; exit 1; }
