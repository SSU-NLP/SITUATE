#!/usr/bin/env bash
set -euo pipefail
: "${PACKAGE:?}" "${PACKAGE_SHA256:?}" "${OUT:?}" "${SHARD:?}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1
WORK=$(mktemp -d /tmp/stage-tracking.XXXXXX)
echo "$PACKAGE_SHA256  $PACKAGE" | sha256sum -c -
tar -xzf "$PACKAGE" -C "$WORK"
cd "$WORK"
SERVER_PID=""
cleanup() {
  code=$?
  if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi
  python - "$OUT/status.json" "$code" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'exit_code':int(sys.argv[2]),'finished_at':time.time()}))
PY
}
trap cleanup EXIT

# Keep the detector's software identical to the recent LoRA run.
python -m pip install 'transformers==4.56.2' 'peft==0.17.1' 'tokenizers==0.22.2' \
  'nltk==3.10.3' sentencepiece pytest
python -m pip freeze > "$OUT/detector_packages.txt"
python -m pytest -q tests/test_stage_axis_predictor.py tests/test_stage_state_edits.py
python scripts/stage/verify_checkpoint.py --artifacts artifacts --output "$OUT/checkpoint_verification.json"
python scripts/stage/make_config.py --template evaluation/configs/stage_release_evaluation.example.json \
  --tokenizer tokenizer.json --output "$OUT/runtime_config.json"
python scripts/stage/check_contracts.py --evaluation evaluation --inputs inputs \
  --config "$OUT/runtime_config.json" --output "$OUT/contract_checks.json"
python scripts/stage/run_tracking.py --evaluation evaluation --inputs inputs \
  --config "$OUT/runtime_config.json" --axes "$OUT/axes" --output "$OUT/predictions" \
  --shard "$SHARD" --num-shards 2 --preflight-only
python -c "import nltk; nltk.download('punkt_tab', raise_on_error=True)"
python scripts/stage/extract_axes.py --inputs inputs --artifacts artifacts \
  --output "$OUT/axes" --shard "$SHARD" --num-shards 2

# Serving uses an isolated environment so its dependencies cannot alter the detector.
python -m venv --system-site-packages "$WORK/serving"
PY="$WORK/serving/bin/python"
"$PY" -m pip install 'vllm==0.25.1' 'json-repair>=0.59.5' jieba rank-bm25 openai ninja
export PATH="$WORK/serving/bin:$PATH"
"$PY" -m pip freeze > "$OUT/serving_packages.txt"

start_server() {
  model="$1"
  label="$2"
  revision=$("$PY" - "$model" <<'PY'
import json,sys
print(json.load(open('model_revisions.json'))[sys.argv[1]])
PY
)
  "$PY" -m vllm.entrypoints.openai.api_server --model "$model" --revision "$revision" \
    --served-model-name "$model" --tensor-parallel-size 2 --dtype bfloat16 \
    --max-model-len 24000 --gpu-memory-utilization 0.88 --max-num-seqs 64 \
    --max-num-batched-tokens 8192 --enforce-eager --generation-config vllm \
    --reasoning-parser qwen3 --seed 1 --port 8001 > "$OUT/server_${label}.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 900); do
    curl -sf http://127.0.0.1:8001/health >/dev/null && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -60 "$OUT/server_${label}.log"; return 1; }
    sleep 2
  done
  return 1
}
stop_server() {
  kill "$SERVER_PID"
  wait "$SERVER_PID" || true
  SERVER_PID=""
  sleep 5
}
predict() {
  "$PY" scripts/stage/run_tracking.py --evaluation evaluation --inputs inputs \
    --config "$OUT/runtime_config.json" --axes "$OUT/axes" --output "$OUT/predictions" \
    --shard "$SHARD" --num-shards 2 "$@"
}
evaluate() {
  "$PY" scripts/stage/run_evaluation.py --evaluation evaluation --inputs inputs \
    --config "$OUT/runtime_config.json" --predictions "$OUT/predictions" --output "$OUT/evaluation" \
    --shard "$SHARD" --num-shards 2 "$@"
}

# Validate the entire generation-to-judge path before expanding to the shard.
# These fixed first movies are retained in the final run; outputs are resumed.
start_server Qwen/Qwen3-8B actor_smoke
predict --limit 1
stop_server
start_server Qwen/Qwen3-32B judge_smoke
evaluate --limit 1
stop_server
start_server Qwen/Qwen3-8B actor_full
predict
stop_server
start_server Qwen/Qwen3-32B judge_full
evaluate
stop_server
"$PY" scripts/stage/summarize.py --inputs inputs --evaluation "$OUT/evaluation" \
  --output "$OUT/summary.json" --shard "$SHARD" --num-shards 2
