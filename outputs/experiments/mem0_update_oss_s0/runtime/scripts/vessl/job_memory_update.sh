#!/usr/bin/env bash
set -euo pipefail
: "${CODE:?missing immutable code prefix}"
: "${OUT:?missing output directory}"
: "${ARM:?missing arm}"
[[ "$ARM" == stateful || "$ARM" == stateless ]]
WORK="$(mktemp -d /tmp/mem0-update.XXXXXX)"
mkdir -p "$OUT" "$WORK/output"
exec > >(tee "$OUT/job.log") 2>&1
finish() {
  local status=$?
  if [[ -d "$WORK/output/models" ]]; then cp -r "$WORK/output/models" "$OUT/"; fi
  exit "$status"
}
trap finish EXIT
cp -r "$CODE/src" "$CODE/scripts" "$CODE/configs" "$WORK/"
cp "$CODE/training.pt" "$WORK/output/training.pt"
cd "$WORK"
export PYTHONPATH="$WORK/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
python - <<'PY'
import torch, json
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print(json.dumps({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}), flush=True)
PY
seed_list="$(python -c 'import json; print(*json.load(open("configs/mem0_update_v1.json"))["seeds"])')"
for seed in $seed_list; do
  python scripts/mem0_update/run.py train --config configs/mem0_update_v1.json \
    --output "$WORK/output" --arm "$ARM" --seed "$seed" --device cuda:0
  cp -r "$WORK/output/models" "$OUT/"
done
python - <<'PY'
import json, os
from pathlib import Path
seeds = json.loads(Path('configs/mem0_update_v1.json').read_text())['seeds']
Path(os.environ['OUT'], 'complete.json').write_text(json.dumps({'arm': os.environ['ARM'], 'seeds': seeds}))
PY
