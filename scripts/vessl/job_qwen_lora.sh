#!/usr/bin/env bash
set -euo pipefail
CODE="${CODE:?}"
OUT="${OUT:?}"
W=/tmp/hiontop-lora
SHARED=/shared/hiontop/code
D=outputs/experiments/m4m5_phase1_anchor/data
mkdir -p "$OUT" "$W/source" "$W/data/eacl_phase1" "$W/test_raw"
exec > >(tee "$OUT/job.log") 2>&1
cp -r "$CODE/src" "$CODE/scripts" "$W/"
cp "$SHARED/pg_shorts_personkey_cap20_scant.jsonl" "$W/data/eacl_phase1/"
for split in phase1__personkey_i2 dev3 test3; do
  cp "$SHARED/$D/${split}_index.npz" "$W/source/"
done
for name in run gohitler mymotherslife; do
  cp "/legacy/Hi-OnTop/outputs/eacl_phase0/${name}_eventrecall_goldtopn_master.jsonl" "$W/test_raw/"
done
ln -s "$SHARED/benchmarks" "$W/benchmarks"
cd "$W"
python -m pip install 'transformers==4.56.2' 'peft==0.17.1' sentencepiece nltk scikit-learn
python -m pip freeze > "$OUT/packages.txt"
python scripts/eacl_phase0/train_qwen_lora.py --self-test
REVISION=$(python - <<'PY'
import json
p='/shared/hiontop/qwen17b-20260925/embeddings/embedding_manifest.json'
m=json.load(open(p))
print(m.get('revision') or m['shards'][0]['revision'])
PY
)
python -u scripts/eacl_phase0/train_qwen_lora.py \
  --source "$W/source" --raw "$W/data/eacl_phase1/pg_shorts_personkey_cap20_scant.jsonl" \
  --dev-source /shared/hiontop/data/dev3 --test-source "$W/test_raw" \
  --out "$OUT" --seed 1 --revision "$REVISION"
