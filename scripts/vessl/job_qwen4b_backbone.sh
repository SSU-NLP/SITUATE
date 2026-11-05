#!/usr/bin/env bash
set -euo pipefail
CODE="${CODE:?CODE must point to uploaded experiment code}"
OUT="${OUT:?OUT must be a new stage output directory}"
STAGE="${STAGE:?STAGE must be embed or train}"
SHARED=/shared/hiontop/code
W=/tmp/hiontop-qwen4b
D=outputs/experiments/m4m5_phase1_anchor/data
mkdir -p "$OUT" "$W/$D" "$W/data/eacl_phase1" "$W/source" "$W/test_raw"
exec > >(tee "$OUT/job.log") 2>&1
cp -r "$CODE/src" "$CODE/scripts" "$W/"
cp "$SHARED/pg_shorts_personkey_cap20_scant.jsonl" "$W/data/eacl_phase1/"
for split in phase1__personkey_i2 dev3 test3; do
  cp "$SHARED/$D/${split}_index.npz" "$W/source/"
done
ln -s "$SHARED/benchmarks" "$W/benchmarks"
cd "$W"
if [ "$STAGE" = embed ]; then
  for name in run gohitler mymotherslife; do
    f="/legacy/Hi-OnTop/outputs/eacl_phase0/${name}_eventrecall_goldtopn_master.jsonl"
    [ -s "$f" ] || { echo "Missing test source: $f"; exit 1; }
    cp "$f" "$W/test_raw/"
  done
  python -m pip install 'transformers==4.56.2' 'sentence-transformers==5.1.1' nltk scikit-learn
  MODE=--embed-only
elif [ "$STAGE" = train ]; then
  EMB="${EMB:?EMB must point to completed embedding output}"
  for split in phase1__personkey_i2 dev3 test3; do
    cp "$EMB/${split}_emb.npy" "$EMB/${split}_index.npz" "$W/$D/"
  done
  python -m pip install nltk scikit-learn
  MODE=--train-only
else
  echo "Invalid STAGE: $STAGE"; exit 1
fi
python -u scripts/eacl_phase0/run_qwen4b_backbone.py \
  --source "$W/source" \
  --raw "$W/data/eacl_phase1/pg_shorts_personkey_cap20_scant.jsonl" \
  --dev-source /shared/hiontop/data/dev3 --test-source "$W/test_raw" \
  --persist "$OUT" --seed 0 "$MODE"
