#!/usr/bin/env bash
# Final corpus-scale run: data-scale ablation (frac) x {train, gate1, gate2 held-out}.
# Answers doc §9: does MORE data fix the leakage / beat surprise? Run after build_corpus done.
set -e
cd "$(dirname "$0")/../.."
export HF_DATASETS_CACHE=/tmp/claude-0/-root/d13a28d4-7d56-4cb2-9443-281710e6c27a/scratchpad/hf_datasets

for FRAC in 0.1 0.3 1.0; do
  TAG="corpus_f${FRAC}"
  echo "############### FRAC=$FRAC (tag=$TAG) ###############"
  uv run python scripts/idm_lam/train_corpus.py --frac "$FRAC" --epochs 10 --batch 8192 --tag "$TAG" 2>&1 | grep -E "corpus|ep|DONE"
  uv run python scripts/idm_lam/gate1_heldout.py --tag "$TAG" 2>&1 | grep -vE "Batches|Download|Fetch|warn" | sed -n '/Gate-1/,$p'
  uv run python scripts/idm_lam/gate2_heldout.py --tag "$TAG" 2>&1 | grep -vE "Batches|Download|Fetch|warn" | sed -n '/=====/,$p'
done
echo "ALL DONE"
