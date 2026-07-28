#!/usr/bin/env bash
# M5 Phase 2 — 학습 코퍼스 ablation 큐 (fiction / share / both + 통제군).
# 파이프·timeout 금지 (버퍼링/조기 kill). python -u 로 스트리밍.
#   usage: bash scripts/eacl_phase0/run_corpus_ablation.sh
set -u
cd "$(git rev-parse --show-toplevel)"
D=outputs/experiments/m5_phase2_corpus_ablation
mkdir -p "$D"
S=2                       # 시드 (사용자 결정: 축소 실행)
COMMON="--variant weak --seeds $S --pos-weight-from fiction --dump $D/sweep.jsonl"

run () {  # run <tag> <extra args...>
  local tag=$1; shift
  echo "############ $tag ############"
  uv run python -u scripts/eacl_phase0/slot_seed_sweep.py --tag "$tag" $COMMON "$@" \
    || echo "!!!! $tag FAILED (exit $?)"
}

# [0] 재현성 안전망 — 리팩터가 기존 Phase 2 수치를 깨지 않았나 (기존 설정: full-doc, 150 epochs)
run fiction_repro --dataset slot_dataset_all.npz --train-corpus fiction --chunk-len 0 --epochs 150

# [1] 주 실험 — 3조건. chunk_len/max_steps 로 업데이트 수 고정 (C3), pos_weight fiction 고정 (C7)
for C in fiction share both; do
  run "$C" --dataset slot_dataset_all.npz --train-corpus "$C" --chunk-len 128 --max-steps 3000
done

# [2] 통제군
#  C2 크기 교란: both 를 fiction 전이 수(3090)로 서브샘플
run both_matched --dataset slot_dataset_all.npz --train-corpus both --chunk-len 128 --max-steps 3000 \
    --subsample-trans 3090
#  C5 gold-vs-silver: SHARE place/time 을 M3 silver 로
run both_silver --dataset slot_dataset_silver.npz --train-corpus both --chunk-len 128 --max-steps 3000
#  C9 protagonist 상수라벨 노이즈: protagonist 축 손실 제외
run share_maskproto --dataset slot_dataset_all.npz --train-corpus share --chunk-len 128 --max-steps 3000 \
    --axis-mask 01111

echo "############ DONE ############"
