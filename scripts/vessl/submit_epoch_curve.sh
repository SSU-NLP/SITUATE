#!/usr/bin/env bash
# epoch 1~N 성능 곡선 Job 제출 — seed 당 Job 1개 (A100 1장).
#
#   DRY_RUN=1 bash scripts/vessl/submit_epoch_curve.sh      # 명령만 출력
#   SEEDS="0" bash scripts/vessl/submit_epoch_curve.sh       # seed 0 만
#   SEEDS="0 1 2 3" bash scripts/vessl/submit_epoch_curve.sh
#
# 코드는 먼저 올린다(데이터는 /shared/hiontop/code 기존 스테이징 재사용):
#   UPLOAD=1 bash scripts/vessl/submit_epoch_curve.sh
# 결과: /shared/hiontop/out/epoch_curve_e${EPOCHS}/seed{s}/{epoch_curve.json,job.log,*.pt}
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-10}"
DRY_RUN="${DRY_RUN:-0}"
UPLOAD="${UPLOAD:-0}"
SPEC="${SPEC:-resourcespec-a100x1}"
VOL="${VOL:-objvol-gsvyr0eu87wt}"      # ssu-volume → /shared
IMAGE="${IMAGE:-quay.io/vessl-ai/torch@sha256:8bd418d49eeed841a607582b614c9d3ba86590b07d1be671816176c47253344a}"
PREFIX="${PREFIX:-hiontop/epc1/code}"

if [ "$UPLOAD" = "1" ]; then
  # 필요한 코드만 스테이징 디렉터리로 모아 한 번에 올린다(.pyc 제외).
  STG="$(mktemp -d)"
  mkdir -p "$STG/scripts/eacl_phase0" "$STG/scripts/eacl_phase1" "$STG/scripts/vessl"
  cp scripts/eacl_phase0/{train_anchor_m4m5,eval_all_models_alldocs,train_direct_boundary,epoch_curve}.py \
     "$STG/scripts/eacl_phase0/"
  cp scripts/eacl_phase1/author_review_list.py "$STG/scripts/eacl_phase1/"
  cp scripts/vessl/job_epoch_curve.sh "$STG/scripts/vessl/"
  cp -r src "$STG/"
  find "$STG" \( -name "*.pyc" -o -name __pycache__ \) -prune -exec rm -rf {} +
  vesslctl volume upload "$VOL" "$STG" --remote-prefix "$PREFIX/" --overwrite ${UPLOAD_DRY:+--dry-run}
  rm -rf "$STG"
  exit 0
fi

for s in $SEEDS; do
  cmd=(vesslctl job create
       -n "hiontop-epc1-e${EPOCHS}-s${s}"
       -r "$SPEC" -i "$IMAGE"
       --object-volume "${VOL}:/shared"
       -e "SEED=$s" -e "EPOCHS=$EPOCHS" -e "CODE=/shared/$PREFIX"
       --cmd "bash /shared/$PREFIX/scripts/vessl/job_epoch_curve.sh")
  if [ "$DRY_RUN" = "1" ]; then printf '%q ' "${cmd[@]}"; echo
  else "${cmd[@]}"
  fi
done
echo "비용 ≈ \$1.55/hr × $(echo $SEEDS | wc -w) · 상태: vesslctl job list · 로그: vesslctl job logs <slug>"
