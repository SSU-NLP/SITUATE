#!/usr/bin/env bash
# VESSL Job 내부 — M5 를 논문 설정으로 epoch 1~10 학습하며 매 epoch 스냅숏을 남기고,
# 스냅숏마다 6편(dev-3 + test3) Score 를 online_nf c=1.0 (논문 작동점)으로 채점한다.
#
# 코드는 $CODE (로컬에서 `vesslctl volume upload` 로 올린 사본), 데이터·벤치마크는 기존 스테이징
# `/shared/hiontop/code` 를 심볼릭으로 붙인다. 산출은 컨테이너 로컬에 쓰고 끝에 $OUT 으로 복사한다.
#   env: SEED (기본 0) · EPOCHS (기본 10) · CODE · OUT
set -euo pipefail
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-10}"
CODE="${CODE:-/shared/hiontop/epc1/code}"
SHARED=/shared/hiontop/code
OUT="${OUT:-/shared/hiontop/out/epoch_curve_e${EPOCHS}}/seed${SEED}"
W=/tmp/hiontop
D=outputs/experiments/m4m5_phase1_anchor/data
mkdir -p "$OUT"
exec > >(tee "$OUT/job.log") 2>&1

echo "$(date -Is) seed=$SEED epochs=$EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# ── 작업 트리 조립 ────────────────────────────────────────────────────────────
mkdir -p "$W/outputs/experiments/m4m5_phase1_anchor" "$W/data/eacl_phase1"
cp -r "$CODE/scripts" "$CODE/src" "$W/"
find "$W" -name "*.pyc" -delete
# 🔴 데이터는 로컬 디스크로 **복사**한다. 오브젝트 스토리지 위 mmap 은 배치마다 랜덤 읽기라
#    epoch 1 도 12분 넘게 못 끝냈다(2026-09-25 실측). 순차 복사는 한 번이면 된다.
mkdir -p "$W/$D"
for f in dev3_emb.npy dev3_index.npz test3_emb.npy test3_index.npz \
         phase1__personkey_i2_emb.npy phase1__personkey_i2_index.npz; do
  cp "$SHARED/$D/$f" "$W/$D/"
done
echo "$(date -Is) 데이터 로컬 복사 완료: $(du -sh "$W/$D" | cut -f1)"
ln -sfn "$SHARED/benchmarks" "$W/benchmarks"
ln -sfn "$SHARED/pg_shorts_personkey_cap20_scant.jsonl" \
  "$W/data/eacl_phase1/pg_shorts_personkey_cap20_scant.jsonl"
cd "$W"

# ── 사전검사 — 하나라도 없으면 GPU 시간을 쓰기 전에 죽는다 ──────────────────
for f in dev3_emb.npy dev3_index.npz test3_emb.npy test3_index.npz \
         phase1__personkey_i2_emb.npy phase1__personkey_i2_index.npz; do
  [ -s "$D/$f" ] || { echo "🔴 데이터 없음: $D/$f"; exit 1; }
done
[ -d benchmarks/superdialseg/src/super_dialseg ] || { echo "🔴 superdialseg 없음"; exit 1; }
python -m pip install --no-cache-dir -q nltk scikit-learn 2>&1 | tail -2 || true
python - <<'EOF'
import sys
sys.path[:0] = ["src", "scripts/eacl_phase0"]
import train_anchor_m4m5, eval_all_models_alldocs  # noqa: F401
from hi_ontop.dts_scoring import score_dialogues
print("import OK · 채점기", score_dialogues([[0, 1, 0, 0]], [[0, 1, 0, 0]])["score"])
EOF

# ── 학습 — 논문 설정 그대로, 조기종료만 끄고 매 epoch 스냅숏 ─────────────────
python scripts/eacl_phase0/train_anchor_m4m5.py \
  --split phase1__personkey_i2 --models M5 --seed "$SEED" \
  --epochs "$EPOCHS" --patience 0 --ckpt-every 1 --log-every 1 \
  --group-split author --author-map data/eacl_phase1/pg_shorts_personkey_cap20_scant.jsonl \
  --ns-suffix epc1

# ── 채점 — 스냅숏 전부 + 마지막 epoch 복원본(=999) ───────────────────────────
python scripts/eacl_phase0/epoch_curve.py --models M5 --device cuda:0 \
  --pattern "{model}_phase1__personkey_i2_per_slot_gsauthor_ep${EPOCHS}_epc1*seed${SEED}.pt" \
  --out "$OUT"

cp outputs/experiments/m4m5_phase1_anchor/*epc1*seed${SEED}* "$OUT/" 2>/dev/null || true
echo "$(date -Is) DONE"
