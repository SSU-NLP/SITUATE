#!/usr/bin/env bash
# M4 학습을 VESSL 로 보내기 위한 스테이징 — 코드 + I2 앵커 데이터.
# 임베딩 4.65GB 가 대부분의 시간을 먹는다. I2 인덱스는 임베딩을 심볼릭으로 재사용하므로
# **원본 파일**을 올려 i2 이름으로 둔다(Job 안에서는 심볼릭이 안 풀린다).
set -euo pipefail
cd /root/Hi-OnTop
S=/shared/hiontop/code
D=outputs/experiments/m4m5_phase1_anchor/data
mkdir -p "$S/scripts/eacl_phase0" "$S/scripts/eacl_phase1" "$S/src" "$S/$D"
cp scripts/eacl_phase0/train_anchor_m4m5.py "$S/scripts/eacl_phase0/"
cp scripts/eacl_phase1/author_review_list.py "$S/scripts/eacl_phase1/"
cp -r src/hi_ontop "$S/src/"
# 🔴 손상된 .pyc 가 컨테이너에서 "marshal data too short" 로 즉사시킨다(2026-08-04 실측)
find "$S/src" -name "*.pyc" -delete; find "$S/src" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
cp data/eacl_phase1/pg_shorts_personkey_cap20_scant.jsonl "$S/"
for f in dev3_emb.npy dev3_index.npz test3_emb.npy test3_index.npz phase1__personkey_i2_index.npz; do
  cp "$D/$f" "$S/$D/"; done
# 임베딩은 심볼릭 원본을 i2 이름으로 (4.65GB — 가장 오래 걸린다)
cp "$(readlink -f $D/phase1__personkey_i2_emb.npy)" "$S/$D/phase1__personkey_i2_emb.npy"
echo "스테이징 완료: $(du -sh $S/$D | cut -f1)"
