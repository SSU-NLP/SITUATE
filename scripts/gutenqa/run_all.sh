#!/usr/bin/env bash
# GutenQA 다운스트림 검색 평가 (D-B39) — 전 단계 재현 실행.
# 단계마다 산출물이 있으면 건너뛴다. 빈 출력 디렉토리에서 처음부터 완주 가능.
#
#   bash scripts/gutenqa/run_all.sh
#
# 🔴 Gate 0(00/00b) 은 오염·부분집합을 **동결**한다. 이미 동결된 뒤 다시 돌려도
#    같은 결과가 나와야 한다(MANUAL_TITLE_MAP·SPLIT_SEED 가 코드에 박혀 있음).
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=".venv/bin/python"
OUT="outputs/experiments/2026-07-29_gutenqa_chunking"
mkdir -p "$OUT/logs"

echo "== Gate 0: 오염 체크 =="
[ -f "$OUT/contamination.json" ] || $PY scripts/gutenqa/00_contamination_check.py
[ -f "$OUT/book_subsets.json" ]  || $PY scripts/gutenqa/00b_book_subsets.py

echo "== Phase 1: 데이터 수집 =="
[ -f "$OUT/data/questions.jsonl" ] || $PY scripts/gutenqa/01_fetch.py

echo "== Phase 1: 베이스라인 5종 (재현 게이트) =="
for m in lumberchunker recursive paragraph semantic propositions; do
  if [ -f "$OUT/retrieval/$m.json" ]; then echo "  skip $m"; continue; fi
  $PY scripts/gutenqa/02_eval_harness.py --method "$m" 2>&1 | tee "$OUT/logs/eval_$m.log"
done

echo "== Phase 2: M5 분절 =="
[ -f "$OUT/m5_chunk_stats.json" ] || \
  $PY scripts/gutenqa/03_segment_m5.py --rates 0.05 0.10 0.15 0.20 0.30 0.50 --frozen \
    2>&1 | tee "$OUT/logs/segment_m5.log"

echo "== Phase 2: M5 평가 =="
for f in "$OUT"/data/chunks/m5__*.jsonl; do
  n="$(basename "$f" .jsonl)"
  if [ -f "$OUT/retrieval/$n.json" ]; then echo "  skip $n"; continue; fi
  $PY scripts/gutenqa/02_eval_harness.py --chunks "$f" --name "$n" 2>&1 | tee "$OUT/logs/eval_$n.log"
done

echo "== Phase 3: 집계 =="
$PY scripts/gutenqa/04_aggregate.py
echo "완료 -> $OUT/SUMMARY_TABLES.md"
