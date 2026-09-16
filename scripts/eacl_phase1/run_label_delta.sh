#!/bin/bash
# 델타 라벨링 — 로컬 vLLM 복제에 (샤드×축) 작업을 분산한다.
#
# 구조는 `EACL/phase0-annotation-guide.md` §③ 실측을 따른다.
#   · **동시 요청 수 = 병렬로 도는 (샤드×축) 작업 수.** `llm_s5_states.py` 는 요청을 하나씩만
#     보내므로 xargs -P 를 올려도 작업 수가 적으면 GPU 가 논다. 샤드를 잘게 쪼개야 한다.
#   · 그래서 5축을 **한꺼번에** 돌린다. Phase A/B 로 나누면 A 구간 병렬도가 1/5 로 떨어진다.
#   · Stage-1(`story_lists.json`)은 축 간 공유 캐시라 `prewarm_stage1.py` 로 먼저 채운다.
#     안 하면 같은 문서의 5축이 동시에 Stage-1 을 중복 호출한다.
#
# 왜 `run_label_resilient.sh` 를 안 쓰나: 그쪽은 `resilient_shard.sh` 를 3인자로만 불러
# BASEURL 이 8001 로 고정된다 → 복제 1개만 쓰고 GPU 3장이 논다.
#
# 전부 resume-safe: {axis}_states.jsonl 캐시가 남아 이미 끝난 문서는 즉시 건너뛴다.
set -u
cd /root/Hi-OnTop
SHARD_DIR="${1:?usage: run_label_delta.sh <shard dir> <out> [nrep] [P]}"
OUT="${2:?}"
NREP="${3:-4}"
P="${4:-192}"
PY=/root/Hi-OnTop/.venv/bin/python
mkdir -p "$OUT"

echo "=== DELTA LABEL start $(date +%H:%M:%S) · 복제 $NREP · 동시 $P ==="

for _ in $(seq 1 180); do
  n=0
  for g in $(seq 0 $((NREP-1))); do
    curl -sf -m 30 "http://127.0.0.1:$((8001+g))/v1/models" >/dev/null 2>&1 && n=$((n+1))
  done
  [ "$n" -eq "$NREP" ] && break
  sleep 20
done
[ "${n:-0}" -eq "$NREP" ] || { echo "[FAIL] 복제 ${n:-0}/$NREP 만 준비됨 — 중단"; exit 1; }
echo "복제 $NREP 개 준비 완료 $(date +%H:%M:%S)"

# ── Stage-1 예열 ─────────────────────────────────────────────────────────────
# 축 간 공유 캐시라 먼저 채운다. 복제 수만큼 포트를 넘겨 분산한다.
PORTS=$(seq -s, 8001 $((8000+NREP)))
MASTER="$OUT/_prewarm_master.jsonl"
cat "$SHARD_DIR"/*.jsonl > "$MASTER"
echo "--- Stage-1 예열 (ports $PORTS · workers $P) ---"
$PY scripts/eacl_phase1/prewarm_stage1.py "$MASTER" "$OUT" \
    --ports "$PORTS" --workers "$P" >> "$OUT/log_prewarm.log" 2>&1 \
  && echo "[ok] prewarm" || echo "[warn] prewarm 일부 실패 — 축 단계에서 개별 재시도된다"

# ── 5축 동시 ─────────────────────────────────────────────────────────────────
echo "--- 5축 동시 ($(ls "$SHARD_DIR"/*.jsonl | wc -l)샤드 × 5축) ---"
i=0
for s in $(cd "$SHARD_DIR" && ls *.jsonl); do
  for a in protagonist space time causality intentionality; do
    echo "$SHARD_DIR/$s $a http://127.0.0.1:$((8001 + i % NREP))/v1"
    i=$((i+1))
  done
done | xargs -P"$P" -n3 bash -c \
  'bash scripts/eacl_phase1/resilient_shard.sh "$1" "$2" "'"$OUT"'" 20 "$3"' _

echo "=== DELTA LABEL done $(date +%H:%M:%S) · 건너뛴 문서 $(wc -l < "$OUT/_skipped.log" 2>/dev/null || echo 0) ==="
echo "LABEL_ALLDONE"
