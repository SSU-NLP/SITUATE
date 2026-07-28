#!/bin/bash
# 샤드 단위 라벨링 — **문제 문서를 건너뛰고 계속**.
#
# 왜 필요한가: llm_s5_states 는 Stage-1 JSON 파싱 실패 시 예외로 죽는다. 샤드에 120문서가 있으면
# 문서 하나 때문에 나머지 118개가 통째로 날아간다(shard35 실측: 120 중 2개만 처리하고 사망).
#
# 실패 원인(양쪽 함정):
#   · --stage1-max-tokens 큼(8000) → 입력+출력 > max-model-len → 400 BadRequest
#   · --stage1-max-tokens 작음(2000) → 엔티티 많은 문서에서 목록이 **절단** → JSONDecodeError
# 두 경우 모두 해당 문서만 빼고 진행하는 게 합리적이다(전체 8,000편 중 소수).
#
# 검출: Stage-1 이 성공하면 {OUT}/{doc}/story_lists.json 이 생긴다. 실패 후 "샤드 순서상 아직
# story_lists.json 이 없는 첫 문서" = 방금 죽인 문서. 그것을 샤드에서 제거하고 재시도한다.
set -u
cd /root/Hi-OnTop
SHARD="${1:?usage: resilient_shard.sh <shard.jsonl> <axis> <out> [max_skips]}"
AXIS="${2:?}"
OUT="${3:?}"
MAXSKIP="${4:-20}"
BASEURL="${5:-http://localhost:8001/v1}"
PY=/root/Hi-OnTop/.venv/bin/python
COMMON="--model qwen36-35b --engine vllm --no-reasoning \
--base-url $BASEURL --prompt-version v3.1 --stage1-max-tokens 8000"
NAME=$(basename "$SHARD" .jsonl)
WORK="$OUT/_work_${NAME}_${AXIS}.jsonl"
SKIPLOG="$OUT/_skipped.log"
cp "$SHARD" "$WORK"

for i in $(seq 0 "$MAXSKIP"); do
  if $PY scripts/eacl_phase0/llm_s5_states.py "$WORK" "$OUT" $COMMON --axis "$AXIS" \
       >> "$OUT/log_${NAME}_${AXIS}.log" 2>&1; then
    echo "[ok] $NAME $AXIS (skips=$i)"
    rm -f "$WORK"; exit 0
  fi
  # 범인 특정 2단계:
  #  ① 로그 마지막 RuntimeError "[doc] axis: no valid outputs" → Stage-2 실패 문서 (검출기 v2)
  #  ② 없으면 Stage-1 실패 → 샤드 순서상 story_lists.json 이 아직 없는 첫 문서
  BAD=$(grep -aoE '^RuntimeError: \[[^]]+\]' "$OUT/log_${NAME}_${AXIS}.log" 2>/dev/null \
        | tail -1 | sed -E 's/^RuntimeError: \[([^]]+)\]$/\1/')
  if [ -z "$BAD" ]; then
  BAD=$($PY - "$WORK" "$OUT" <<'EOF'
import json, sys, pathlib
work, out = sys.argv[1], pathlib.Path(sys.argv[2])
seen = []
for line in open(work):
    d = json.loads(line)["doc_id"]
    if not seen or seen[-1] != d:
        seen.append(d)
for d in seen:
    if not (out / d / "story_lists.json").exists():
        print(d); break
EOF
)
  fi
  if [ -z "$BAD" ]; then
    echo "[FAIL] $NAME $AXIS — 실패했으나 문제 문서 특정 불가(재시도 중단)"; exit 1
  fi
  echo "$(date +%H:%M:%S) $NAME $AXIS SKIP $BAD" >> "$SKIPLOG"
  $PY - "$WORK" "$BAD" <<'EOF'
import json, sys
work, bad = sys.argv[1], sys.argv[2]
rows = [l for l in open(work) if json.loads(l)["doc_id"] != bad]
open(work, "w").writelines(rows)
EOF
done
echo "[FAIL] $NAME $AXIS — skip 상한 $MAXSKIP 초과"; exit 1
