#!/usr/bin/env bash
# Job 이 읽을 코드·데이터를 /shared 에 스테이징한다 (capregion Job 관례).
#
#   bash scripts/vessl/stage_to_shared.sh <샤드디렉토리> [worklist]
#
# private repo 라 Job 안에서 git clone 이 안 되므로, 필요한 스크립트만 복사한다.
# 모델(67GB)은 별도 — /shared/models/Qwen3.6-35B-A3B 에 이미 배치.
set -euo pipefail
cd /root/Hi-OnTop

SHARD_SRC="${1:?usage: stage_to_shared.sh <샤드디렉토리> [worklist]}"
WORKLIST_SRC="${2:-configs/vessl/worklist.txt}"
STAGE="${STAGE:-/shared/hiontop}"

[ -d "$SHARD_SRC" ] || { echo "샤드 디렉토리 없음: $SHARD_SRC"; exit 1; }
[ -f "$WORKLIST_SRC" ] || { echo "worklist 없음: $WORKLIST_SRC"; exit 1; }

echo "=== 스테이징 → $STAGE ==="
mkdir -p "$STAGE/code/scripts/eacl_phase0" "$STAGE/code/scripts/eacl_phase1" \
         "$STAGE/code/src" "$STAGE/data/shards"

# 1. 클라이언트 코드 (llm_s5_states.py + resilient_shard.sh + 의존 모듈)
cp scripts/eacl_phase0/llm_s5_states.py   "$STAGE/code/scripts/eacl_phase0/"
cp scripts/eacl_phase1/resilient_shard.sh "$STAGE/code/scripts/eacl_phase1/"
cp -r src/hi_ontop                         "$STAGE/code/src/" 2>/dev/null || true

# 2. 실행 스크립트
cp scripts/vessl/job_shard.sh "$STAGE/"
chmod +x "$STAGE/job_shard.sh"

# 3. worklist + 샤드 데이터
cp "$WORKLIST_SRC" "$STAGE/worklist.txt"
cp "$SHARD_SRC"/*.jsonl "$STAGE/data/shards/"

echo "--- 결과 ---"
printf 'code      : %s 파일\n' "$(find "$STAGE/code" -type f | wc -l)"
printf 'shards    : %s 개 (%s)\n' "$(ls "$STAGE/data/shards" | wc -l)" "$(du -sh "$STAGE/data/shards" | cut -f1)"
printf 'worklist  : %s 줄\n' "$(grep -cvE '^\s*(#|$)' "$STAGE/worklist.txt")"
printf 'model     : %s\n' "$(du -sh /shared/models/Qwen3.6-35B-A3B 2>/dev/null | cut -f1 || echo '없음!')"
