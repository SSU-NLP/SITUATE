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
# 목적지는 **런 네임스페이스로 분리**한다. 같은 이름의 샤드가 다른 코퍼스를 가리키면
# 조용히 엉뚱한 문서를 라벨링한다(2026-08-03: 구 샤드 100개에 delta 아닌 문서 1,829편).
SHARD_DEST="${SHARD_DEST:-$STAGE/data/shards}"
WORKLIST_DEST="${WORKLIST_DEST:-$STAGE/worklist.txt}"

[ -d "$SHARD_SRC" ] || { echo "샤드 디렉토리 없음: $SHARD_SRC"; exit 1; }
[ -f "$WORKLIST_SRC" ] || { echo "worklist 없음: $WORKLIST_SRC"; exit 1; }

echo "=== 스테이징 → $STAGE (샤드 $SHARD_DEST · worklist $WORKLIST_DEST) ==="
mkdir -p "$STAGE/code/scripts/eacl_phase0" "$STAGE/code/scripts/eacl_phase1" \
         "$STAGE/code/src" "$SHARD_DEST"

# 1. 클라이언트 코드 (llm_s5_states.py + resilient_shard.sh + 의존 모듈)
cp scripts/eacl_phase0/llm_s5_states.py   "$STAGE/code/scripts/eacl_phase0/"
cp scripts/eacl_phase1/resilient_shard.sh "$STAGE/code/scripts/eacl_phase1/"
cp scripts/eacl_phase1/prewarm_stage1.py  "$STAGE/code/scripts/eacl_phase1/"   # job_shard.sh §3 이 부른다
cp -r src/hi_ontop                         "$STAGE/code/src/" 2>/dev/null || true

# 2. 실행 스크립트
cp scripts/vessl/job_shard.sh "$STAGE/"
chmod +x "$STAGE/job_shard.sh"

# 3. worklist + 샤드 데이터
cp "$WORKLIST_SRC" "$WORKLIST_DEST"
cp "$SHARD_SRC"/*.jsonl "$SHARD_DEST/"

# 4. preflight — worklist 가 가리키는 샤드가 전부 스테이징됐나. 여기서 못 잡으면
#    A100 4장을 1시간 켠 뒤에야 알게 된다(2026-08-03 실측: 5회차 전부 그렇게 날림).
miss=0
while read -r name _axis; do
  [ -s "$SHARD_DEST/$name" ] || [ -s "$SHARD_DEST/$name.jsonl" ] || { echo "  !! 샤드 없음: $name"; miss=$((miss+1)); }
done < <(grep -vE '^\s*(#|$)' "$WORKLIST_DEST" | sort -u -k1,1)

echo "--- 결과 ---"
printf 'code      : %s 파일\n' "$(find "$STAGE/code" -type f | wc -l)"
printf 'shards    : %s 개 (%s)\n' "$(ls "$SHARD_DEST" | wc -l)" "$(du -sh "$SHARD_DEST" | cut -f1)"
printf 'worklist  : %s 줄 · 미스 %s\n' "$(grep -cvE '^\s*(#|$)' "$WORKLIST_DEST")" "$miss"
printf 'model     : %s\n' "$(du -sh /shared/models/Qwen3.6-35B-A3B 2>/dev/null | cut -f1 || echo '없음!')"
[ "$miss" -eq 0 ] || { echo "🔴 worklist 가 없는 샤드를 가리킨다 — 제출 금지"; exit 1; }
