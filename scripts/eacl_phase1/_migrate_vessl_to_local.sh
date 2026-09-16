#!/bin/bash
# VESSL 담당분을 로컬로 이관한다.
#
# 왜: VESSL 이 다섯 번 연속 산출물을 못 냈다(delta-cap20 / b / c / c-r1 / c-r2 / d).
# 마지막 시도는 태스크가 `[ok]` 로 끝났는데도 `<doc_id>/` 디렉터리가 0개였다 — geesefs 쓰기가
# 반영되지 않는 것으로 보인다. 로컬은 같은 코드로 정상 동작하며 59 축파일/분을 낸다.
# 디버깅을 계속하기보다 확실한 경로로 몰아준다. 남은 Job 은 토큰 권한이 없어 종료 못 하지만
# 스스로 끝난다.
#
# 이미 끝난 문서는 {axis}_states.jsonl 캐시로 즉시 건너뛰므로 재작업이 없다.
set -u
REPO=/root/Hi-OnTop
cd "$REPO"
JOURNAL="$REPO/EACL/UNATTENDED_2026-08-02_delta_cap20.md"
j(){ printf '\n- **[%s]** %s\n' "$(TZ=Asia/Seoul date '+%m-%d %H:%M KST')" "$1" >> "$JOURNAL"; }

echo "── 감독기·드라이버 정지"
pkill -f 'supervise_delta.sh' 2>/dev/null; sleep 2
pkill -f 'run_label_delta.sh' 2>/dev/null; sleep 3
pkill -f llm_s5_states 2>/dev/null; sleep 5
pkill -9 -f llm_s5_states 2>/dev/null; sleep 2

echo "── 전체 3,055편을 로컬 200샤드로 재편"
"$REPO/.venv/bin/python" - <<'PY'
import json
from collections import defaultdict
from pathlib import Path
L = Path('data/eacl_phase1/delta_local'); V = Path('data/eacl_phase1/delta_vessl')
rows = [json.loads(l) for p in (L, V) for f in sorted(p.glob('*.jsonl')) for l in open(f)]
by = defaultdict(list)
for r in rows: by[r['doc_id']].append(r)
docs = sorted(by.items(), key=lambda t: -len(t[1]))          # 긴 편부터 라운드로빈
NS = 200
buckets = [[] for _ in range(NS)]
for i, (_, rs) in enumerate(docs): buckets[i % NS].append(rs)
for p in L.glob('*.jsonl'): p.unlink()
for i, b in enumerate(buckets):
    with (L / f"shard{i:03d}.jsonl").open('w', encoding='utf-8') as f:
        for rs in b:
            for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"   {len(docs):,}편 → {NS}샤드 (편/샤드 {min(len(b) for b in buckets)}~{max(len(b) for b in buckets)})")
PY

# VESSL 담당 목록을 비워 감독기가 재제출·이관을 더 시도하지 않게 한다
mkdir -p data/eacl_phase1/_delta_vessl_retired
mv data/eacl_phase1/delta_vessl/*.jsonl data/eacl_phase1/_delta_vessl_retired/ 2>/dev/null
echo "── delta_vessl 비움 (감독기 재제출 중단)"

j "🔴 VESSL 5회 연속 산출물 0 — 마지막 시도는 태스크가 \`[ok]\` 인데도 문서 디렉터리 0개(geesefs 쓰기 미반영 추정). **VESSL 담당 1,519편을 로컬로 이관**하고 \`delta_vessl\` 을 비워 재제출을 중단했다. 로컬 전체 3,055편 · 200샤드 · 실측 59 축파일/분 → 약 3.4시간 예상. 남은 Job 은 terminate 권한이 없어 스스로 끝나기를 기다린다."
echo "── 완료"
