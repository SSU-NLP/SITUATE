#!/bin/bash
# 가이드(EACL/phase0-annotation-guide.md) 실측에 맞춰 델타 라벨링을 다시 세운다.
#
# 고치는 것 넷:
#  ① 설정 파리티 — 로컬 복제에 `--reasoning-parser qwen3` 가 빠져 있었다. 기존 9,848편은
#     VESSL job_shard.sh(파서 포함)로 라벨링됐다. 가이드 §⑦: "이전 런 결과를 이어받는 것도
#     같은 기준" → 파서 없이 만든 델타 주석은 폐기하고 다시 만든다.
#  ② 샤드 세분화 — "동시 요청 수 = 병렬 (샤드×축) 작업 수". 32샤드면 Phase A 가 32작업뿐이라
#     GPU 가 50% 에서 막힌다. 200샤드 × 5축 = 1,000작업.
#  ③ 구조 — Phase A/B 로 나누면 Phase A 병렬도가 1/5 로 떨어진다. prewarm 으로 Stage-1 을
#     먼저 채우고 5축을 한꺼번에 돌린다.
#  ④ 로컬 출력은 cephfs(871MB/s) 라 그대로 둔다. geesefs 직접쓰기 문제는 VESSL 쪽 사안.
set -u
REPO=/root/Hi-OnTop
cd "$REPO"
JOURNAL="$REPO/EACL/UNATTENDED_2026-08-02_delta_cap20.md"
j(){ printf '\n- **[%s]** %s\n' "$(TZ=Asia/Seoul date '+%m-%d %H:%M KST')" "$1" >> "$JOURNAL"; }

echo "── 1. 감독기·드라이버 정지"
pkill -f 'supervise_delta.sh' 2>/dev/null; sleep 2
pkill -f 'run_label_delta.sh' 2>/dev/null; sleep 3
pkill -f llm_s5_states 2>/dev/null; sleep 5
pkill -9 -f llm_s5_states 2>/dev/null; sleep 2

echo "── 2. 파서 없이 만든 델타 주석 폐기"
"$REPO/.venv/bin/python" - <<'PY'
import json, shutil
from pathlib import Path
root = Path('outputs/eacl_phase0/phase1_all_v31')
want = {json.loads(l)["doc_id"]
        for d in ("delta_local", "delta_vessl")
        for f in Path(f"data/eacl_phase1/{d}").glob("*.jsonl") for l in open(f)}
n = 0
for d in want:
    p = root / d
    if p.exists():
        shutil.rmtree(p); n += 1
print(f"   델타 주석 {n}편 삭제 · 남은 문서 {sum(1 for x in root.iterdir() if x.is_dir()):,}")
PY

echo "── 3. 로컬 샤드 32 → 200"
"$REPO/.venv/bin/python" - <<'PY'
import json
from collections import defaultdict
from pathlib import Path
src = Path('data/eacl_phase1/delta_local')
rows = [json.loads(l) for f in sorted(src.glob('*.jsonl')) for l in open(f)]
by = defaultdict(list)
for r in rows: by[r['doc_id']].append(r)
docs = sorted(by.items(), key=lambda t: -len(t[1]))       # 긴 편부터 라운드로빈
NS = 200
buckets = [[] for _ in range(NS)]
for i, (_, rs) in enumerate(docs): buckets[i % NS].append(rs)
for p in src.glob('*.jsonl'): p.unlink()
for i, b in enumerate(buckets):
    with (src / f"shard{i:03d}.jsonl").open('w', encoding='utf-8') as f:
        for rs in b:
            for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"   {len(docs):,}편 → {NS}샤드 (편/샤드 {min(len(b) for b in buckets)}~{max(len(b) for b in buckets)})")
PY

echo "── 4. vLLM 4복제 재기동 (--reasoning-parser qwen3 포함)"
for p in $(pgrep -f api_server); do kill "$p" 2>/dev/null; done
sleep 15
for p in $(pgrep -f api_server); do kill -9 "$p" 2>/dev/null; done
sleep 10
export PATH="$REPO/.venv-vllm/bin:$PATH"
command -v ninja >/dev/null || { echo "🔴 ninja 없음 — 중단"; j "🔴 ninja 를 PATH 에서 못 찾아 재기동 중단"; exit 1; }
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g nohup "$REPO/.venv-vllm/bin/python" -m vllm.entrypoints.openai.api_server \
    --model /root/models/Qwen3.6-35B-A3B --served-model-name qwen36-35b --quantization fp8 \
    --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --max-num-seqs 300 --reasoning-parser qwen3 --trust-remote-code --enable-prefix-caching \
    --port $((8001+g)) > "$REPO/outputs/runs/vllm_delta_$g.log" 2>&1 &
  disown
  echo "   복제 $g 기동 (port $((8001+g)))"
  sleep 5
done

j "가이드 실측에 맞춰 재구성 — ① 로컬 복제에 \`--reasoning-parser qwen3\` 추가(기존 9,848편과 설정 파리티; 파서 없이 만든 델타 주석 전량 폐기) ② 로컬 샤드 32→200 (동시 요청 수 = 샤드×축 작업 수) ③ prewarm+5축 동시 구조 ④ VESSL \`delta-cap20d\` 는 파서 포함이라 그대로 진행"
echo "── 완료. 감독기는 별도로 기동한다."
