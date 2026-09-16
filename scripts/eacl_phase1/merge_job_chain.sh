#!/bin/bash
# VESSL job(b61, 그룹 4·5) 산출을 로컬 사슬 디렉토리로 병합한다.
# 🔒 **로컬에 없는 문서만** 복사한다 — 로컬 산출을 덮어쓰지 않는다(중복 append 방지).
# job 은 parts/*.tar 로 회수하므로 tar 를 풀어 staging 에 놓고 비교한다.
set -u
REPO=/root/Hi-OnTop
SRC=/shared/hiontop/out/phase1_chain_qwen_v31b
DST="$REPO/outputs/eacl_phase0/phase1_chain_qwen_v31"
STAGE=/tmp/chain_job_stage
JOURNAL="$REPO/EACL/UNATTENDED_2026-08-01_chain_gptoss.md"
LOG="$REPO/outputs/runs/merge_job_chain.log"
POLL="${POLL:-300}"; MAX_WAIT_H="${MAX_WAIT_H:-6}"
mkdir -p "$STAGE" "$(dirname "$LOG")"
kst(){ TZ=Asia/Seoul date '+%H:%M KST'; }
say(){ echo "$(TZ=Asia/Seoul date -Is) $*" | tee -a "$LOG"; }
journal(){ printf '\n%s\n' "$1" >> "$JOURNAL"; }

deadline=$(( $(date +%s) + MAX_WAIT_H*3600 ))
say "b61 job 완료 대기"
while :; do
  alive=$("$REPO/.venv/bin/python" - <<'PY'
import json,subprocess
d=json.loads(subprocess.run(['vesslctl','job','list','--per-page','300','-o','json'],
                            capture_output=True,text=True).stdout)
print(sum(1 for j in d if str(j.get('name','')).startswith('b61')
          and j.get('workloadState') in ('running','created','pending')))
PY
)
  done_sh=$(grep -l "40 추출 완료" "$SRC"/_stage_*.log 2>/dev/null | wc -l)
  say "b61 잔여 $alive · 완료 샤드 $done_sh"
  [ "${alive:-1}" -eq 0 ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    journal "### [$(kst)] 🔴 b61 병합 대기 타임아웃 — 부분 산출만 병합한다."
    break
  fi
  sleep "$POLL"
done

# ── tar 전개 후 로컬에 없는 문서만 복사 ──────────────────────────────────
n_tar=$(ls "$SRC"/parts/*.tar 2>/dev/null | wc -l)
for t in "$SRC"/parts/*.tar; do [ -f "$t" ] && tar -xf "$t" -C "$STAGE" 2>/dev/null; done
copied=0; skipped=0
for d in "$STAGE"/*/; do
  doc=$(basename "$d")
  [ -f "$d/causalchain_states.jsonl" ] || continue
  if [ -s "$DST/$doc/causalchain_states.jsonl" ]; then skipped=$((skipped+1)); continue; fi
  mkdir -p "$DST/$doc" && cp "$d"/*.json* "$DST/$doc/" 2>/dev/null && copied=$((copied+1))
done
total=$(find "$DST" -name causalchain_states.jsonl -size +0 | wc -l)
journal "### [$(kst)] job 산출 병합 — tar ${n_tar}개 · **신규 ${copied}편** 복사 · 기존 유지 ${skipped}편
합계 라벨 보유 **${total}편** (\`phase1_chain_qwen_v31/\`)"
say "병합 완료: 신규 $copied · 스킵 $skipped · 합계 $total"
