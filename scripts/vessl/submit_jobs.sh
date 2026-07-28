#!/usr/bin/env bash
# S5 축 추출을 VESSL Job N개로 제출한다. Job 1개 = A100 1장 = vLLM 복제 1개.
#
#   DRY_RUN=1 N=2  RUN=test1 bash scripts/vessl/submit_jobs.sh    # 명령만 출력
#            N=2  RUN=test1 bash scripts/vessl/submit_jobs.sh     # 검증용 2개
#            N=20 RUN=phase2 bash scripts/vessl/submit_jobs.sh    # 본run
#
# 비용: resourcespec-a100x1 = $1.55/hr. N=20 → $31.00/hr.
# 상태: vesslctl job list · 로그: vesslctl job logs <slug>
set -euo pipefail

N="${N:-2}"
RUN="${RUN:?RUN 필요 — 출력 폴더명 겸 job 이름 접두사}"
DRY_RUN="${DRY_RUN:-0}"

SPEC="${SPEC:-resourcespec-a100x1}"
VOL="${VOL:-objvol-gsvyr0eu87wt}"      # ssu-volume → /shared
# capregion Job 들이 성공적으로 쓰는 이미지 (이 클러스터에서 검증됨)
IMAGE="${IMAGE:-quay.io/vessl-ai/torch@sha256:8bd418d49eeed841a607582b614c9d3ba86590b07d1be671816176c47253344a}"
STAGE="${STAGE:-/shared/hiontop}"
OUT_DIR="$STAGE/out/$RUN"
CONC="${CONC:-130}"

echo "=== 제출 계획 ==="
printf '  job    : %s-shard{0..%d}\n' "$RUN" "$((N-1))"
printf '  spec   : %s × %d\n' "$SPEC" "$N"
printf '  볼륨   : %s → /shared\n' "$VOL"
printf '  출력   : %s\n' "$OUT_DIR"
printf '  비용   : $%.2f/시간\n' "$(awk -v n="$N" 'BEGIN{print n*1.55}')"
echo

for i in $(seq 0 $((N-1))); do
  cmd=(vesslctl job create
       -n "${RUN}-shard${i}"
       -r "$SPEC"
       -i "$IMAGE"
       --object-volume "${VOL}:/shared"
       --working-dir "$STAGE/code"
       -e "SHARD_ID=$i" -e "NUM_SHARDS=$N"
       -e "OUT_DIR=$OUT_DIR" -e "STAGE=$STAGE" -e "CONC=$CONC"
       --cmd "bash $STAGE/job_shard.sh")
  if [ "$DRY_RUN" = "1" ]; then printf '%q ' "${cmd[@]}"; echo; echo
  else "${cmd[@]}" || echo "[FAIL] shard $i 제출 실패"
  fi
done

[ "$DRY_RUN" = "1" ] && exit 0
echo
echo "제출 완료: $N개"
echo "  상태: vesslctl job list | head -$((N+1))"
echo "  로그: vesslctl job logs ${RUN}-shard0"
echo "  중단: vesslctl job terminate <slug>"
