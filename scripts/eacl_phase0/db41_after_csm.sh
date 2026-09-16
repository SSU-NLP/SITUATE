#!/usr/bin/env bash
# CSM 학습이 끝나면 **동결 작동점으로 표 전체를 재실행**한다 (🔒 D-B41).
#
# 왜 스크립트인가: 학습이 ~11시간 남아 세션이 살아 있을 보장이 없다. 기계적인 부분(재실행·덤프)은
# 여기서 자동으로 끝내 두고, 세션이 돌아오면 **REPORT 산문만** 쓰면 되게 한다.
#
# ⚠️ kill 은 PID 로. `pkill -f db41_after_csm` 은 자기 명령줄까지 매치해 자폭한다(CLAUDE.md).
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
OUT=outputs/experiments/2026-07-29_baseline_compare
LOG=$OUT/db41_rerun.log
CKPT_DIR=external/Dialogue-Topic-Segmenter/checkpoints
TARGET=${TARGET:-249050}        # 24905 step/epoch x 10 epoch

say() { echo "[$(TZ=Asia/Seoul date '+%m/%d %H:%M KST')] $*" | tee -a "$LOG"; }

say "감시 시작 — CSM 학습(torchrun) 종료 대기"
while ps -eo args | grep -q "[t]rain\.py -t ./data/train/dailydialog"; do sleep 120; done

LAST=$(ls "$CKPT_DIR" 2>/dev/null | sed 's/cpt_//;s/\.pth//' | sort -n | tail -1)
say "학습 프로세스 종료. 마지막 체크포인트 step=${LAST:-없음} / 목표 $TARGET"
if [ -z "${LAST:-}" ]; then say "🔴 체크포인트 없음 — 중단"; exit 1; fi
if [ "$LAST" -lt $((TARGET - 25000)) ]; then
  say "⚠️ 완주 전 종료로 보인다(step $LAST). 표에 진행도를 반드시 병기할 것."
fi

# 1) 서사(dev-3) — 우리 방법 + baseline 5종. LLM 2행은 캐시가 있어 재실행 불필요.
say "① dev-3 재실행"
$PY -u scripts/eacl_phase0/baseline_compare.py --datasets dev3 \
    --methods M5,M4,M1-CosDrift,TextTiling,GraphSeg,GreedySeg,CSM,RoBERTa >>"$LOG" 2>&1
say "① 완료 (exit $?)"

# 2) 대화(DTS-3) — OOD 전이 패널
say "② DTS-3 재실행"
$PY -u scripts/eacl_phase0/baseline_compare.py \
    --datasets dts_full_tiage,dts_full_dialseg711,dts_full_superseg \
    --methods M5,M4,M1-CosDrift,TextTiling,GraphSeg,GreedySeg,CSM,RoBERTa >>"$LOG" 2>&1
say "② 완료 (exit $?)"

# 3) CSM 체크포인트 곡선 마무리 (dev = superseg/validation, test 미개봉)
say "③ CSM 체크포인트 sweep 마무리"
$PY -u scripts/eacl_phase0/csm_ckpt_sweep.py --n_dial 60 --device cuda:0 \
    --backbone bert-base-uncased >>"$LOG" 2>&1
say "③ 완료 (exit $?)"

say "✅ 기계 작업 끝. 남은 것 = REPORT.md 산문 + ledger D-B42 (세션이 작성)"
