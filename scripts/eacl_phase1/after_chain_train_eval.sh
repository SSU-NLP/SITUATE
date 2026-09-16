#!/bin/bash
# Qwen3.6 사슬 추출이 끝나면 → 앵커 인과 열 교체 → M5 학습 → W3/I2/chain 판정.
#
# 사용자 판정 규칙(사전 고정):
#   W3 > I2   면 의도축 안 바꾼다
#   I2 > chain 면 인과축 안 바꾼다
# 🔴 판정 자체는 사람이 한다. 이 스크립트는 **수치만** 만들어 저널에 적는다.
set -u
REPO=/root/Hi-OnTop
CHAIN_DIR="$REPO/outputs/eacl_phase0/phase1_chain_qwen_v31"
SPLIT=phase1_chainq_i2
NS=_${SPLIT}_per_slot_slot_pwone
JOURNAL="$REPO/EACL/UNATTENDED_2026-08-01_chain_gptoss.md"
LOG="$REPO/outputs/runs/after_chain_train_eval.log"
MIN_DOCS="${MIN_DOCS:-10000}"; POLL="${POLL:-300}"; MAX_WAIT_H="${MAX_WAIT_H:-8}"
mkdir -p "$(dirname "$LOG")"
kst(){ TZ=Asia/Seoul date '+%H:%M KST'; }
say(){ echo "$(TZ=Asia/Seoul date -Is) $*" | tee -a "$LOG"; }
journal(){ printf '\n%s\n' "$1" >> "$JOURNAL"; }

deadline=$(( $(date +%s) + MAX_WAIT_H*3600 ))
say "대기 시작 — Qwen3.6 사슬 추출 완료 감시(문서 ${MIN_DOCS}편 이상 + 추출 프로세스 종료)"

while :; do
  n=$(find "$CHAIN_DIR" -name causalchain_states.jsonl -size +0 2>/dev/null | wc -l)
  running=$(ps -eo args | grep -c "chain_shard\.p[y]")
  say "사슬 문서 $n · 추출 프로세스 $running"
  if [ "$n" -ge "$MIN_DOCS" ] && [ "$running" -eq 0 ]; then
    journal "### [$(kst)] ✅ Qwen3.6 사슬 확보 ${n}편 — 앵커 열 교체 → M5 학습 착수"
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    journal "### [$(kst)] 🔴 사슬 대기 타임아웃 — 문서 ${n}편(${MIN_DOCS} 미만). 학습 착수 안 함."
    say "타임아웃"; exit 1
  fi
  sleep "$POLL"
done

# ── 1. 앵커 인과 열을 Qwen3.6 사슬(strict)로 교체 + 의도 I2 ──────────────
say "앵커 재계산 시작"
"$REPO/.venv/bin/python" "$REPO/scripts/eacl_phase0/recompute_axes.py" \
  --causality chainstrict --intent i2 --chain-src "outputs/eacl_phase0/phase1_chain_qwen_v31" \
  --out-split "$SPLIT" >> "$LOG" 2>&1 || { journal "### [$(kst)] 🔴 앵커 재계산 실패"; exit 1; }
rate=$(grep -o "인과 발화율 [0-9.]* -> [0-9.]*" "$LOG" | tail -1)
brate=$(grep -o "앵커 경계율(count≥3) [0-9.]* -> [0-9.]*" "$LOG" | tail -1)
journal "### [$(kst)] 앵커 열 교체 완료 — \`$SPLIT\`
- $rate · $brate"

# ── 2. M5 학습 3시드 (GPU 3장, 1장은 여유로 둔다) ────────────────────────
say "학습 착수"
PIDS=()
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s "$REPO/.venv/bin/python" \
    "$REPO/scripts/eacl_phase0/train_anchor_m4m5.py" --split "$SPLIT" --models M5 \
    --pos-weight-mode one --pred-space slot --dev3 --seed "$s" \
    > "$REPO/outputs/experiments/m4m5_phase1_anchor/train_${SPLIT}_s${s}.log" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
nck=$(ls "$REPO/outputs/experiments/m4m5_phase1_anchor/M5${NS}_seed"*.pt 2>/dev/null | wc -l)
say "학습 완료 체크포인트 $nck/3"
[ "$nck" -lt 3 ] && { journal "### [$(kst)] 🔴 학습 미완 — 체크포인트 ${nck}/3"; exit 1; }

# ── 3. 세 조건 동일 하네스 평가 ──────────────────────────────────────────
say "평가 시작"
OUTJSON="$REPO/outputs/experiments/m4m5_phase1_anchor/W3_I2_CHAIN_DEV3.json"
RES=$(cd "$REPO" && CUDA_VISIBLE_DEVICES=0 .venv/bin/python - <<PYEOF 2>&1 | tail -8
import sys, json
from pathlib import Path
import numpy as np, torch
REPO=Path('.').resolve(); sys.path[:0]=[str(REPO/'src'), str(REPO/'scripts/eacl_phase0')]
from db29_final_eval import load
from train_anchor_m4m5 import Corpus, axis_probs
import train_direct_boundary as m8
dev='cuda'; dev3=Corpus('dev3')
G=[dev3.gold[int(dev3.to[i]):int(dev3.to[i+1])].astype(np.int8) for i in range(dev3.n_docs)]
DOCS=['pieman','tunnel','monkey']
COND=[('W3  (동결 × 동결)','_per_slot_slot_pwone'),
      ('I2  (동결 × I2)','_phase1_i2_per_slot_slot_pwone'),
      ('chain (사슬 × I2)','$NS')]
out={}
for nm,ns in COND:
    A,N,AU,PD=[],[],[],{d:[] for d in DOCS}
    for sd in (0,1,2):
        m=load('M5',sd,dev,ns=ns)[0]
        with torch.no_grad():
            P=axis_probs(m,dev3,list(range(dev3.n_docs)),dev,bs=4,with_anchor=False)
        S=[p.sum(1) for p in P]; r=m8.score_signals(S,G)
        A.append(r['adaptive']['score']); N.append(r['adaptive']['n_pred']); AU.append(r['auc_pooled'])
        from hi_ontop.dts_scoring import score_dialogues
        for i,d in enumerate(DOCS):
            b=m8.cuts_adaptive(S[i])[:len(G[i])]
            PD[d].append(score_dialogues([np.concatenate([G[i],[0]]).tolist()],[np.concatenate([b,[0]]).tolist()])['score'])
        del m; torch.cuda.empty_cache()
    out[nm]={'score':float(np.mean(A)),'sd':float(np.std(A)),'n_pred':float(np.mean(N)),
             'auc':float(np.mean(AU)),'per_doc':{d:float(np.mean(v)) for d,v in PD.items()}}
    o=out[nm]
    print(f"{nm:20s} {o['score']:.4f}±{o['sd']:.4f} · 예측 {o['n_pred']:.0f} ({o['n_pred']/66:.2f}배) · AUC {o['auc']:.3f} · {o['per_doc']['pieman']:.3f}/{o['per_doc']['tunnel']:.3f}/{o['per_doc']['monkey']:.3f}")
json.dump(out,open('$OUTJSON','w'),indent=2,ensure_ascii=False)
w3,i2,ch=[out[k]['score'] for k in out]
print(f"\nΔ(I2−W3) {i2-w3:+.4f}   Δ(chain−I2) {ch-i2:+.4f}")
PYEOF
)
journal "### [$(kst)] ✅ W3 / I2 / chain 판정 수치 (dev-3, 적응임계, 3시드)
\`\`\`
$RES
\`\`\`
판정 규칙(사용자 사전 지정): W3 > I2 면 의도축 유지 · I2 > chain 이면 인과축 유지.
🔴 **판정은 사람이 한다.** 산출 \`W3_I2_CHAIN_DEV3.json\`"
say "완료"
