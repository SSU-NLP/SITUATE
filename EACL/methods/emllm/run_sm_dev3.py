#!/usr/bin/env python3
"""EM-LLM **SM 변형** 정식 구현 — dev-3 (하네스 A).

## SM 이 무엇인가 (Fountas et al. 2025)
EM-LLM 의 사건 경계 = 토큰 서프라이즈가 **직전 구간의 통계**를 넘을 때:

    divide_t  ⟺  s_t  >  mean(s[window]) + γ · std(s[window])

원본 구현(`em_llm/attention/context_manager.py:921-923`, `em_llm/attention/em_llm.py:218-225`):

```python
avg_st = max(global_remainder_ed - n_local, 0)
avg_ed = max(global_remainder_ed - exc_length, n_init)
divide = exc_surprisal > gamma * std(remainder_surprisal[:, avg_st:avg_ed]) \
                       + mean(remainder_surprisal[:, avg_st:avg_ed])
```

⇒ 창은 **직전 `n_local` 토큰의 이동 윈도우**다. `n_local` 기본값 **4096**(`config/llama3.yaml:19`),
γ 기본값 **1.1**(`em_llm.py:19`).

## 🔴 기존 저장소 수치(0.2711)는 SM 이 아니다
`2026-07-31_m9_attn_entity/REPORT.md` 의 EM-LLM 행은 **OP-1**(고정 분위수)이고,
OP-2 조차 *"μ+1.1σ online **full-past**"* 였다 — 즉 **이동 윈도우가 아니라 전체 과거**를 썼다.
문서가 4096토큰을 넘으면 원본과 갈라진다(dev-3 에서 tunnel≈4.6k · monkey≈6.8k 토큰).
⇒ 기존 0.2711 은 **EM-LLM 행이 아니라 ablation** 으로 재배치한다(사용자 지시, 2026-08-03).

## 범위 — SM 만, SM+ 제외
원본은 SM 뒤에 **그래프 정제**(modularity/conductance 로 경계 재배치)를 얹을 수 있다(SM+).
그건 경계 이후 구간의 유사도가 필요해 **미래 문맥을 쓴다** ⇒ 온라인 주장과 충돌하므로 **제외**.
(우리 저장소의 `+modularity` 0.2978 행이 그 SM+ 계열이며, 온라인 비교표에 넣으면 안 된다.)

## γ 는 고정
γ = 1.1 (원본 기본값). **문서별 튜닝 없음**(사용자 지시). 창 크기도 원본 n_local=4096 고정.

## 입력
LM 재실행 없이 `2026-07-31_m9_attn_entity/cache/{doc}__emllm.npz` 의 **문장별 surprise** 를 쓴다
(그 실험이 생산한 것). 토큰 창을 문장 창으로 환산하기 위해 문장별 토큰 수가 필요하며,
LM 토크나이저로 실측한다.

    .venv/bin/python EACL/methods/emllm/run_sm_dev3.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))

CACHE = REPO / "outputs/experiments/2026-07-31_m9_attn_entity/cache"
PRED = REPO / "outputs/experiments/2026-07-29_baseline_compare/preds"
OUT = REPO / "outputs/experiments/2026-08-03_emllm_sm_dev3"

GAMMA = 1.1        # em_llm.py:19 원본 기본값
N_LOCAL = 4096     # config/llama3.yaml:19 원본 기본값 (토큰)
WARMUP = 3         # 창 표본이 이 미만이면 판정 보류 (원본의 first-chunk 분기에 대응)


def sentences_of(doc: str) -> list[str]:
    rows = [json.loads(l) for l in
            (REPO / f"outputs/eacl_phase0/{doc}_michelmann_master.jsonl").open()]
    return [rows[0]["previous"]] + [r["current"] for r in rows]


def sm_boundaries(surprise: np.ndarray, tok_len: np.ndarray, gamma: float,
                  n_local: int, warmup: int) -> list[int]:
    """SM 규칙 — 직전 `n_local` **토큰**에 해당하는 문장들로 창을 만든다.

    원본이 토큰 단위 창을 쓰므로, 문장 단위 surprise 에서는 **누적 토큰 수를 거슬러 올라가며**
    창의 시작 문장을 정한다. 이것이 `avg_st = max(ed − n_local, 0)` 의 문장판이다.
    창이 문서 전체를 덮으면 자동으로 full-past 와 같아진다(짧은 문서에서 원본도 그렇다).
    """
    n = len(surprise)
    cum = np.concatenate([[0], np.cumsum(tok_len)])     # cum[i] = 문장 i 이전까지의 토큰 수
    bnd = []
    for i in range(1, n):
        lo_tok = max(0, cum[i] - n_local)
        st = int(np.searchsorted(cum, lo_tok, side="right") - 1)
        st = max(0, min(st, i - 1))
        win = surprise[st:i]
        if len(win) < warmup:
            continue
        if surprise[i] > win.mean() + gamma * win.std():
            bnd.append(i)
    return bnd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=GAMMA, help="고정. 문서별 튜닝 금지")
    ap.add_argument("--n-local", type=int, default=N_LOCAL, help="원본 기본 4096 토큰")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B",
                    help="🔴 surprise 를 만든 LM 과 **같은** 토크나이저여야 한다 (n_local 은 그 LM 의 토큰 수다)")
    # 2026-08-04 정정: Llama 토크나이저로 세던 것을 Qwen2.5-7B 로 맞춤)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    from baseline_compare import load_dataset          # noqa: PLC0415
    data = {d: g for d, _s, g in load_dataset("dev3")}

    preds, meta = {}, {}
    print(f"γ={args.gamma} (고정) · n_local={args.n_local}토큰 · warmup={args.warmup} · SM+ 제외")
    for doc in ("pieman", "tunnel", "monkey"):
        z = np.load(CACHE / f"{doc}__emllm.npz", allow_pickle=True)
        surprise = np.asarray(z["surprise"], dtype=float)
        sents = sentences_of(doc)
        if len(sents) != len(surprise):
            print(f"  ⚠️ {doc}: 문장 {len(sents)} vs surprise {len(surprise)} — 짧은 쪽에 맞춘다")
        m = min(len(sents), len(surprise))
        sents, surprise = sents[:m], surprise[:m]
        tok_len = np.array([len(tok.encode(s, add_special_tokens=False)) for s in sents])

        bnd = sm_boundaries(surprise, tok_len, args.gamma, args.n_local, args.warmup)
        gold = data[doc]
        # 전이 i (1-based, gold 인덱스 i-1) = 문장 i 앞의 경계
        vec = [0] * len(gold)
        for b in bnd:
            if 1 <= b <= len(vec):
                vec[b - 1] = 1
        preds[doc] = vec
        win_cov = float(np.mean([min(args.n_local, c) / max(1, c)
                                 for c in np.cumsum(tok_len)[1:]]))
        meta[doc] = {"n_sent": m, "n_tokens": int(tok_len.sum()), "n_pred": sum(vec),
                     "n_gold": int(sum(gold)), "window_covers_full_past_frac": round(win_cov, 3)}
        print(f"  {doc}: 문장 {m} · 토큰 {tok_len.sum()} · 예측 {sum(vec)} · gold {sum(gold)}"
              f" · 창이 전체과거를 덮는 비율 {win_cov:.2f}")

    PRED.mkdir(parents=True, exist_ok=True)
    (PRED / "EM-LLM-SM_dev3.json").write_text(json.dumps(preds))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sm_dev3.json").write_text(json.dumps({"config": vars(args), "meta": meta,
                                                  "preds": preds}, ensure_ascii=False, indent=2))
    print(f"\n저장: {PRED / 'EM-LLM-SM_dev3.json'}")
    print("다음: scripts/eacl_phase0/preds_table.py --datasets dev3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
