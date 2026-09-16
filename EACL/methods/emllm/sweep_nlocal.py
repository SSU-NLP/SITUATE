#!/usr/bin/env python3
"""EM-LLM SM 의 `n_local` 민감도 — **이식 검증**용 (성능 경쟁 아님).

## 왜 스윕하나
임계 **규칙** `s_t > μ_window + γ·σ_window` 는 고정이다 — 바꾸면 EM-LLM 이 아니다.
`n_local` 은 그 규칙의 **하이퍼파라미터**이고, 원 논문의 4096 은 **무한 문맥 LLM 의 로컬 캐시
크기**로 정해진 값이지 분절 성능을 위해 조율된 값이 아니다.

dev-3 문서는 1.2k~6.2k 토큰이라 4096 토큰 창이 **과거의 93~100% 를 덮는다**
(`run_sm_dev3.py` 실측). 그러면 이동 윈도우가 전체 과거와 구분되지 않아
**적응적 임계가 사실상 고정 임계로 퇴화**한다.

⇒ 스윕의 목적은 EM-LLM 을 이기는 것이 아니라 **"이식 과정에서 원 방법을 불리하게 만들지
않았음"을 보이는 것**이다. 외부 참조점이 이것 하나뿐이라 특히 중요하다.

## 보고 규약
- **주 표에는 원본 설정(γ=1.1, n_local=4096)만** 싣는다.
- 스윕은 **부록 한 줄**로. 예: *"본 데이터의 문서 길이에서는 이 창이 과거의 93~100% 를 덮어
  이동 윈도우가 전체 과거와 거의 구분되지 않으므로, n_local ∈ {256,…,4096} 민감도를
  부록 X 에 제시한다. 최고 성능은 n_local = Y 에서 Z 였다."*
- 🔴 **스윕 최고값을 주 표에 올리지 않는다** — 그건 dev 튜닝이고, 우리가 금지한 것이다.

    .venv/bin/python EACL/methods/emllm/sweep_nlocal.py
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emllm.run_sm_dev3 import CACHE, GAMMA, WARMUP, sentences_of, sm_boundaries  # noqa: E402

OUT = REPO / "outputs/experiments/2026-08-03_emllm_sm_dev3"
DOCS = ("pieman", "tunnel", "monkey")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-locals", default="256,512,1024,2048,4096")
    ap.add_argument("--gamma", type=float, default=GAMMA, help="🔒 고정 — 스윕 대상 아님")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from baseline_compare import load_dataset          # noqa: PLC0415
    from preds_table import score_of                   # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    data = load_dataset("dev3")
    docs = [d for d, _s, _g in data]
    G = [g for _d, _s, g in data]

    cached = {}
    for doc in DOCS:
        z = np.load(CACHE / f"{doc}__emllm.npz", allow_pickle=True)
        s = np.asarray(z["surprise"], dtype=float)
        sents = sentences_of(doc)
        m = min(len(sents), len(s))
        cached[doc] = (s[:m],
                       np.array([len(tok.encode(x, add_special_tokens=False))
                                 for x in sents[:m]]))

    print(f"γ={args.gamma} (고정) · warmup={WARMUP} · SM+ 제외 · dev-3")
    print(f"{'n_local':>8}{'Score':>9}{'예측경계':>9}{'창 커버리지':>12}")
    rows = {}
    for nl in [int(x) for x in args.n_locals.split(",")]:
        preds, npred, covs = {}, 0, []
        for doc in DOCS:
            s, tl = cached[doc]
            bnd = sm_boundaries(s, tl, args.gamma, nl, WARMUP)
            gold = dict(zip(docs, G))[doc]
            vec = [0] * len(gold)
            for b in bnd:
                if 1 <= b <= len(vec):
                    vec[b - 1] = 1
            preds[doc] = vec
            npred += sum(vec)
            cum = np.cumsum(tl)
            covs.append(float(np.mean([min(nl, c) / max(1, c) for c in cum[1:]])))
        sc = score_of([np.asarray(preds[d], np.int8) for d in docs], G)["score"]
        rows[nl] = {"score": sc, "n_pred": npred, "coverage": round(float(np.mean(covs)), 3)}
        print(f"{nl:>8}{sc:>9.4f}{npred:>9}{np.mean(covs):>12.2f}")

    best = max(rows, key=lambda k: rows[k]["score"])
    base = rows.get(4096, {}).get("score")
    print(f"\n원본 설정(n_local=4096) = {base:.4f}")
    print(f"스윕 최고 = n_local {best} 에서 {rows[best]['score']:.4f}"
          f"  (원본 대비 {rows[best]['score'] - base:+.4f})")
    print("🔴 최고값을 주 표에 올리지 않는다 — 주 표는 원본 설정 고정, 스윕은 부록.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep_nlocal.json").write_text(json.dumps(
        {"gamma": args.gamma, "rows": rows, "baseline_n_local": 4096,
         "best_n_local": best}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
