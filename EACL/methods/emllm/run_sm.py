#!/usr/bin/env python3
"""EM-LLM SM 변형을 **6편 전체**(dev3 + test3)에 돌린다.

`run_sm_dev3.py` 의 dev3 전용판을 일반화한 것. 규칙·하이퍼는 동일:
`s_t > μ(win) + γ·σ(win)`, γ=1.1 · n_local=4096 토큰 고정, **SM+ 제외**.
문장 목록은 `baseline_compare.load_dataset` 에서 받아 dev3/test3 를 같은 경로로 처리한다
(dev3 는 `*_michelmann_master`, test3 는 `*_eventrecall_goldtopn_master` 로 파일이 다르다).

    .venv/bin/python EACL/methods/emllm/run_sm.py
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

from emllm.run_sm_dev3 import CACHE, GAMMA, N_LOCAL, WARMUP, sm_boundaries  # noqa: E402

PRED = REPO / "outputs/experiments/2026-07-29_baseline_compare/preds"
OUT = REPO / "outputs/experiments/2026-08-03_emllm_sm_dev3"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dev3,test3")
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--n-local", type=int, default=N_LOCAL)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    # D-07: surprise 를 뽑은 LM 과 n_local 을 세는 토크나이저는 **반드시 같아야** 한다
    # (n_local 은 그 LM 자신의 토큰 수). 캐시 태그를 바꿀 때 --tokenizer 도 같이 바꾼다.
    ap.add_argument("--cache-tag", default="",
                    help="surprise 캐시 네임스페이스. 빈 값이면 {doc}__emllm.npz(Qwen 기본), "
                         "값이 있으면 {doc}__emllm_{tag}.npz")
    ap.add_argument("--label", default="EM-LLM-SM",
                    help="예측 덤프·표에 쓸 행 이름")
    args = ap.parse_args()
    suffix = f"_{args.cache_tag}" if args.cache_tag else ""

    from transformers import AutoTokenizer

    from baseline_compare import load_dataset            # noqa: PLC0415
    from hi_ontop.dts_scoring import score_dialogues     # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"γ={args.gamma} (고정) · n_local={args.n_local}토큰 · SM+ 제외")

    allrows = {}
    for ds in args.datasets.split(","):
        data = load_dataset(ds)
        preds, missing = {}, []
        for doc, sents, gold in data:
            f = CACHE / f"{doc}__emllm{suffix}.npz"
            if not f.exists():
                missing.append(doc)
                continue
            s = np.asarray(np.load(f, allow_pickle=True)["surprise"], dtype=float)
            m = min(len(sents), len(s))
            tl = np.array([len(tok.encode(x, add_special_tokens=False)) for x in sents[:m]])
            bnd = sm_boundaries(s[:m], tl, args.gamma, args.n_local, WARMUP)
            vec = [0] * len(gold)
            for b in bnd:
                if 1 <= b <= len(vec):
                    vec[b - 1] = 1
            preds[doc] = vec
        if missing:
            print(f"  🔴 {ds}: surprise 캐시 없음 {missing} — 이 데이터셋 건너뜀")
            continue

        docs = [d for d, _s, _g in data]
        G = [g for _d, _s, g in data]
        ys = [np.concatenate([np.asarray(g, np.int8), [0]]).tolist() for g in G]
        ps = [np.concatenate([np.asarray(preds[d][:len(g)], np.int8), [0]]).tolist()
              for d, g in zip(docs, G)]
        sc = score_dialogues(ys, ps)
        npred = sum(sum(v) for v in preds.values())
        ngold = int(sum(int(g.sum()) for g in G))
        PRED.mkdir(parents=True, exist_ok=True)
        (PRED / f"{args.label}_{ds}.json").write_text(json.dumps(preds))
        allrows[ds] = {"pk": sc["pk"], "wd": sc["wd"], "f1": sc["f1"], "score": sc["score"],
                       "n_pred": npred, "n_gold": ngold}
        print(f"  {args.label} {ds:<6} Pk {sc['pk']:.4f} WD {sc['wd']:.4f} "
              f"F1 {sc['f1']:.4f} Score {sc['score']:.4f} · 예측 {npred}/gold {ngold}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"sm_6docs{suffix}.json").write_text(json.dumps(
        {"config": vars(args), "rows": allrows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
