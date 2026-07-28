#!/usr/bin/env python3
"""EM-LLM 을 3개 DTS(tiage/dialseg711/superseg)에서 공식 SuperDialseg SCORE 로 실측.

- 데이터: Def-DTS 번들 jsonl (turn=[NEWLINE], 세그먼트경계=[BOUNDARY]). 끝-turn 규약.
- 채점기: 공식 ``SegmentationEvaluation`` (validate_official_scorer.load_official_scorer).
  Score = 0.5*F1 + 0.25*(1-Pk) + 0.25*(1-WD), 대화별 매크로 평균.
- EM-LLM: segment_manual_emllm 의 encode_doc/surprise_boundaries/refine_boundaries 재사용.
  boundary=새 세그먼트 첫 turn → 끝-turn 규약 -1 매핑 (pred[b-1]=1).
- 배선검증(--sanity): nltk TextTiling 으로 논문값(tiage .363/dialseg711 .382/superseg .471) 재현.

usage:
  uv run python scripts/run_emllm_dts.py --sanity          # TextTiling 배선검증
  uv run python scripts/run_emllm_dts.py --datasets tiage  # EM-LLM 실측
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

DTS = REPO / "benchmarks/Def-DTS/data/DTS_session_datasets"
from validate_official_scorer import load_official_scorer  # noqa: E402


def load_defdts(ds, split="test"):
    """Def-DTS jsonl → list of (utts, yt) 끝-turn 라벨."""
    import json
    out = []
    for line in (DTS / f"{ds}_{split}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        dlg = json.loads(line)["dialogue"]
        utts, yt = [], []
        for tok in dlg.split("[NEWLINE]"):
            t = tok.strip()
            if not t:
                continue
            if t == "[BOUNDARY]":
                if yt:
                    yt[-1] = 1
                continue
            utts.append(t)
            yt.append(0)
        if len(utts) >= 2:
            yt[-1] = 0
            out.append((utts, yt))
    return out


def boundaries_to_endturn(boundaries, n):
    """새 세그먼트 첫 turn 인덱스 → 끝-turn 규약 이진 pred (pred[b-1]=1)."""
    pred = [0] * n
    for b in boundaries:
        if 1 <= b < n:
            pred[b - 1] = 1
    if pred:
        pred[-1] = 0
    return pred


def run_sanity(datasets):
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.tokenize import TextTilingTokenizer
    from validate_official_scorer import texttiling_predict
    SegEval = load_official_scorer()
    tt = TextTilingTokenizer(w=10, k=6)
    PAPER = {"tiage": 0.363, "dialseg711": 0.382, "superseg": 0.471}
    print("=== 배선검증: TextTiling (Def-DTS 데이터, 공식 채점기) ===")
    for ds in datasets:
        ev = SegEval(window_size="auto")
        miss = 0
        for utts, yt in load_defdts(ds):
            try:
                pred = texttiling_predict(tt, utts)
            except Exception:
                miss += 1
                continue
            if len(pred) != len(yt):
                miss += 1
                continue
            yt2, pred = list(yt), list(pred)
            yt2[-1] = 0
            pred[-1] = 0
            ev.add(yt2, pred)
        r = ev.compute()
        p = PAPER.get(ds)
        print(f"{ds:11s}: F1={r['f1(binary)']:.4f} Pk={r['pk']:.4f} WD={r['windowdiff']:.4f} "
              f"Score={r['total_score']:.4f} (#dlg={r['#sample']}, miss={miss}) | paper≈{p}")


def run_emllm(datasets, gamma, metrics):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from segment_manual_emllm import (DEFAULT_LM, encode_doc, refine_boundaries,
                                       surprise_boundaries)
    SegEval = load_official_scorer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LM 로딩] {DEFAULT_LM} ({device}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_LM)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_LM, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    print(f"\n=== EM-LLM DTS 실측 (γ={gamma}, metrics={metrics}) ===")
    hdr = f"{'dataset':11s} {'method':12s} {'F1':>7} {'Pk':>7} {'WD':>7} {'Score':>7} {'#dlg':>6}"
    print(hdr)
    results = {}
    for ds in datasets:
        dialogs = load_defdts(ds)
        evs = {m: SegEval(window_size="auto") for m in metrics}
        t0 = time.perf_counter()
        for di, (utts, yt) in enumerate(dialogs):
            surprise, A = encode_doc(utts, tok, model, device, refine_from_layer=0)
            events = surprise_boundaries(surprise, gamma, win=0, warmup=3)
            for m in metrics:
                if m == "surprise" or not events:
                    b = list(events)
                else:
                    b = refine_boundaries(events, A, metric=m, min_size=1)
                b = sorted({int(x) for x in b if 0 < int(x) < len(utts)})
                pred = boundaries_to_endturn(b, len(utts))
                yt2 = list(yt); yt2[-1] = 0
                evs[m].add(yt2, pred)
            if (di + 1) % 100 == 0:
                print(f"  [{ds}] {di+1}/{len(dialogs)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        for m in metrics:
            r = evs[m].compute()
            results[(ds, m)] = r
            name = {"surprise": "EM surprise", "modularity": "EM +modul", "conductance": "EM +conduct"}[m]
            print(f"{ds:11s} {name:12s} {r['f1(binary)']:>7.4f} {r['pk']:>7.4f} "
                  f"{r['windowdiff']:>7.4f} {r['total_score']:>7.4f} {r['#sample']:>6}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["tiage", "dialseg711", "superseg"])
    ap.add_argument("--gamma", type=float, default=1.1)
    ap.add_argument("--metrics", nargs="+", default=["surprise", "modularity", "conductance"])
    args = ap.parse_args()
    if args.sanity:
        run_sanity(args.datasets)
    else:
        run_emllm(args.datasets, args.gamma, args.metrics)


if __name__ == "__main__":
    main()
