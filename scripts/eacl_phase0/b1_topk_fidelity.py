#!/usr/bin/env python3
"""B1 을 폐쇄 API 로 돌릴 수 있는가 — **top-k 절단이 Bayesian surprise 를 보존하는지** 실측.

배경. B1 의 원 정의는 인접 두 위치의 **전체 어휘 분포** 사이 KL 이다. 폐쇄 API(GPT-4.1)는
top-20 logprobs 만 준다. 그래서 API 사용자가 실제로 계산할 수 있는 것은
  KL_topk = KL( p|_S 정규화 || q|_S 정규화 ),  S = top-k(p) ∪ top-k(q)
뿐이다. 이 근사가 원 정의를 보존하는지 **전체 로짓을 가진 로컬 모델에서** 직접 잰다.

이 스크립트는 판정만 한다 — 보존되면 API B1 이 정당해지고, 안 되면 "실행 불가" 의 근거가 된다.

사용:
  PYTHONPATH=<tf5> .venv/bin/python scripts/eacl_phase0/b1_topk_fidelity.py \
      outputs/eacl_phase0/pieman_michelmann_master.jsonl --model /shared/models/Qwen3.6-35B-A3B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import b1_bayes_surprise as B  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

KS = (5, 10, 20, 100, 1000)


def kl_full(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(torch.sum(p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).item())


def kl_topk(p: torch.Tensor, q: torch.Tensor, k: int) -> float:
    """API 사용자가 실제로 계산 가능한 값: 두 top-k 의 합집합 위에서 각각 재정규화한 KL."""
    idx = torch.unique(torch.cat([p.topk(k).indices, q.topk(k).indices]))
    ps, qs = p[idx], q[idx]
    ps = ps / ps.sum()
    qs = qs / qs.sum()
    return float(torch.sum(ps * (torch.log(ps + 1e-12) - torch.log(qs + 1e-12))).item())


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def newsent_max(trans, spans, n_trans):
    """b1_deployable.b1_newsent_max 와 동일 readout — 새 문장 첫 구간의 최대 transient."""
    first = {}
    for gi, (_a, _b, si, _wp) in enumerate(spans):
        first.setdefault(si, gi)
    out = []
    for t in range(1, n_trans + 1):
        a = first.get(t)
        b = first.get(t + 1, len(trans))
        seg = trans[a:b] if a is not None else []
        out.append(float(max(seg)) if seg else 0.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("master", type=Path)
    ap.add_argument("--model", default="/shared/models/Qwen3.6-35B-A3B")
    ap.add_argument("--drift", type=float, default=0.10)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/experiments/2026-07-31_b1_model_scale/topk_fidelity.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = B.read(args.master)
    doc = rows[0]["doc_id"]
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()

    sentences = B.reconstruct_sentences(rows)
    full, spans = B.build_words(sentences)
    dists, _ = B.word_endpoint_dists(model, tok, full, spans, args.ctx, args.stride, "cuda")

    raw = {"full": [0.0], **{f"top{k}": [0.0] for k in KS}}
    for i in range(1, len(dists)):
        p, q = dists[i], dists[i - 1]
        raw["full"].append(max(kl_full(p, q), 0.0))
        for k in KS:
            raw[f"top{k}"].append(max(kl_topk(p, q, k), 0.0))

    n_tr = len(rows)
    gold = [0] * (n_tr + 1)
    for r in rows:
        if int(r["boundary"]):
            gold[int(r["transition"]) - 1] = 1

    res = {"doc": doc, "model": args.model, "n_words": len(dists), "variants": {}}
    sigs = {}
    for name, rw in raw.items():
        tr = B.transient_from_raw(rw, args.drift)
        sigs[name] = newsent_max(tr, spans, n_tr)

    base_w = np.array(raw["full"][1:])
    for name in raw:
        s = np.array(sigs[name])
        thr = s.mean() + 1.0 * s.std()
        pred = [0] * (n_tr + 1)
        for t, v in enumerate(s):
            if v >= thr:
                pred[t] = 1
        sc = score_dialogues([gold], [pred])
        res["variants"][name] = {
            "word_kl_spearman_vs_full": spearman(base_w, np.array(raw[name][1:])),
            "sent_signal_spearman_vs_full": spearman(np.array(sigs["full"]), s),
            "score_c1.0": sc["score"], "n_pred": int(sum(pred)),
            "mean_word_kl": float(np.mean(raw[name][1:])),
        }

    print(f"{'변형':>8s} {'단어KL ρ':>10s} {'문장신호 ρ':>11s} {'Score(c=1)':>11s} {'경계':>5s} {'평균KL':>9s}")
    for name, v in res["variants"].items():
        print(f"{name:>8s} {v['word_kl_spearman_vs_full']:10.4f} "
              f"{v['sent_signal_spearman_vs_full']:11.4f} {v['score_c1.0']:11.4f} "
              f"{v['n_pred']:5d} {v['mean_word_kl']:9.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
