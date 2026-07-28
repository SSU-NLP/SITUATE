#!/usr/bin/env python3
"""무과금 go/no-go 관문: 오라클 유형 화제체인이 유사도가 놓친 evidence 세션을
실제로 커버하는가 — 그리고 그게 '동일크기 랜덤 체인'을 넘는가(=구조가 이득의 원천인가).

gonogo §4 통제(구조 vs 크기)를 QA 없이 구조적으로 판정.
멀티홉(cross-session) 질문 대상. LLM 호출 0 (임베딩만 로컬).

usage:
  uv run python scripts/exp_typed/04_coverage_gate.py --eps outputs/exp_typed/episodes_typed.jsonl --K 3
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, embed, read_jsonl

LOCOMO = REPO / "benchmarks/locomo/locomo10.json"


def sess_ids(ev):
    s = set()
    for e in ev or []:
        m = re.match(r"D(\d+):", str(e))
        if m:
            s.add(int(m.group(1)) - 1)
    return s


class UF:
    def __init__(s, n): s.p = list(range(n))
    def f(s, x):
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)


def chain_components(emap):
    ids = sorted(emap)
    uf = UF(max(ids) + 1)
    for i in ids:
        e = emap[i]
        if e["ttype"] in ("continue", "elaborate") and (i - 1) in emap:
            uf.u(i, i - 1)
        rt = e.get("return_target")
        if e["ttype"] == "return" and rt is not None and rt in emap:
            uf.u(i, rt)
    comp = defaultdict(set)
    for i in ids:
        comp[uf.f(i)].add(i)
    return {i: comp[uf.f(i)] for i in ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default=str(REPO / "outputs/exp_typed/episodes_typed.jsonl"))
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--cat", type=int, default=1, help="LoCoMo 카테고리(1=multi-hop)")
    ap.add_argument("--rand-draws", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(LOCOMO.read_text())
    eps = read_jsonl(args.eps)
    by_conv = defaultdict(dict)
    for e in eps:
        by_conv[e["conv_id"]][e["ep_id"]] = e

    from collections import Counter
    print(f"[eps] {args.eps}")
    print("  type dist:", dict(Counter(e["ttype"] for e in eps)),
          " return_valid:", sum(1 for e in eps if e["ttype"] == "return" and e.get("return_target") is not None))

    rng = random.Random(args.seed)
    n = miss = orc = 0
    rnd = 0.0
    sizes = []
    for idx, s in enumerate(data):
        cid = str(s.get("sample_id", idx))
        emap = by_conv[cid]
        ids = sorted(emap)
        n2c = chain_components(emap)
        ev_mat = embed([emap[i]["text"][:2000] for i in ids])
        qs = [q for q in s.get("qa", []) if q.get("category") == args.cat]
        if not qs:
            continue
        qv = embed([q["question"] for q in qs])
        for qi, q in enumerate(qs):
            need = sess_ids(q.get("evidence")) & set(ids)
            if len(need) < 2:
                continue
            n += 1
            sims = ev_mat @ qv[qi]
            topK = set(np.array(ids)[np.argsort(-sims)[:args.K]])
            missed = need - topK
            if not missed:
                continue
            miss += 1
            reach = set().union(*[n2c[t] for t in topK])
            sizes.append(len(reach))
            if missed & reach:
                orc += 1
            pool = [i for i in ids if i not in topK]
            want = max(0, len(reach) - len(topK))
            hits = 0
            for _ in range(args.rand_draws):
                extra = set(rng.sample(pool, min(len(pool), want)))
                if missed & (topK | extra):
                    hits += 1
            rnd += hits / args.rand_draws

    print(f"\n=== 커버율 관문 (cat={args.cat}, K={args.K}) ===")
    print(f"cross-session 질문: {n},  유사도가 evidence 놓침(≥1): {miss}")
    print(f"  오라클 타입체인 ≥1 커버 : {orc} ({100*orc/max(1,miss):.0f}%)")
    print(f"  동일크기 랜덤체인 ≥1 커버: {rnd:.0f} ({100*rnd/max(1,miss):.0f}%)")
    print(f"  타입 순효과(오라클−랜덤) : {100*(orc-rnd)/max(1,miss):+.1f}%p")
    print(f"  seed 도달세션 평균/전체  : {np.mean(sizes):.1f}")


if __name__ == "__main__":
    main()
