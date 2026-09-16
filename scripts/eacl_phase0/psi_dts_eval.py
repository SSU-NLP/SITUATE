#!/usr/bin/env python3
"""D-B62 — ψ 재개봉 조건들의 **DTS 대화 벤치마크 전이 테스트**.

🔴 지위: **전이 확인이지 판정 근거가 아니다.** ψ 채택 여부는 앵커 val OOF 게이트가 이미 닫았다
   (8칸 전부 미달, REPORT.md §2.4). DTS 는 D-M5 에서 **scope 밖** 선언 — 성능 주장에 쓰지 않고,
   여기 숫자를 보고 조건을 다시 고르지 않는다.

🔴 모델은 **서사 픽션 앵커로만** 학습됐다(대화 재학습·재라벨 없음). 입력은 얼린 mpnet 임베딩뿐.
   DTS 공식 test 전체는 D-B25(2026-07-28)에서 이미 개봉된 자료다.

경계 규칙 둘 다 병기 — DTS 는 gold 밀도가 앵커(6.4%)와 2~4배 다르므로 **적응임계가 메인**이다
(D-B25 ②, D-B39 C_MAIN=1.0). 고정분위수는 앵커 밀도 사전을 그대로 쓴 참고값.
지표는 전부 동결 구현 호출: Score · BOR · purity · coverage · F(p,c) · L1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts/eacl_phase0")]
from psi_dev3_eval import COND, COND_CHAINQ, load, row_metrics  # noqa: E402
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402
import train_direct_boundary as m8  # noqa: E402

OUT = REPO / "outputs/experiments/psi_reopen_2026-08-01"
# DTS 공식 test 전체(D-B25) + AMI. 소표본판(dts_tiage 등)은 이들의 부분집합이라 생략.
SETS = ["dts_full_tiage", "dts_full_dialseg711", "dts_full_superseg", "dts_ami"]


def conds(book: bool) -> list[tuple[str, str, str]]:
    """(split 표기, 조건 표기, ns) 목록."""
    out = []
    for split, base in (("i2", COND), ("chainq", COND_CHAINQ)):
        for name, ns in base:
            out.append((split, name, ns.replace("_pwone", "_gsbook_pwone") if book else ns))
    return out


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    book = len(sys.argv) > 2 and sys.argv[2] == "book"
    tag = "_book" if book else ""

    corpora = {}
    for ds in SETS:
        c = Corpus(ds)
        G = [c.gold[int(c.to[i]):int(c.to[i + 1])].astype(np.int8) for i in range(c.n_docs)]
        corpora[ds] = (c, G, int(sum(int(g.sum()) for g in G)))
        z = [np.zeros(len(g), np.int8) for g in G]
        print(f"[{ds}] 대화 {c.n_docs} · 전이 {sum(len(g) for g in G)} · gold {corpora[ds][2]} "
              f"(밀도 {np.concatenate(G).mean():.4f}) · 무경계 하한 "
              f"{row_metrics(z, G, corpora[ds][2])['score']:.4f}", flush=True)
    print()

    out: dict[str, dict] = {}
    for split, name, ns in conds(book):
        m, _ = load(ns, seed, dev)
        for ds in SETS:
            c, G, n_gold = corpora[ds]
            with torch.no_grad():
                P = axis_probs(m, c, list(range(c.n_docs)), dev, bs=8, with_anchor=False)
            S = [x.sum(1) for x in P]
            for rule, fn in (("적응임계", m8.cuts_adaptive), ("고정분위수", m8.cuts_pct)):
                Pb = [fn(s)[:len(g)] for s, g in zip(S, G)]
                r = row_metrics(Pb, G, n_gold)
                out[f"{ds}|{rule}|{split}|{name}"] = r
                print(f"{ds:20s} {rule:6s} {split:6s} {name:30s} "
                      f"Score {r['score']:.4f} BOR {r['bor']:5.2f} pur {r['purity']:.3f} "
                      f"cov {r['coverage']:.3f} F {r['f_pc']:.3f} L1 {r['L1']:5.2f}", flush=True)
        del m
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"DTS_PSI{tag}_seed{seed}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {OUT / f'DTS_PSI{tag}_seed{seed}.json'}")


if __name__ == "__main__":
    main()
