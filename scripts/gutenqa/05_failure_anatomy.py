#!/usr/bin/env python3
"""Phase A 진단 — A1 청크 길이 분포 · A2 실패 문항 해부 · A3 길이 OOD (새 실행 없음, 재분석만).

## A2 가 가능한 이유
`compute_DCG` 는 relevance one-hot 이므로 `dcg@k = 1/log2(rank+1)`(rank 1-based)이다.
⇒ **저장된 DCG 값에서 순위를 역산**할 수 있다: `rank = 2^(1/dcg) - 1`.
그리고 `gold_in_some_chunk` 플래그가 이미 있으므로 다음을 정확히 분리할 수 있다:

| 구분 | 조건 | 의미 |
|---|---|---|
| ceiling 실패 | `gold_in_some_chunk == False` | gold 가 어떤 청크에도 없음 (방법 무관, 데이터셋 문제) |
| **순위 실패** | `gold_in_some_chunk == True and dcg@20 == 0` | **정답 청크는 존재하는데 top-20 밖으로 밀림** |
| 성공 | `dcg@20 > 0` | rank ≤ 20 |

산문 5종 ceiling 이 82% 대로 동일하므로 방법 간 격차는 **전부 순위 실패 차이**다. 이 스크립트는
그 순위 실패 문항의 정답 청크를 직접 열어 속성을 본다.

## 측정 항목
- **A1** 방법별 청크 단어수 p5/p50/p95/max · CV · 상한초과(>800단어) · 하한미달(<50단어) 비율
- **A2** 순위 실패 vs 성공 문항의 정답 청크 속성:
  ① 청크 단어수 ② 그 방법 자기 분포 내 백분위 ③ **희석률** = gold 단어수 / 청크 단어수
  ④ 같은 문항에서 LumberChunker 는 성공하는데 M5 는 실패하는 경우의 대조
- **A3** 책별 DCG@20 × 책 문장수 상관 (Spearman). 학습 길이 상한 900문장 이내 도서 수도 보고.

    uv run python scripts/gutenqa/05_failure_anatomy.py
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"
DATA = BASE / "data"
RET = BASE / "retrieval"

TRAIN_MAX_SENT = 900          # 앵커 코퍼스 길이 상한 (build_pg_shorts.py --max-sent)
BIG_WORDS = 800               # A1 상한초과 기준
SMALL_WORDS = 50              # A1 하한미달 기준
FOCUS = ["lumberchunker", "recursive", "paragraph", "semantic",
         "m5__rate0.15", "m5__frozen", "m5__rate0.3", "m5__rate0.5"]


def rank_from_dcg(d: float) -> float:
    """dcg = 1/log2(rank+1) 역산. 0 이면 top-k 밖 → inf."""
    if d <= 0:
        return math.inf
    return 2 ** (1.0 / d) - 1


def load_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "PHASE_A_DIAGNOSIS.md"))
    a = ap.parse_args()

    qs = {q["qid"]: q for q in load_jsonl(DATA / "questions.jsonl")}
    subs = json.loads((BASE / "book_subsets.json").read_text(encoding="utf-8"))
    n_sent = {}
    for f in glob.glob(str(BASE / "m5_scores/*.npz")):
        bid = int(re.search(r"book(\d+)", f).group(1))
        n_sent[bid] = int(np.load(f)["n_sent"])

    chunks = {}
    for name in FOCUS:
        p = DATA / f"chunks/{name}.jsonl"
        if p.exists():
            chunks[name] = {r["book_id"]: r["chunks"] for r in load_jsonl(p)}
    results = {}
    for name in FOCUS:
        p = RET / f"{name}.json"
        if p.exists():
            results[name] = json.loads(p.read_text(encoding="utf-8"))

    L = ["# Phase A 진단 — 길이 분포 · 실패 문항 해부 · 길이 OOD\n",
         "기존 산출물만 재분석. 새 실행·새 LLM 호출 없음.\n"]

    # ================= A1 =================
    L += ["## A1. 청크 길이 분포\n",
          f"CV = 표준편차/평균. 상한초과 = >{BIG_WORDS}단어, 하한미달 = <{SMALL_WORDS}단어.\n",
          "| 방법 | n청크 | 평균 | p5 | p50 | p95 | max | CV | >800단어 | <50단어 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    a1 = {}
    for name in FOCUS:
        if name not in chunks:
            continue
        w = np.array([len(c.split()) for v in chunks[name].values() for c in v], dtype=float)
        st = {"n": int(w.size), "mean": w.mean(), "p5": np.percentile(w, 5),
              "p50": np.percentile(w, 50), "p95": np.percentile(w, 95), "max": w.max(),
              "cv": w.std() / w.mean(), "big": (w > BIG_WORDS).mean(),
              "small": (w < SMALL_WORDS).mean()}
        a1[name] = st
        L.append(f"| {name} | {st['n']:,} | {st['mean']:.0f} | {st['p5']:.0f} | {st['p50']:.0f} | "
                 f"{st['p95']:.0f} | {st['max']:.0f} | **{st['cv']:.2f}** | "
                 f"{st['big']*100:.1f}% | {st['small']*100:.1f}% |")

    # ================= A2 =================
    L += ["\n## A2. 실패 문항 해부 (순위 실패만)\n"]
    a2 = {}
    for name in FOCUS:
        if name not in results or name not in chunks:
            continue
        recs = results[name]["records"]
        wdist = np.array([len(c.split()) for v in chunks[name].values() for c in v], dtype=float)
        rows_ok, rows_bad = [], []
        for r in recs:
            if not r["gold_in_some_chunk"]:
                continue                                     # ceiling 실패는 제외
            q = qs[r["qid"]]
            g = q["gold"].lower()
            gw = len(q["gold"].split())
            # gold 를 포함하는 청크(들) — 가장 작은 것을 대표로
            cand = [c for c in chunks[name][r["book_id"]] if g in c.lower()]
            if not cand:
                continue
            cw = min(len(c.split()) for c in cand)
            item = {"qid": r["qid"], "chunk_words": cw,
                    "pct": float((wdist < cw).mean()),
                    "dilution": gw / max(1, cw),
                    "rank": rank_from_dcg(r["dcg@20"])}
            (rows_bad if r["dcg@20"] == 0 else rows_ok).append(item)
        n_all = len(rows_ok) + len(rows_bad)
        if not n_all:
            continue
        f = lambda rs, k: float(np.median([x[k] for x in rs])) if rs else float("nan")  # noqa: E731
        a2[name] = {
            "n_rank_eval": n_all, "n_rank_fail": len(rows_bad),
            "rank_fail_share": len(rows_bad) / n_all,
            "med_words_fail": f(rows_bad, "chunk_words"), "med_words_ok": f(rows_ok, "chunk_words"),
            "med_pct_fail": f(rows_bad, "pct"), "med_pct_ok": f(rows_ok, "pct"),
            "med_dilution_fail": f(rows_bad, "dilution"), "med_dilution_ok": f(rows_ok, "dilution"),
            "med_rank_ok": f(rows_ok, "rank"),
        }
    L += ["gold 이 어딘가엔 있는 문항만 대상. `순위실패` = 정답 청크가 존재하는데 top-20 밖.\n",
          "| 방법 | 대상문항 | 순위실패 | 실패율 | 정답청크 단어수(실패/성공) | 자기분포 백분위(실패/성공) | 희석률(실패/성공) |",
          "|---|---|---|---|---|---|---|"]
    for name, s in a2.items():
        L.append(f"| {name} | {s['n_rank_eval']:,} | {s['n_rank_fail']:,} | "
                 f"**{s['rank_fail_share']*100:.1f}%** | "
                 f"{s['med_words_fail']:.0f} / {s['med_words_ok']:.0f} | "
                 f"{s['med_pct_fail']*100:.0f}% / {s['med_pct_ok']*100:.0f}% | "
                 f"{s['med_dilution_fail']:.4f} / {s['med_dilution_ok']:.4f} |")

    # A2-④ 같은 문항 대조: LumberChunker 성공 & M5 실패
    L += ["\n### A2-④ 같은 문항에서 LumberChunker 성공 · M5 실패\n"]
    lc = {r["qid"]: r for r in results["lumberchunker"]["records"]}
    for m5 in [n for n in FOCUS if n.startswith("m5__")]:
        if m5 not in results:
            continue
        mm = {r["qid"]: r for r in results[m5]["records"]}
        both = [qid for qid in mm
                if mm[qid]["gold_in_some_chunk"] and lc.get(qid, {}).get("gold_in_some_chunk")]
        m5_fail_lc_ok = [qid for qid in both if mm[qid]["dcg@20"] == 0 and lc[qid]["dcg@20"] > 0]
        m5_ok_lc_fail = [qid for qid in both if mm[qid]["dcg@20"] > 0 and lc[qid]["dcg@20"] == 0]

        def prof(qids, name):
            ws, dl = [], []
            for qid in qids:
                q = qs[qid]
                g = q["gold"].lower()
                cand = [c for c in chunks[name][q["book_id"]] if g in c.lower()]
                if cand:
                    cw = min(len(c.split()) for c in cand)
                    ws.append(cw)
                    dl.append(len(q["gold"].split()) / max(1, cw))
            return (float(np.median(ws)) if ws else float("nan"),
                    float(np.median(dl)) if dl else float("nan"))

        w_m5, d_m5 = prof(m5_fail_lc_ok, m5)
        w_lc, d_lc = prof(m5_fail_lc_ok, "lumberchunker")
        L.append(f"**{m5}** — 공통 대상 {len(both):,}문항 · "
                 f"M5실패/LC성공 **{len(m5_fail_lc_ok):,}** · M5성공/LC실패 {len(m5_ok_lc_fail):,}")
        L.append(f"- 그 {len(m5_fail_lc_ok):,}문항의 정답청크 단어수 중앙값: "
                 f"M5 {w_m5:.0f} vs LumberChunker {w_lc:.0f}")
        L.append(f"- 희석률 중앙값: M5 {d_m5:.4f} vs LumberChunker {d_lc:.4f}\n")

    # ================= A3 =================
    L += ["## A3. 길이 OOD — 책 길이 × DCG@20\n"]
    from scipy.stats import spearmanr
    in_domain = [b for b, n in n_sent.items() if n <= TRAIN_MAX_SENT]
    L.append(f"앵커 학습 길이 상한 = **{TRAIN_MAX_SENT}문장**. "
             f"GutenQA 100편 중 그 이내 = **{len(in_domain)}편** "
             f"(book_id {in_domain}, 문장수 {[n_sent[b] for b in in_domain]}).")
    L.append(f"⇒ **in-domain 표본이 {len(in_domain)}편뿐이라 OOD 여부를 부분집합 대조로는 검정할 수 없다.** "
             "길이 연속 상관으로만 본다.\n")
    L += ["| 방법 | Spearman ρ(문장수, 책별 DCG@20) | p |", "|---|---|---|"]
    a3 = {}
    for name in FOCUS:
        if name not in results:
            continue
        per = {}
        for r in results[name]["records"]:
            per.setdefault(r["book_id"], []).append(r["dcg@20"])
        bids = [b for b in sorted(per) if b in n_sent]
        x = [n_sent[b] for b in bids]
        y = [float(np.mean(per[b])) for b in bids]
        rho, p = spearmanr(x, y)
        a3[name] = {"rho": float(rho), "p": float(p), "n_books": len(bids)}
        L.append(f"| {name} | {rho:+.3f} | {p:.2g} |")
    L.append("\n🔴 상관이 M5 에서만 강하게 음수면 길이 특이성, 전 방법 공통이면 벤치마크 성질이다.\n")

    (BASE / "phase_a_diagnosis.json").write_text(
        json.dumps({"A1": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in a1.items()},
                    "A2": a2, "A3": a3,
                    "in_domain_books": in_domain,
                    "book_n_sent": n_sent}, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
