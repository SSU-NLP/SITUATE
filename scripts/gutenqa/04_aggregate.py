#!/usr/bin/env python3
"""GutenQA 결과 집계 → SUMMARY_TABLES.md (D-B39 Phase 3).

`retrieval/*.json` 을 전부 읽어 부분집합 × split 별로 DCG@k / Recall@k / ceiling 을 낸다.
부분집합 정의는 `book_subsets.json` (D-B39 §2 에서 **점수 보기 전 동결**).

## 표 구성
1. 청크 통계 — 방법별 청크 수/책, 평균 단어/청크. **크기 비교 없는 DCG 는 무의미하다.**
2. ceiling — gold 가 어떤 청크에도 통째로 안 들어가 **원천적으로 못 맞히는** 질문 비율.
   작은 청크일수록 불리한 구조적 편향이라 이걸 안 보면 순위 해석이 틀린다.
3. 주표 — 전체 100편, DCG@k / Recall@k
4. 부분집합 — `narrative_fiction`(M5 scope) vs `non_narrative`(도메인 밖) 등
5. 오염 — `contaminated`(2편) vs 나머지 98편

    uv run python scripts/gutenqa/04_aggregate.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"
RET = BASE / "retrieval"
TOPKS = [1, 2, 5, 10, 20]

# 논문 Table 1 발표 수치 (ada-002). **우리 인코더와 다르므로 절대값 비교 불가 — 참조용.**
PAPER_DCG20 = {"lumberchunker": 62.09, "recursive": 54.72, "paragraph": 49.00,
               "propositions": 46.19, "semantic": 44.74}
PAPER_RECALL20 = {"lumberchunker": 77.92}

DISPLAY = {"lumberchunker": "LumberChunker (Gemini)", "recursive": "Recursive (Langchain)",
           "paragraph": "Paragraph", "semantic": "Semantic", "propositions": "Propositions"}


def agg(records: list[dict], ids: set[int], split: str | None = None) -> dict | None:
    rs = [r for r in records if r["book_id"] in ids and (split is None or r["split"] == split)]
    if not rs:
        return None
    out = {"n": len(rs), "ceiling": float(np.mean([r["gold_in_some_chunk"] for r in rs]))}
    for k in TOPKS:
        out[f"dcg@{k}"] = float(np.mean([r[f"dcg@{k}"] for r in rs]))
        out[f"recall@{k}"] = float(np.mean([r[f"recall@{k}"] for r in rs]))
    return out


def fmt_row(label: str, a: dict, extra: str = "") -> str:
    d = "  ".join(f"{a[f'dcg@{k}']*100:5.2f}" for k in TOPKS)
    r = "  ".join(f"{a[f'recall@{k}']*100:5.2f}" for k in TOPKS)
    return f"| {label} | {a['n']} | {a['ceiling']*100:.2f} | {d} | {r} |{extra}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "SUMMARY_TABLES.md"))
    a = ap.parse_args()

    subs = json.loads((BASE / "book_subsets.json").read_text(encoding="utf-8"))
    S = {k: set(v) for k, v in subs["subsets"].items()}

    results = {}
    for p in sorted(RET.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        results[d["name"]] = d
    if not results:
        raise SystemExit(f"🔴 {RET} 에 결과가 없다 — 02_eval_harness.py 를 먼저 돌릴 것")

    base_names = [n for n in ["lumberchunker", "recursive", "paragraph", "semantic", "propositions"]
                  if n in results]
    m5_names = sorted([n for n in results if n.startswith("m5__")],
                      key=lambda n: results[n]["chunk_stats"]["chunks_per_book"])
    order = base_names + m5_names

    L = ["# GutenQA 다운스트림 검색 평가 — 요약표 (D-B39)\n",
         f"- 검색 인코더 = `{next(iter(results.values()))['encoder']}` · "
         f"유사도 = {next(iter(results.values()))['similarity']}",
         "- 지표 정의 = LumberChunker 레포 `Retrieval_Granularities_Eval.ipynb` 그대로 "
         "(relevance = 대소문자 무시 부분문자열 · **첫 히트 one-hot** · DCG 비정규화 · "
         "`Recall@k` 는 실제로는 hit-rate)",
         "- 🔴 논문은 ada-002 를 썼다. **절대값 비교 불가**, 발표 수치는 참조용으로만 병기.\n"]

    # ---- 1. 청크 통계 -------------------------------------------------------
    L += ["## 1. 청크 통계 (크기 비교 없는 DCG 는 무의미)\n",
          "| 방법 | 청크 수 | 청크/책 | 평균 단어/청크 |", "|---|---|---|---|"]
    for n in order:
        c = results[n]["chunk_stats"]
        L.append(f"| {DISPLAY.get(n, n)} | {c['n_chunks']:,} | {c['chunks_per_book']} | "
                 f"{c['mean_words_per_chunk']} |")

    # ---- 2. ceiling ---------------------------------------------------------
    L += ["\n## 2. ceiling — 원천적으로 못 맞히는 질문\n",
          "gold 부분문자열이 **어떤 청크에도 통째로 들어있지 않으면** 어떤 검색기로도 0점이다. "
          "청크가 작을수록 gold 가 쪼개질 확률이 커진다 ⇒ **이 벤치마크는 큰 청크에 구조적으로 유리하다.** "
          "논문 Table 1 은 이 값을 보여주지 않는다.\n",
          "| 방법 | 평균 단어/청크 | ceiling(%) | DCG@20 | ceiling 대비 달성률 |", "|---|---|---|---|---|"]
    for n in order:
        A = agg(results[n]["records"], S["all100"])
        c = results[n]["chunk_stats"]
        ach = A[f"dcg@20"] / A["ceiling"] if A["ceiling"] else float("nan")
        L.append(f"| {DISPLAY.get(n, n)} | {c['mean_words_per_chunk']} | {A['ceiling']*100:.2f} | "
                 f"{A['dcg@20']*100:.2f} | {ach*100:.1f}% |")

    hdr = ("| 방법 | n | ceil% | " + "  ".join(f"D@{k}" for k in TOPKS) + " | "
           + "  ".join(f"R@{k}" for k in TOPKS) + " |")
    sep = "|---|---|---|---|---|"

    # ---- 3. 주표 ------------------------------------------------------------
    L += ["\n## 3. 주표 — 전체 100편 (DCG@k / Recall@k, %)\n", hdr, sep]
    for n in order:
        A = agg(results[n]["records"], S["all100"])
        paper = (f" 논문 {PAPER_DCG20[n]:.2f}" if n in PAPER_DCG20 else "")
        L.append(fmt_row(DISPLAY.get(n, n), A, paper and f"  <!--{paper}-->" or ""))

    L += ["\n### 논문 발표 수치와의 대조 (ada-002 · 순위 재현 확인용)\n",
          "| 방법 | 논문 DCG@20 | 우리 DCG@20 | 논문 순위 | 우리 순위 |", "|---|---|---|---|---|"]
    ours = {n: agg(results[n]["records"], S["all100"])["dcg@20"] * 100 for n in base_names}
    prank = {n: i + 1 for i, n in enumerate(sorted(PAPER_DCG20, key=lambda x: -PAPER_DCG20[x]))}
    orank = {n: i + 1 for i, n in enumerate(sorted(ours, key=lambda x: -ours[x]))}
    for n in sorted(base_names, key=lambda x: -PAPER_DCG20.get(x, 0)):
        L.append(f"| {DISPLAY.get(n, n)} | {PAPER_DCG20.get(n, float('nan')):.2f} | {ours[n]:.2f} | "
                 f"{prank.get(n, '—')} | {orank.get(n, '—')} |")
    same = all(prank.get(n) == orank.get(n) for n in base_names)
    L.append(f"\n**순위 재현: {'✅ 완전 일치' if same else '🔴 불일치 — 진단 필요'}**\n")

    # ---- 4. 부분집합 --------------------------------------------------------
    for key, title in [("narrative_fiction", "서사 픽션 85편 (M5 scope)"),
                       ("non_narrative", "비서사 11편 (도메인 밖 — 철학·에세이·운문)"),
                       ("nonfiction_narrative", "논픽션 서사 4편 (자서전)")]:
        L += [f"\n## 4. 부분집합 — {title}\n", hdr, sep]
        for n in order:
            A = agg(results[n]["records"], S[key])
            if A:
                L.append(fmt_row(DISPLAY.get(n, n), A))

    # ---- 5. 오염 ------------------------------------------------------------
    L += ["\n## 5. 오염 2편 vs 나머지 98편\n",
          "Dubliners(PG 2814) · The King in Yellow(PG 8492) 는 M5 앵커 학습 데이터에 포함됐다"
          "(D-B39 §1). 제외하지 않고 **유지 + 분리 측정** — 제외하면 오염 효과를 *가정*하게 되지만 "
          "유지하면 *측정*할 수 있다.\n", hdr, sep]
    for n in order:
        for key, lab in [("contaminated", "오염 2편"), ("all100_minus_contaminated", "청정 98편")]:
            A = agg(results[n]["records"], S[key])
            if A:
                L.append(fmt_row(f"{DISPLAY.get(n, n)} — {lab}", A))

    # ---- 6. dev / test ------------------------------------------------------
    for sp in ["dev", "test"]:
        L += [f"\n## 6. {sp} ({len(S[sp])}편)\n", hdr, sep]
        for n in order:
            A = agg(results[n]["records"], S["all100"], split=sp)
            if A:
                L.append(fmt_row(DISPLAY.get(n, n), A))

    # ---- 7. 크기 곡선 -------------------------------------------------------
    if m5_names:
        L += ["\n## 7. M5 크기 곡선 (전체 100편) — 선택 없이 전 구간\n",
              "| M5 변형 | 청크/책 | 평균 단어 | ceiling% | DCG@20 | Recall@20 |", "|---|---|---|---|---|---|"]
        for n in m5_names:
            A = agg(results[n]["records"], S["all100"])
            c = results[n]["chunk_stats"]
            L.append(f"| {n} | {c['chunks_per_book']} | {c['mean_words_per_chunk']} | "
                     f"{A['ceiling']*100:.2f} | {A['dcg@20']*100:.2f} | {A['recall@20']*100:.2f} |")
        L.append("\n베이스라인 참조점: " + " · ".join(
            f"{DISPLAY.get(n, n)} {results[n]['chunk_stats']['mean_words_per_chunk']}단어 "
            f"DCG@20 {ours.get(n, float('nan')):.2f}" for n in base_names))

    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:4]))
    print(f"\n방법 {len(order)}개 집계 → {a.out}")


if __name__ == "__main__":
    main()
