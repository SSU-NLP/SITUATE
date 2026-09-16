#!/usr/bin/env python3
"""GutenQA 5종 입도 + 질문 수집 → 공통 포맷으로 정규화 (D-B39 Phase 1).

## 산출
| 파일 | 내용 |
|---|---|
| `data/books.jsonl` | 책별 **문단 시퀀스**(`GutenQA_Paragraphs` = 세그먼테이션 이전 수동 추출본). M5 입력 원천이자 **문단 경계 정의**(D-B39 §6). |
| `data/questions.jsonl` | 질문 + `Chunk Must Contain` gold 부분문자열 |
| `data/chunks/{method}.jsonl` | 방법별 책당 청크 리스트. `paragraph`/`recursive`/`semantic`/`lumberchunker`/`propositions` |

## 주의
- `propositions` 는 원본이 LumberChunker 청크 안에 리스트로 중첩돼 있다 → **explode** 해서 명제 하나 = 청크 하나로 편다.
- `Chunk Must Contain` 이 **빈 문자열인 질문**이 있다. gold 판정이 불가능하므로 제외하고 개수를 기록한다
  (6개 방법 전부에서 동일하게 빠지므로 편향 없음).

    uv run python scripts/gutenqa/01_fetch.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"
DATA = BASE / "data"

HF = "https://huggingface.co/datasets/LumberChunker"
SOURCES = {  # 로컬파일명 -> (HF 데이터셋, 레포 내 파일)
    "questions":     ("GutenQA", "questions.parquet"),
    "lumberchunker": ("GutenQA", "gutenqa_chunks.parquet"),
    "paragraphs":    ("GutenQA_Paragraphs", "GutenQA_paragraphs.parquet"),
    "recursive":     ("GutenQA_Recursive", "GutenQA_recursive.parquet"),
    "semantic":      ("GutenQA_Semantic", "GutenQA_semantic.parquet"),
    "propositions":  ("GutenQA_Propositions", "gutenqa_propositions_only.parquet"),
}
# 논문/레포 명칭 -> 우리 method 키
METHODS = {"paragraph": "paragraphs", "recursive": "recursive", "semantic": "semantic",
           "lumberchunker": "lumberchunker", "propositions": "propositions"}


def parse_props(v) -> list[str]:
    """`Propositions` 는 파이썬 리스트 리터럴을 담은 **문자열**이다.
    그냥 순회하면 문자 단위로 쪼개져 5,700만 '청크'가 나온다(실측). 반드시 파싱한다."""
    import ast
    if v is None:
        return []
    if isinstance(v, (list, tuple)) or hasattr(v, "tolist"):
        return [str(x) for x in (v.tolist() if hasattr(v, "tolist") else v)]
    s = str(v).strip()
    if not s:
        return []
    try:
        out = ast.literal_eval(s)
        return [str(x) for x in out] if isinstance(out, (list, tuple)) else [s]
    except (ValueError, SyntaxError):
        return [s]      # 파싱 실패한 행은 통째로 청크 1개 취급 (버리지 않는다)


def ensure(name: str):
    import pandas as pd
    p = DATA / f"{name}.parquet"
    if not p.exists():
        import urllib.request
        ds, fn = SOURCES[name]
        p.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF}/{ds}/resolve/main/{fn}"
        print(f"[fetch] {url}")
        urllib.request.urlretrieve(url, p)
    return pd.read_parquet(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="이미 있어도 다시 만든다")
    a = ap.parse_args()

    (DATA / "chunks").mkdir(parents=True, exist_ok=True)
    stats: dict = {}

    # ---- 1. 문단 원천 (M5 입력 + 문단 경계 정의) ----------------------------
    par = ensure("paragraphs").sort_values(["Book ID", "Chunk ID"])
    books_p = DATA / "books.jsonl"
    if a.force or not books_p.exists():
        with books_p.open("w", encoding="utf-8") as f:
            for bid, g in par.groupby("Book ID", sort=True):
                f.write(json.dumps({
                    "book_id": int(bid), "book_name": g["Book Name"].iloc[0],
                    "paragraphs": [str(x) for x in g["Chunk"]],
                    "chapters": [str(x) for x in g["Chapter"]],
                }, ensure_ascii=False) + "\n")
    stats["books"] = int(par["Book ID"].nunique())
    stats["paragraphs_total"] = len(par)

    # ---- 2. 질문 + gold 부분문자열 ------------------------------------------
    q = ensure("questions")
    qs, empty = [], 0
    for i, r in enumerate(q.itertuples(index=False)):
        gold = str(r[q.columns.get_loc("Chunk Must Contain")] or "").strip()
        if not gold:
            empty += 1
            continue
        qs.append({"qid": i, "book_id": int(r[q.columns.get_loc("Book ID")]),
                   "book_name": str(r[q.columns.get_loc("Book Name")]),
                   "question": str(r[q.columns.get_loc("Question")]),
                   "answer": str(r[q.columns.get_loc("Answer")]),
                   "gold": gold})
    with (DATA / "questions.jsonl").open("w", encoding="utf-8") as f:
        for x in qs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    stats["questions_total"] = len(q)
    stats["questions_kept"] = len(qs)
    stats["questions_dropped_empty_gold"] = empty

    # ---- 3. 방법별 청크 -----------------------------------------------------
    stats["methods"] = {}
    for meth, src in METHODS.items():
        df = ensure(src).sort_values(["Book ID", "Chunk ID"])
        col = "Propositions" if meth == "propositions" else "Chunk"
        out = DATA / f"chunks/{meth}.jsonl"
        n_chunks = 0
        with out.open("w", encoding="utf-8") as f:
            for bid, g in df.groupby("Book ID", sort=True):
                if meth == "propositions":
                    chunks = [str(p) for lst in g[col] for p in parse_props(lst)]
                else:
                    chunks = [str(x) for x in g[col]]
                chunks = [c for c in chunks if c.strip()]
                n_chunks += len(chunks)
                f.write(json.dumps({"book_id": int(bid), "chunks": chunks},
                                   ensure_ascii=False) + "\n")
        stats["methods"][meth] = {"chunks": n_chunks,
                                  "chunks_per_book": round(n_chunks / stats["books"], 1)}
        print(f"[chunks] {meth:15s} {n_chunks:>7,} chunks  "
              f"({n_chunks/stats['books']:.0f}/book)")

    (BASE / "fetch_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"\n books={stats['books']} · paragraphs={stats['paragraphs_total']:,}")
    print(f" questions {stats['questions_kept']:,} kept / {stats['questions_total']:,} "
          f"(gold 빈 질문 {empty} 제외)")
    print(f" saved -> {DATA}")


if __name__ == "__main__":
    main()
