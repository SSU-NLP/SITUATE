#!/usr/bin/env python3
"""ada-002 검색 평가 — 논문 Table 1 과 **완전히 같은 조건** (D-B47).

`02_eval_harness.py` 와 지표 코드·판정 규칙·TOPK 가 전부 동일하고 **인코더만 다르다.**
벡터는 `11_ada_embed_cache.py` 가 채운 sqlite 캐시에서 읽는다 — 이 스크립트는 결제하지 않는다.
캐시에 없는 청크가 있으면 **실패**한다(조용히 0벡터로 때우면 점수가 거짓이 된다).

🔴 정규화하지 않는다. ada-002 는 OpenAI 가 이미 단위벡터로 준다(실측 노름 1.000000)
⇒ 레포의 `np.dot` 이 그대로 코사인이다. 우리가 추가로 손대면 논문 조건에서 벗어난다.

산출은 `retrieval_ada/<셀>.json` — mpnet 결과(`retrieval/`)를 덮어쓰지 않는다. 둘을 나란히 보고한다.

    uv run python scripts/gutenqa/12_eval_ada002.py --cells lumberchunker recursive paragraph
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"
DATA = BASE / "data"
OUTDIR = BASE / "retrieval_ada"

_spec = importlib.util.spec_from_file_location(
    "gq_harness", REPO_ROOT / "scripts/gutenqa/02_eval_harness.py")
H = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(H)

_spec2 = importlib.util.spec_from_file_location(
    "gq_ada", REPO_ROOT / "scripts/gutenqa/11_ada_embed_cache.py")
A = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(A)

TOPKS = H.TOPKS
MODEL_LABEL = "openai/text-embedding-ada-002 (논문 Table 1 각주 3 과 동일)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    a = ap.parse_args()

    cache = A.Cache(A.CACHE_DB)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    qs = list(H.load_jsonl(DATA / "questions.jsonl"))
    by_book: dict[int, list[dict]] = {}
    for q in qs:
        by_book.setdefault(q["book_id"], []).append(q)
    qvec = cache.get_many([A.sha(q["question"]) for q in qs])
    missing_q = sum(1 for q in qs if A.sha(q["question"]) not in qvec)
    if missing_q:
        raise SystemExit(f"🔴 질문 임베딩 {missing_q}개 없음 — "
                         "`11_ada_embed_cache.py --questions` 를 먼저 돌릴 것")

    subsets = json.loads((BASE / "book_subsets.json").read_text(encoding="utf-8"))
    split = {r["book_id"]: r["split"] for r in subsets["books"]}
    kmax = max(TOPKS)

    for cell in a.cells:
        cpath = DATA / f"chunks/{cell}.jsonl"
        if not cpath.exists():
            print(f"  skip {cell} (없음)"); continue
        chunks_by_book = {r["book_id"]: r["chunks"] for r in H.load_jsonl(cpath)}
        records = []
        n_chunks = word_total = 0
        for bi, bid in enumerate(sorted(chunks_by_book)):
            chunks = chunks_by_book[bid]
            qlist = by_book.get(bid, [])
            if not chunks or not qlist:
                continue
            n_chunks += len(chunks)
            word_total += sum(len(c.split()) for c in chunks)

            hs = [A.sha(c) for c in chunks]
            vm = cache.get_many(hs)
            miss = [h for h in hs if h not in vm]
            if miss:
                raise SystemExit(
                    f"🔴 {cell} book{bid}: 캐시에 없는 청크 {len(miss)}개. "
                    f"`11_ada_embed_cache.py --cells {cell}` 를 먼저 돌릴 것")
            C = np.stack([vm[h] for h in hs])                       # (nchunks, 1536), 단위벡터
            Q = np.stack([qvec[A.sha(q["question"])] for q in qlist])
            sims = Q @ C.T                                          # 생 내적 = 코사인
            order = np.argsort(-sims, axis=1)[:, :kmax]
            low = [c.lower() for c in chunks]

            for qi, q in enumerate(qlist):
                retrieved = [chunks[j] for j in order[qi]]
                rel = H.relevance(retrieved, q["gold"])
                g = q["gold"].lower()
                rec = {"qid": q["qid"], "book_id": bid, "split": split.get(bid),
                       "gold_in_some_chunk": bool(any(g in c for c in low))}
                for k in TOPKS:
                    rk = rel[:k]
                    rec[f"dcg@{k}"] = H.compute_dcg(rk)
                    rec[f"recall@{k}"] = 1 if 1 in rk else 0
                records.append(rec)
            if (bi + 1) % 25 == 0:
                print(f"  [{cell}] {bi+1}/{len(chunks_by_book)} books", flush=True)

        payload = {
            "name": cell, "chunks_file": str(cpath.relative_to(REPO_ROOT)),
            "encoder": MODEL_LABEL,
            "similarity": "raw dot product (레포 코드와 동일; ada 는 단위벡터라 = 코사인)",
            "topks": TOPKS,
            "chunk_stats": {"n_chunks": n_chunks,
                            "chunks_per_book": round(n_chunks / max(1, len(chunks_by_book)), 1),
                            "mean_words_per_chunk": round(word_total / max(1, n_chunks), 1)},
            "n_questions": len(records), "records": records,
        }
        (OUTDIR / f"{cell}.json").write_text(json.dumps(payload, ensure_ascii=False),
                                             encoding="utf-8")
        m = lambda k: float(np.mean([r[k] for r in records])) * 100      # noqa: E731
        print(f"== {cell} ==  DCG@20 {m('dcg@20'):.2f} · Recall@20 {m('recall@20'):.2f} · "
              f"DCG@1 {m('dcg@1'):.2f} · ceiling {m('gold_in_some_chunk'):.2f}%")


if __name__ == "__main__":
    main()
