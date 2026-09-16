#!/usr/bin/env python3
"""GutenQA 검색 평가 하네스 — DCG@k / Recall@k (D-B39 Phase 1).

## 지표는 LumberChunker 레포 코드 그대로다 (우리 정의로 대체 금지)
`Code/Retrieval_Granularities_Eval.ipynb` 에서 발췌한 원문 로직:

    def find_index_of_match(respostas, gold_label):      # 관련도 = 대소문자 무시 부분문자열
        relevance = []; gold_label = gold_label.lower()
        for item in respostas:
            if gold_label in item.lower():
                relevance.append(1)
                relevance += (len(respostas) - len(relevance)) * [0]   # ← **첫 히트에서 끊는다**
                break
            else: relevance.append(0)
        return relevance

    def compute_Recall(rel): return 1 if 1 in rel else 0               # ← 사실상 hit-rate
    def compute_DCG(rel):    return sum((2**rel[i-1]-1)/log2(i+1) for i in 1..len(rel))

세 가지를 반드시 알아야 한다:
1. **relevance 는 one-hot** — 첫 히트 이후는 전부 0. 따라서 DCG@k = `1/log2(rank+1)` (rank 는 1-based),
   top-k 밖이면 0. **IDCG 정규화 없음 = nDCG 아니라 생 DCG.**
2. **`Recall@k` 는 이름과 달리 재현율이 아니라 hit-rate**(top-k 안에 하나라도 있으면 1). 논문
   Recall@20 77.92 는 "질문의 77.92%에서 gold 청크를 20위 안에 넣었다"는 뜻.
3. **유사도 = 생 내적**(`np.dot`), 코사인 아님. 우리 인코더 `multi-qa-mpnet-base-dot-v1` 도 내적용으로
   학습된 모델이라 정규화하지 않는 쪽이 레포 코드와도, 모델 설계와도 맞다.

## 🔴 여기서만 나오는 진단: `ceiling`
gold 가 **어떤 청크에도 통째로 안 들어가면** 그 질문은 어떤 검색기로도 못 맞힌다. 청크가 작을수록
gold 부분문자열이 두 청크로 쪼개질 확률이 커지므로 **이 벤치마크는 큰 청크에 구조적으로 유리하다.**
`ceiling` = 그 책의 청크 중 하나라도 gold 를 포함하는 질문의 비율 = DCG/Recall 의 상한.
방법별 ceiling 을 같이 보고하지 않으면 순위 해석이 틀린다. (논문 Table 1 은 이걸 안 보여준다.)

    uv run python scripts/gutenqa/02_eval_harness.py --method paragraph
    uv run python scripts/gutenqa/02_eval_harness.py --chunks <path.jsonl> --name m5__r0.0642
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

BASE = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"
DATA = BASE / "data"
OUTDIR = BASE / "retrieval"

# 🔒 D-B39 §5 — 검색용 인코더. 6개 방법 전부 동일. 분절용(mpnet paraphrase-multilingual)과 역할이 다르다.
RETRIEVAL_ENCODER = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
TOPKS = [1, 2, 5, 10, 20]        # 레포 노트북 x_values 와 동일


def compute_dcg(rel: list[int]) -> float:
    """레포 `compute_DCG` 그대로."""
    return sum((2 ** rel[i - 1] - 1) / math.log2(i + 1) for i in range(1, len(rel) + 1))


def relevance(retrieved: list[str], gold: str) -> list[int]:
    """레포 `find_index_of_match` 그대로 — 첫 히트에서 끊고 나머지는 0."""
    g = gold.lower()
    rel: list[int] = []
    for item in retrieved:
        if g in item.lower():
            rel.append(1)
            rel += (len(retrieved) - len(rel)) * [0]
            break
        rel.append(0)
    return rel


def load_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def get_encoder(device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(RETRIEVAL_ENCODER, device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", help="data/chunks/<method>.jsonl 을 쓴다")
    ap.add_argument("--chunks", help="임의 청크 jsonl 경로 (M5 산출물용)")
    ap.add_argument("--name", help="출력 이름 (기본 = --method 또는 --chunks stem)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    if not (a.method or a.chunks):
        ap.error("--method 또는 --chunks 중 하나는 필요")
    # 🔴 .resolve() 필수 — 상대경로로 --chunks 를 주면 아래 relative_to 가 터진다.
    #    (실측: 100/100 책 계산을 다 끝낸 뒤 기록 직전에 죽어 전부 날렸다.)
    cpath = (Path(a.chunks).resolve() if a.chunks else DATA / f"chunks/{a.method}.jsonl")
    name = a.name or (a.method if a.method else cpath.stem)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    qs = list(load_jsonl(DATA / "questions.jsonl"))
    by_book: dict[int, list[dict]] = {}
    for q in qs:
        by_book.setdefault(q["book_id"], []).append(q)
    chunks_by_book = {r["book_id"]: r["chunks"] for r in load_jsonl(cpath)}
    subsets = json.loads((BASE / "book_subsets.json").read_text(encoding="utf-8"))
    split = {r["book_id"]: r["split"] for r in subsets["books"]}

    enc = get_encoder(device)
    kmax = max(TOPKS)
    records = []
    tok_total = n_chunks = 0

    for bi, bid in enumerate(sorted(chunks_by_book)):
        chunks = chunks_by_book[bid]
        qlist = by_book.get(bid, [])
        if not chunks or not qlist:
            continue
        n_chunks += len(chunks)
        tok_total += sum(len(c.split()) for c in chunks)

        # 🔴 정규화하지 않는다 — 레포는 생 내적을 쓰고, 이 인코더도 내적용이다
        C = np.asarray(enc.encode(chunks, batch_size=a.batch, show_progress_bar=False,
                                  convert_to_numpy=True), dtype=np.float32)
        Q = np.asarray(enc.encode([q["question"] for q in qlist], batch_size=a.batch,
                                  show_progress_bar=False, convert_to_numpy=True), dtype=np.float32)
        sims = Q @ C.T                                        # (nq, nchunks)
        order = np.argsort(-sims, axis=1)[:, :kmax]
        low = [c.lower() for c in chunks]

        for qi, q in enumerate(qlist):
            retrieved = [chunks[j] for j in order[qi]]
            rel = relevance(retrieved, q["gold"])
            g = q["gold"].lower()
            in_any = any(g in c for c in low)                 # ceiling 진단
            rec = {"qid": q["qid"], "book_id": bid, "split": split.get(bid),
                   "gold_in_some_chunk": bool(in_any)}
            for k in TOPKS:
                rk = rel[:k]
                rec[f"dcg@{k}"] = compute_dcg(rk)
                rec[f"recall@{k}"] = 1 if 1 in rk else 0
            records.append(rec)
        if (bi + 1) % 20 == 0:
            print(f"  [{name}] {bi+1}/{len(chunks_by_book)} books", flush=True)

    try:
        chunks_file = str(cpath.relative_to(REPO_ROOT))
    except ValueError:
        chunks_file = str(cpath)
    payload = {
        "name": name, "chunks_file": chunks_file,
        "encoder": RETRIEVAL_ENCODER, "similarity": "raw dot product (레포 코드와 동일)",
        "topks": TOPKS,
        "chunk_stats": {"n_chunks": n_chunks,
                        "chunks_per_book": round(n_chunks / max(1, len(chunks_by_book)), 1),
                        "mean_words_per_chunk": round(tok_total / max(1, n_chunks), 1)},
        "n_questions": len(records),
        "records": records,
    }
    out = OUTDIR / f"{name}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    ceil = float(np.mean([r["gold_in_some_chunk"] for r in records]))
    print(f"\n== {name} ==")
    print(f"  청크 {n_chunks:,} ({payload['chunk_stats']['chunks_per_book']}/book, "
          f"평균 {payload['chunk_stats']['mean_words_per_chunk']} 단어)")
    print(f"  ceiling(gold 가 어떤 청크엔가 통째로 들어있는 질문 비율) = {ceil:.4f}")
    for k in TOPKS:
        d = float(np.mean([r[f"dcg@{k}"] for r in records]))
        rr = float(np.mean([r[f"recall@{k}"] for r in records]))
        print(f"  DCG@{k:<2d} {d*100:6.2f}   Recall@{k:<2d} {rr*100:6.2f}")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
