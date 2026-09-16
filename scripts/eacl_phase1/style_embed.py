#!/usr/bin/env python3
"""편 단위 임베딩 요약 — 모델이 실제로 보는 표현 공간에서의 다양성 측정용.

🔴 왜 필요한가 (codex 2026-08-02): 기능어·길이·서지 지표는 전부 **대리 변수**다. M5 는
   동결된 문장 임베딩만 보므로, 코퍼스가 **그 공간에서** 얼마나 넓게 퍼져 있는지가 가장
   직접적인 다양성 척도다.

절차: 편마다 문장 20개를 균등 간격으로 뽑아 인코딩 → 평균 → L2 정규화 = 편 벡터.
인코더는 학습 파이프라인과 **동일**(`paraphrase-multilingual-mpnet-base-v2`, normalize=True).
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/eacl_phase0"), str(REPO / "scripts/eacl_phase1")]
from build_narrative_pack import split_sentences  # noqa: E402
from build_pg_shorts import split_volume, stage0_piece  # noqa: E402

CACHE = REPO / "data/eacl_phase1/pg_cache"
CAT = REPO / "data/eacl_phase1/pgcat.csv"
OUT = REPO / "outputs/experiments/2026-08-02_author_cap"
MIN_SENT, MAX_SENT, MIN_MED = 30, 900, 5.0
EXCL = ["Pohl, Frederik"]
PER_PIECE = 20
MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def main() -> None:
    from sentence_transformers import SentenceTransformer
    meta = {int(r["Text#"]): r for r in csv.DictReader(CAT.open(encoding="utf-8"))}
    keys, flat, seen = [], [], set()
    files = sorted(CACHE.glob("*.txt"), key=lambda f: int(f.stem))
    for i, f in enumerate(files):
        m = meta.get(int(f.stem))
        if m is None:
            continue
        au = m.get("Authors", "")
        if any(a in au for a in EXCL):
            continue
        pieces, reason = split_volume(f.read_text(encoding="utf-8", errors="ignore"))
        if reason == "chaptered_work":
            continue
        for title, body in pieces:
            k = (au, title.upper())
            if title and k in seen:
                continue
            sents, _ = stage0_piece(split_sentences(body))
            if not (MIN_SENT <= len(sents) <= MAX_SENT):
                continue
            if statistics.median(len(x.split()) for x in sents) < MIN_MED:
                continue
            seen.add(k)
            idx = np.linspace(0, len(sents) - 1, PER_PIECE).astype(int)   # 균등 간격 표집
            keys.append({"author": au, "gid": int(f.stem)})
            flat.extend(sents[j] for j in idx)
        if (i + 1) % 500 == 0:
            print(f"  수집 {i+1}/{len(files)}권 · 편 {len(keys):,}", flush=True)

    print(f"[인코딩] 편 {len(keys):,} × {PER_PIECE}문장 = {len(flat):,}문장", flush=True)
    enc = SentenceTransformer(MODEL, device="cuda")
    E = enc.encode(flat, batch_size=512, normalize_embeddings=True,
                   show_progress_bar=True, convert_to_numpy=True)
    V = E.reshape(len(keys), PER_PIECE, -1).mean(1)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    np.save(OUT / "piece_emb.npy", V.astype(np.float32))
    (OUT / "piece_emb_keys.json").write_text(json.dumps(keys), encoding="utf-8")
    print(f"[완료] {V.shape} -> {OUT/'piece_emb.npy'}")


if __name__ == "__main__":
    main()
