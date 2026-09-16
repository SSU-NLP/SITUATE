#!/usr/bin/env python3
"""B1 길이 바운드 도출 — **튜닝이 아니라 선언된 성질에서** (D-B42 §1).

앵커에서 뽑으려던 `07_derive_length_bounds.py` 는 폐기했다(`_discarded/` 참조).
두 바운드 모두 **점수를 최적화해 고른 값이 아니다.**

| 바운드 | 출처 | 튜닝? |
|---|---|---|
| **MAX_WORDS** | 검색 인코더 `max_seq_length`(토큰) ÷ 실측 토큰/단어 비율 | ❌ 인코더 사양 |
| **MIN_WORDS** | GutenQA **dev 20편** gold span 단어수 p95 | ❌ dev 분포, 점수 미사용 |

**MAX**: 인코더가 `max_seq_length` 를 넘는 텍스트를 **잘라버린다**. 그 뒤의 gold 는 임베딩에
반영되지 않는다. 청크를 그보다 크게 만들 이유가 없다.
🔴 단 이것은 하드 상한이 **아니다** — 큰 청크는 gold 를 못 봐도 앞부분이 질문과 맞아 검색에
성공하기도 한다(실측: `m5__rate0.15` 는 gold 가 보이는 비율 49.6% 인데 Recall@20 61.8%).
**핸디캡이지 천장이 아니다.**

**MIN**: 청크가 gold span 보다 짧으면 애초에 담을 수 없다. dev gold p95 를 쓰면 95% 의 gold 를
담을 수 있는 최소 크기가 된다. **test 80편은 보지 않는다.**

    uv run python scripts/gutenqa/07b_declare_bounds.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"
DATA = BASE / "data"
OUT = BASE / "b1_length_bounds.json"

RETRIEVAL_ENCODER = "sentence-transformers/multi-qa-mpnet-base-dot-v1"   # 02_eval_harness 와 동일
GOLD_PCT = 95            # MIN = dev gold span 단어수 p95


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-chunks", type=int, default=200,
                    help="토큰/단어 비율 실측에 쓸 청크 수")
    a = ap.parse_args()

    # ── MAX: 인코더 용량 ──────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(RETRIEVAL_ENCODER, device="cpu")
    max_tok = int(enc.max_seq_length)
    tok = enc.tokenizer
    texts = []
    with (DATA / "chunks/lumberchunker.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            texts += json.loads(line)["chunks"]
            if len(texts) >= a.sample_chunks:
                break
    texts = texts[:a.sample_chunks]
    ratios = [len(tok.encode(t, add_special_tokens=False)) / max(1, len(t.split()))
              for t in texts]
    tpw = float(np.mean(ratios))
    MAX_WORDS = int(max_tok / tpw)

    # ── MIN: dev gold span p95 ───────────────────────────────────────
    subs = json.loads((BASE / "book_subsets.json").read_text(encoding="utf-8"))
    dev = set(subs["subsets"]["dev"])
    gold = [len(json.loads(l)["gold"].split())
            for l in (DATA / "questions.jsonl").open(encoding="utf-8")
            if l.strip() and json.loads(l)["book_id"] in dev]
    g = np.array(gold, dtype=float)
    MIN_WORDS = int(round(np.percentile(g, GOLD_PCT)))

    payload = {
        "MIN_WORDS": MIN_WORDS, "MAX_WORDS": MAX_WORDS,
        "max_source": {"encoder": RETRIEVAL_ENCODER, "max_seq_length_tokens": max_tok,
                       "tokens_per_word_measured": round(tpw, 4),
                       "n_chunks_sampled": len(texts),
                       "note": "인코더 사양 — 튜닝 아님. 하드 상한이 아니라 핸디캡."},
        "min_source": {"split": "GutenQA dev 20편", "n_questions": int(g.size),
                       "percentile": GOLD_PCT,
                       "gold_words_p50": float(np.percentile(g, 50)),
                       "gold_words_p95": float(np.percentile(g, 95)),
                       "gold_words_max": float(g.max()),
                       "note": "dev 분포에서 도출. 점수 미사용, test 미열람."},
        "frozen_at": "2026-07-30 (D-B42 §1)",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"MAX: 인코더 {RETRIEVAL_ENCODER}")
    print(f"     max_seq_length = {max_tok} 토큰 · 실측 {tpw:.3f} 토큰/단어 (n={len(texts)})")
    print(f"     ⇒ MAX_WORDS = {MAX_WORDS}")
    print(f"MIN: dev 20편 gold span n={g.size} · p50 {np.percentile(g,50):.0f} · "
          f"p95 {np.percentile(g,95):.0f} · max {g.max():.0f}")
    print(f"     ⇒ MIN_WORDS = {MIN_WORDS}")
    print(f"\n🔒 saved -> {OUT}")


if __name__ == "__main__":
    main()
