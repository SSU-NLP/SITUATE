#!/usr/bin/env python3
"""B1 사전등록 ①단계 — 길이 상한·하한을 **앵커 학습 코퍼스에서** 도출 (GutenQA 미열람).

## 왜 앵커에서 뽑는가
LumberChunker 의 p95(586단어)나 θ=550 을 쓰고 싶은 유혹이 있지만, 그건 **경쟁자의 작동점을 평가
데이터에서 베끼는 것**이다. 대신 M5 가 **학습된 도메인**(앵커 = PG 영어 단편)에서 동결 임계
`fixed_quantile r=0.0642` 로 분절했을 때 나오는 길이 분포의 p5/p95 를 쓴다.
- GutenQA 를 한 번도 보지 않고 정해진다.
- "학습 도메인에서 이 모델이 자연스럽게 내는 조각 크기" 라는 원리적 근거가 붙는다.

## 절차
1. 앵커 코퍼스(`data/eacl_phase1/final_master.jsonl`, 10,598편)에서 seed 고정 표본 N편.
   각 doc 의 문장열 재구성 = `[t=1 의 previous] + [각 t 의 current]`.
2. `hi_ontop.m5_default.segment()` 로 분절 (동결 설정 그대로, `configs/m5_default.json` 무변경).
3. 조각별 단어수 분포 → **MIN_WORDS = p5 · MAX_WORDS = p95** (정수 반올림).

산출 `b1_length_bounds.json`. 이 값을 사전등록에 못박은 뒤 B1 을 실행한다.

    uv run python scripts/gutenqa/07_derive_length_bounds.py --n-docs 500 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

MASTER = REPO_ROOT / "data/eacl_phase1/final_master.jsonl"
OUT = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking/b1_length_bounds.json"
SEED = 20260711          # 레포 관례 시드


def iter_docs(path: Path, wanted: set[str]):
    """doc_id 별 문장열 재구성. transition 순서대로 스트리밍(파일이 doc 단위로 정렬돼 있음)."""
    cur_id, prev_sents = None, []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            did = r["doc_id"]
            if did != cur_id:
                if cur_id is not None and cur_id in wanted and len(prev_sents) >= 2:
                    yield cur_id, prev_sents
                cur_id, prev_sents = did, []
            if did not in wanted:
                continue
            if r["transition"] == 1:
                prev_sents = [r["previous"], r["current"]]
            else:
                prev_sents.append(r["current"])
    if cur_id is not None and cur_id in wanted and len(prev_sents) >= 2:
        yield cur_id, prev_sents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=500)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 전체 doc_id 목록 (meta 사이드카에서 — 본문 안 읽음)
    meta = REPO_ROOT / "data/eacl_phase1/all_master.meta.jsonl"
    all_ids = [json.loads(l)["doc_id"] for l in meta.open(encoding="utf-8") if l.strip()]
    rng = random.Random(SEED)
    wanted = set(rng.sample(all_ids, min(a.n_docs, len(all_ids))))
    print(f"[b1-bounds] 앵커 {len(all_ids):,}편 중 표본 {len(wanted)}편 (seed={SEED})")

    from hi_ontop.m5_default import segment
    cfg = json.loads((REPO_ROOT / "configs/m5_default.json").read_text(encoding="utf-8"))
    print(f"[b1-bounds] 동결 설정 그대로: {cfg['threshold']['mode']} r={cfg['threshold']['rate']}")

    words, n_ok, n_sent_tot = [], 0, 0
    for did, sents in iter_docs(MASTER, wanted):
        r = segment(sents, device=device)
        cuts = [0] + list(r["boundaries"]) + [len(sents)]
        for x, y in zip(cuts[:-1], cuts[1:]):
            seg = " ".join(sents[x:y]).split()
            if seg:
                words.append(len(seg))
        n_ok += 1
        n_sent_tot += len(sents)
        if n_ok % 100 == 0:
            print(f"  {n_ok}/{len(wanted)} docs · 조각 {len(words):,}", flush=True)

    w = np.array(words, dtype=float)
    st = {
        "n_docs": n_ok, "n_sentences": n_sent_tot, "n_segments": int(w.size),
        "mean": float(w.mean()), "cv": float(w.std() / w.mean()),
        "p5": float(np.percentile(w, 5)), "p25": float(np.percentile(w, 25)),
        "p50": float(np.percentile(w, 50)), "p75": float(np.percentile(w, 75)),
        "p95": float(np.percentile(w, 95)), "max": float(w.max()),
        "MIN_WORDS": int(round(np.percentile(w, 5))),
        "MAX_WORDS": int(round(np.percentile(w, 95))),
        "source": "앵커 학습 코퍼스(PG 영어 단편) · 동결 fixed_quantile r=0.0642 · GutenQA 미열람",
        "seed": SEED,
    }
    OUT.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n앵커 도메인 M5 조각 길이 (단어): 평균 {st['mean']:.1f} · CV {st['cv']:.2f}")
    print(f"  p5 {st['p5']:.0f} · p25 {st['p25']:.0f} · p50 {st['p50']:.0f} · "
          f"p75 {st['p75']:.0f} · p95 {st['p95']:.0f} · max {st['max']:.0f}")
    print(f"\n🔒 MIN_WORDS = {st['MIN_WORDS']}  ·  MAX_WORDS = {st['MAX_WORDS']}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
