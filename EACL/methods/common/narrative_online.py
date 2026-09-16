#!/usr/bin/env python3
"""서사(NES) 무대 — **온라인 베이스라인** 공통 러너.

평가셋 = dev3(pieman·tunnel·monkey) + test3(run·gohitler·mymotherslife) **6편**.
지표 4종 = **Pk · WindowDiff · F1 · Score** (`hi_ontop.dts_scoring.score_dialogues`).
채점 = deployable (경계 수 자가결정, gold 개수 미사용).

## 온라인 규약
전이 i(문장 i+1 앞)의 판정에 **문장 i+2+d 이후를 쓰지 않는다.**
- `delay=0` → **zero-lookahead**: 전이가 생기는 순간 확정
- `delay=d>0` → **bounded-lookahead**: d 문장만큼 뒤를 본 뒤 확정

두 등급을 이름에 박는다(`-online` / `-online-d{d}`). 뭉뚱그리면 "미래를 안 쓴다"는
주장이 무너진다 — 스트리밍 구현 상당수가 사실 bounded 다.

## 왜 prefix-recompute 인가
바뀌는 변수가 **미래 문맥 하나뿐**이라 `offline − online` 이 그 기여로 읽힌다.
스트리밍 근사(running threshold 등)를 쓰면 근사 오차가 섞인다. 근사판이 필요하면
별도 이름으로 따로 낸다.

## 비용
문서당 O(n²) 호출. 서사 문서는 95~503 전이라 경량 방법(TextTiling 등)에만 쓴다.
무거운 모델은 자체 스트리밍 구현이 필요하다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))

PRED = REPO / "outputs/experiments/2026-07-29_baseline_compare/preds"
DATASETS = ("dev3", "test3")
METRICS4 = ("pk", "wd", "f1", "score")


def load_all() -> dict[str, list[tuple[str, list[str], np.ndarray]]]:
    from baseline_compare import load_dataset      # noqa: PLC0415
    return {ds: load_dataset(ds) for ds in DATASETS}


def online_flags(offline_fn, sents: list[str], delay: int = 0) -> list[int]:
    """오프라인 분절기를 prefix 로 반복 호출해 온라인 판정을 만든다.

    `offline_fn(sents_prefix) -> list[int]` 는 **전이별 0/1**(길이 len-1)을 돌려줘야 한다.
    전이 i 의 판정은 `sents[:i+2+delay]` 만 보고 내린다.
    실패(예외/길이 불일치)는 **경계 없음(0)** 으로 떨어뜨리고 miss 로 센다.
    """
    n = len(sents)
    out = [0] * (n - 1)
    miss = 0
    for i in range(n - 1):
        upto = min(n, i + 2 + delay)
        try:
            f = offline_fn(sents[:upto])
        except Exception:  # noqa: BLE001
            f = None
        if f is None or len(f) <= i:
            miss += 1
            continue
        out[i] = int(f[i])
    return out, miss


# ---------------------------------------------------------------- 방법들 ---
def texttiling_offline_fn(w: int = 10, k: int = 6):
    """nltk TextTiling → 전이별 0/1. SuperDialseg `TexttilingSegmenter` 와 같은 규약."""
    from nltk.tokenize.texttiling import TextTilingTokenizer

    tt = TextTilingTokenizer(w=w, k=k)

    def fn(sents: list[str]) -> list[int] | None:
        doc = "\n\n".join((s.strip() or ".") for s in sents)
        tiles = tt.tokenize(doc)
        per = []
        for tile in tiles:
            lines = [x for x in tile.strip().split("\n\n") if x]
            if not lines:
                continue
            per += [0] * len(lines)
            per[-1] = 1
        if len(per) != len(sents):
            return None
        # per[i]=1 → 문장 i 가 타일을 끝냄 → 전이 i (문장 i+1 앞) 가 경계
        return per[:-1]

    return fn


def streaming_texttiling_flags(sents: list[str], w: int = 10, k: int = 6,
                               c: float = 0.5, min_gap: int = 4,
                               warmup_gaps: int = 3) -> list[int]:
    """`StreamingTextTiling` (bounded-lookahead) — push 루프가 인과성을 보장한다."""
    from hi_ontop.baselines import StreamingTextTiling

    seg = StreamingTextTiling(w=w, k=k, c=c, min_gap=min_gap, warmup_gaps=warmup_gaps)
    out = [0] * (len(sents) - 1)
    for s in sents:
        # push 는 **새로 확정된 boundary 문장 index(1-based)** 를 돌려준다.
        # DTS 규약대로 "문장 b 가 세그먼트를 끝냈다" 는 뜻 → 전이 b (문장 b|b+1 사이) 가 경계.
        # 우리 전이 인덱싱은 0-based(전이 i = 문장 i+1 앞)이므로 out[b-1] 이다.
        for b in seg.push(s):
            if 1 <= b <= len(out):
                out[b - 1] = 1
    return out


def dump(name: str, ds: str, preds: dict[str, list[int]]) -> None:
    PRED.mkdir(parents=True, exist_ok=True)
    (PRED / f"{name}_{ds}.json").write_text(json.dumps(preds))


def score4(preds: dict[str, list[int]], data) -> dict:
    """4지표만 — Pk · WindowDiff · F1 · Score."""
    from hi_ontop.dts_scoring import score_dialogues      # noqa: PLC0415

    docs = [d for d, _s, _g in data]
    G = [g for _d, _s, g in data]
    ys = [np.concatenate([np.asarray(g, np.int8), [0]]).tolist() for g in G]
    ps = [np.concatenate([np.asarray(preds[d][:len(g)], np.int8), [0]]).tolist()
          for d, g in zip(docs, G)]
    s = score_dialogues(ys, ps)
    return {"pk": s["pk"], "wd": s["wd"], "f1": s["f1"], "score": s["score"]}
