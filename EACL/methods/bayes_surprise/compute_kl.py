#!/usr/bin/env python3
"""B1-BayesSurprise 재현 1단계 — 저자 배포 logits 에서 단어별 Bayesian surprise(KL) 계산.

## 무엇을 재현하는가
Kumar et al. (2023) 의 핵심 신호는 **연속한 두 시점의 다음-단어 분포 사이 KL 발산**이다
(surprisal 이 아니다). 저자 코드 `analysis/surprise_helper.py::compute_cross_entr`:

    prob[i]  = softmax(logits[i])                     # 50257 차원
    kld[i]   = scipy.stats.entropy(prob[i], prob[i-1])  # = KL(P_i || P_{i-1}), 단위 nat
    cross[i] = sum(-prob[i] * log2(prob[i-1]))          # 교차엔트로피, 단위 bit

Zenodo 배포본(10.5281/zenodo.8404102)에 **GPT-2-XL(문맥 1024)의 단어별 logits 전체**가
들어 있으므로, 모델을 다시 돌리지 않고도 저자와 **같은 입력**에서 같은 신호를 만들 수 있다.
= 우리 신호 구현의 검증이자 이후 회귀 재현의 입력.

## 🔴 우리 기존 구현과의 차이 (정정 대상)
`scripts/eacl_phase0/b1_bayes_surprise.py` 는 GPT-2-XL 을 **우리가 직접 돌려** KL 을 계산하고,
전이별 readout(onset-max)까지 붙인 우리 파이프라인용 각색이다. 논문 수치와 대조된 적이 없다.
이 파일은 그와 별개로 **저자 입력·저자 공식**만 쓴다.

## 성능
logits CSV 는 스토리 합계 8.3GB(행마다 50,257개 실수)다. `ast.literal_eval` 은 너무 느려
`np.fromstring` 으로 바꿨다. 값은 같고 속도만 다르다.

    uv run python EACL/methods/bayes_surprise/compute_kl.py --story pieman
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = DATA / "extract-embeddings-data/results"
OUT = HERE / "signals"
STORIES = ("pieman", "tunnel", "monkey")


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """scipy.stats.entropy(p, q) 와 동일 — 정규화 후 sum(p*log(p/q)), 단위 nat."""
    p = p / p.sum()
    q = q / q.sum()
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def cross_entropy_bits(p: np.ndarray, q: np.ndarray) -> float:
    """저자 `cross_entr` — sum(-p * log2(q)), 단위 bit."""
    return float(np.sum(-p * np.log2(np.clip(q, 1e-300, None))))


def process(story: str) -> dict:
    path = RESULTS / story / f"{story}gpt2-xl-c_1024_surp_entr.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — Zenodo 데이터를 먼저 풀어야 한다")

    csv.field_size_limit(sys.maxsize)
    kls, ces, surprise, entropy_col, words = [], [], [], [], []
    prev = None

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            logits = np.fromstring(row["logits"].strip()[1:-1], sep=",")
            if logits.size != 50257:
                raise ValueError(f"{story} 행 {i}: logits 차원 {logits.size} (50257 이어야 함)")
            p = softmax(logits)
            if prev is None:
                kls.append(0.0)
                ces.append(0.0)
            else:
                kls.append(kl(p, prev))
                ces.append(cross_entropy_bits(p, prev))
            prev = p
            surprise.append(float(row["Surprise"]))
            entropy_col.append(float(row["Entropy"]))
            words.append(row.get("Target Word", ""))
            if (i + 1) % 500 == 0:
                print(f"    {story}: {i + 1}단어", flush=True)

    out = {
        "kl": np.array(kls), "cross_entropy": np.array(ces),
        "surprisal": np.array(surprise), "entropy": np.array(entropy_col),
        "words": np.array(words, dtype=object),
    }
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / f"{story}_signals.npz", **out)
    print(f"  {story}: {len(kls)}단어 · KL 평균 {np.mean(kls[1:]):.4f} "
          f"· surprisal 평균 {np.mean(surprise):.4f} → {OUT / (story + '_signals.npz')}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", choices=[*STORIES, "all"], default="all")
    args = ap.parse_args()
    for s in (STORIES if args.story == "all" else [args.story]):
        print(f"### {s}")
        process(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
