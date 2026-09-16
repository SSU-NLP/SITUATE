#!/usr/bin/env python3
"""Embedding Delta **온라인 변형** — 서사 6편.

## 원본 (Schneider et al. 2021)
spaCy 토큰 벡터 → KMeans(k=1000) 코드북 → 문장을 클러스터 히스토그램으로 → 좌/우 창(각 25문장)의
**정규화 코사인 거리** = delta 신호 → 이동평균(10) 평활 → `argrelmax(order=20)` 피크 = 경계.

## 🔴 온라인화하려면 세 곳이 미래를 쓴다 — 전부 바꿔야 했고, 전부 공시한다

| 원본 | 미래 사용 | 온라인 대체 |
|---|---|---|
| KMeans 코드북을 **문서 전체 토큰**에 적합 | 전 문서 | `partial_fit` 로 **도착한 토큰까지만** 갱신 |
| 우측 창 w 문장 | **앞으로 w 문장** | 그대로 필요 ⇒ **bounded-lookahead (delay=w)** |
| `argrelmax(order=20)` | 앞뒤 20점 | 과거 신호의 **running mean + c·std** 임계 |

⇒ 이 구현은 **zero-lookahead 가 아니다.** 우측 창 때문에 구조적으로 `delay = w` 다.
이름에 박는다: `EmbeddingDelta-online-d{w}`.
🔴 원본과 **다른 방법**이므로 원 논문 수치와 대조하지 않는다. 서사 무대의 온라인 베이스라인으로만 쓴다.

## 임계
`c` 는 원본의 `mean + std/2` 규칙(`mycode.py::change_indices`)을 따라 **0.5 고정**.
문서별 튜닝 없음.

    .venv-stss/bin/python EACL/methods/embedding_delta/run_narrative_online.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))

PRED = REPO / "outputs/experiments/2026-07-29_baseline_compare/preds"
OUT = REPO / "outputs/experiments/2026-08-03_narrative_online"


def normed_cosine(a_hist: np.ndarray, b_hist: np.ndarray, standard: np.ndarray) -> float:
    """원본 `mycode.py::normed_cosine.func` — 히스토그램을 L2 정규화 후 전역 분포를 빼고 코사인 거리."""
    from scipy.spatial.distance import cosine

    if a_hist.sum() == 0 or b_hist.sum() == 0:
        return 0.0
    a = a_hist / np.linalg.norm(a_hist) - standard
    b = b_hist / np.linalg.norm(b_hist) - standard
    if not np.any(a) or not np.any(b):
        return 0.0
    d = cosine(a, b)
    return 0.0 if np.isnan(d) else float(d)


def run_doc(sents: list[str], nlp, k: int, w: int, smooth: int, c: float,
            warmup: int, right_w: int | None = None, seed: int = 0) -> list[int]:
    """`right_w` 로 **우측 창만** 줄여 lookahead 를 낮춘다 (좌측 창은 w 그대로).

    원본은 좌우 대칭(각 w)이라 구조적으로 `delay = w` 다. 우측 창을 `right_w` 문장으로
    줄이면 `delay = right_w - 1` 이 되고, `right_w=1` 이면 **zero-lookahead** 다 —
    전이 g-1 을 문장 g 가 도착한 그 순간 확정한다.

    🔴 이건 **원본과 다른 방법**이다(대칭 창이 원 정의의 일부). DialSTART 때와 같은 개조이며
    행 이름과 캡션에 반드시 공시한다. 좌측 창은 인과적이라 w 를 유지한다 — 좌우를 같이
    줄이면 신호 자체가 약해져 'zero 라서 나쁘다' 와 '창이 좁아서 나쁘다' 가 섞인다.

    `right_w=None`(기본) 이면 원본 경로 그대로다 — 기존 `-d25` 행은 비트 단위로 불변.
    """
    from sklearn.cluster import MiniBatchKMeans

    n = len(sents)
    sent_vecs = [np.asarray([t.vector for t in nlp(s)], dtype=np.float32) for s in sents]

    km, buf = None, []
    labels_per_sent: list[np.ndarray] = []
    counts = np.zeros(k, dtype=float)          # 전역 분포 추정 (도착한 토큰까지만)
    sig: list[float] = []                      # gap g 의 delta 신호 (g = 0..n-2)
    out = [0] * (n - 1)
    seen_sig: list[float] = []

    for t in range(n):
        v = sent_vecs[t]
        # ── 코드북 온라인 갱신 (도착한 토큰까지만) ──
        if km is None:
            buf.append(v)
            tot = sum(len(x) for x in buf)
            if tot >= k:
                # 🔴 seed 없이 두면 코드북이 실행마다 달라져 점수가 흔들린다(실측 Score
                # 0.2811 ↔ 0.2809). CLAUDE.md '모든 randomness 에 seed 고정' 위반이라 박는다.
                km = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=100,
                                     random_state=seed)
                km.partial_fit(np.concatenate(buf, axis=0))
                for x in buf:
                    lab = km.predict(x) if len(x) else np.array([], dtype=int)
                    labels_per_sent.append(lab)
                    counts += np.bincount(lab, minlength=k)
                buf = []
            else:
                labels_per_sent.append(None)   # 아직 코드북 없음
                continue
        else:
            if len(v):
                km.partial_fit(v)
                lab = km.predict(v)
            else:
                lab = np.array([], dtype=int)
            labels_per_sent.append(lab)
            counts += np.bincount(lab, minlength=k)

        # ── gap g 를 이제(우측 창이 채워졌으므로) 평가한다 ──
        #   원본 경로   : g = t - w              ⇒ delay = w
        #   우측창 축소 : g = t - right_w + 1    ⇒ delay = right_w - 1 (right_w=1 이면 0)
        rw = w if right_w is None else right_w
        g = (t - w) if right_w is None else (t - rw + 1)
        if g < w:                              # 좌측 창도 채워져야 한다
            continue
        left = [labels_per_sent[i] for i in range(g - w, g) if labels_per_sent[i] is not None]
        right = [labels_per_sent[i] for i in range(g, min(n, g + rw)) if labels_per_sent[i] is not None]
        if not left or not right:
            continue
        std = counts / (np.linalg.norm(counts) or 1.0)
        a = np.bincount(np.concatenate(left), minlength=k).astype(float)
        b = np.bincount(np.concatenate(right), minlength=k).astype(float)
        s = normed_cosine(a, b, std)
        sig.append(s)

        # ── 평활: 과거 smooth 개 이동평균 (미래 미사용) ──
        sm = float(np.mean(sig[-smooth:]))
        seen_sig.append(sm)
        if len(seen_sig) < warmup:
            continue
        past = np.asarray(seen_sig[:-1])
        thr = past.mean() + c * past.std()
        if sm > thr and 0 <= g - 1 < len(out):
            out[g - 1] = 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1000, help="원본 num_clusters")
    ap.add_argument("-w", type=int, default=25, help="원본 window_size (= delay)")
    ap.add_argument("--smooth", type=int, default=10, help="원본 smoothing_size")
    ap.add_argument("--c", type=float, default=0.5, help="원본 mean+std/2 규칙")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="MiniBatchKMeans 코드북 seed")
    ap.add_argument("--right-w", type=int, default=None,
                    help="우측 창 문장 수. 지정하면 delay = right_w - 1 (1 이면 zero-lookahead). "
                         "미지정이면 원본 대칭 창(delay = w)")
    ap.add_argument("--datasets", default="dev3,test3")
    args = ap.parse_args()

    import spacy

    from hi_ontop.dts_scoring import score_dialogues     # noqa: PLC0415

    # 🔴 `baseline_compare.load_dataset` 은 import 만으로 torch 를 끌어온다(M4/M5 학습 코드 경유).
    # 이 스크립트는 spaCy 전용 환경(`.venv-stss`)에서 도는데 거기엔 torch 가 없다.
    # 2GB 설치 대신 **같은 규약의 로더를 여기 둔다** — 파일 경로·정렬·문장 조립 규칙은 동일하다.
    def load_dataset(name: str):
        if name == "dev3":
            src = [(d, REPO / f"outputs/eacl_phase0/{d}_michelmann_master.jsonl")
                   for d in ("pieman", "tunnel", "monkey")]
        elif name.startswith("test3"):
            tag = "goldauthor" if name.endswith("goldauthor") else "goldtopn"
            src = [(d, REPO / f"outputs/eacl_phase0/{d}_eventrecall_{tag}_master.jsonl")
                   for d in ("run", "gohitler", "mymotherslife")]
        else:
            raise ValueError(name)
        out = []
        for doc, p in src:
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            rows.sort(key=lambda r: r["transition"])
            sents = [rows[0]["previous"]] + [r["current"] for r in rows]
            gold = np.asarray([int(r["boundary"]) for r in rows], dtype=np.int8)
            out.append((doc, sents, gold))
        return out

    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 10_000_000
    delay = args.w if args.right_w is None else args.right_w - 1
    name = ("EmbeddingDelta-online-zero" if delay == 0
            else f"EmbeddingDelta-online-d{delay}")
    print(f"{name} · k={args.k} w={args.w} right_w={args.right_w or args.w} "
          f"smooth={args.smooth} c={args.c} ⇒ delay={delay}")

    rows = {}
    for ds in args.datasets.split(","):
        data = load_dataset(ds)
        preds = {}
        for doc, sents, _g in data:
            preds[doc] = run_doc(sents, nlp, args.k, args.w, args.smooth, args.c, args.warmup,
                                 right_w=args.right_w, seed=args.seed)
            print(f"  {doc}: 문장 {len(sents)} · 예측 {sum(preds[doc])}")
        docs = [d for d, _s, _g in data]
        G = [g for _d, _s, g in data]
        ys = [np.concatenate([np.asarray(g, np.int8), [0]]).tolist() for g in G]
        ps = [np.concatenate([np.asarray(preds[d][:len(g)], np.int8), [0]]).tolist()
              for d, g in zip(docs, G)]
        sc = score_dialogues(ys, ps)
        PRED.mkdir(parents=True, exist_ok=True)
        (PRED / f"{name}_{ds}.json").write_text(json.dumps(preds))
        rows[ds] = {"pk": sc["pk"], "wd": sc["wd"], "f1": sc["f1"], "score": sc["score"],
                    "n_pred": sum(sum(v) for v in preds.values()),
                    "n_gold": int(sum(int(g.sum()) for g in G))}
        print(f"  {name} {ds:<6} Pk {sc['pk']:.4f} WD {sc['wd']:.4f} F1 {sc['f1']:.4f} "
              f"Score {sc['score']:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "embedding_delta_online.json"
    f.write_text(json.dumps({"config": vars(args), "rows": rows}, ensure_ascii=False, indent=2))
    print(f"\n저장: {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
