#!/usr/bin/env python3
"""B1 — 임계 규칙을 **분위수 → 분위수 + 범위제약**으로 교체 (D-B42 사전등록).

## 무엇이 아닌가
"길이 통제를 사후에 얹은 것" 이 **아니다.** 분할·병합 지점을 **모델 자기 신호**(M5 전이 점수
s = Σ_k P(reset_k))로 고르므로, 임계 규칙 자체가 바뀐 것이다. EM-LLM 의
`min_block_size`/`max_block_size` 와 동형이며, LumberChunker 가 프롬프트로 하는 일
(*"Avoid very long groups… keeping groups manageable"*)과 Recursive 가 문자 예산으로 하는 일을
코드로 하는 것이다.

## 규칙 (사전 동결)
- **상한 초과** (>MAX_WORDS): 그 청크 **내부의 문단 경계** 중 M5 점수가 가장 높은 곳에서 분할.
  분할 후 각 조각에 **재귀** 적용. 내부 문단 경계가 없으면(단일 문단) 그대로 둔다 —
  🔴 문단 중간은 절대 자르지 않는다(gold 부분문자열 보존 = 측정 타당성, D-B39 §6).
- **하한 미달** (<MIN_WORDS): 좌/우 이웃 중 **공유 경계의 M5 점수가 더 낮은 쪽**과 병합.
  (점수가 낮다 = 그 경계가 애초에 약했다.) 변화가 없을 때까지 반복, 최대 `--merge-iters` 회.
- 적용 순서 = **분할 먼저, 병합 나중** (분할이 새로운 하한 미달을 만들 수 있으므로).

## 경계 조건
- MIN_WORDS/MAX_WORDS 는 `b1_length_bounds.json` 에서 읽는다 — **앵커 학습 코퍼스에서 도출,
  GutenQA 미열람**(`07_derive_length_bounds.py`).
- 상한·하한이 충돌하는 극단(단일 문단이 MAX 초과)은 손대지 않고 개수를 보고한다.

    uv run python scripts/gutenqa/08_b1_length_normalize.py --cells m5__rate0.15 m5__frozen m5__rate0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/eacl_phase0"))
sys.path.insert(0, str(REPO_ROOT / "scripts/eacl_phase1"))

BASE = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"
DATA = BASE / "data"
CACHE = BASE / "m5_scores"
BOUNDS = BASE / "b1_length_bounds.json"


def load_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def para_boundary_scores(paras: list[str], scores: np.ndarray) -> np.ndarray:
    """문단 j(j>=1)를 여는 전이의 M5 점수. 길이 = len(paras); [0] 은 미사용(-inf)."""
    from build_narrative_pack import split_sentences
    from build_pg_shorts import stage0_piece
    starts, n = [], 0
    for p in paras:
        starts.append(n)
        s, _ = stage0_piece(split_sentences(p))
        n += len(s)
    out = np.full(len(paras), -np.inf, dtype=np.float64)
    for j in range(1, len(paras)):
        i = starts[j] - 1
        if 0 <= i < len(scores):
            out[j] = float(scores[i])
    return out


def normalize(edges: list[int], pw: list[int], bs: np.ndarray,
              min_w: int, max_w: int, merge_iters: int) -> tuple[list[int], dict]:
    """edges = 절단 문단 인덱스 경계열 [0, ..., n_para]. pw = 문단별 단어수. bs = 문단경계 점수."""
    stat = {"split": 0, "merge": 0, "unsplittable": 0}
    cum = np.concatenate([[0], np.cumsum(pw)])
    words = lambda a, b: int(cum[b] - cum[a])                       # noqa: E731

    # ── 분할 (재귀) ────────────────────────────────────────────────
    def split_rec(a: int, b: int, depth: int = 0) -> list[int]:
        if words(a, b) <= max_w or b - a <= 1 or depth > 40:
            if words(a, b) > max_w and b - a <= 1:
                stat["unsplittable"] += 1
            return [a]
        inner = np.arange(a + 1, b)
        best = int(inner[int(np.argmax(bs[inner]))])                # 자기 신호 최댓값 지점
        stat["split"] += 1
        return split_rec(a, best, depth + 1) + split_rec(best, b, depth + 1)

    new_edges = []
    for a, b in zip(edges[:-1], edges[1:]):
        new_edges.extend(split_rec(a, b))
    new_edges.append(edges[-1])
    edges = sorted(set(new_edges))

    # ── 병합 (반복) ────────────────────────────────────────────────
    for _ in range(merge_iters):
        changed = False
        i = 0
        while i < len(edges) - 1:
            a, b = edges[i], edges[i + 1]
            if words(a, b) >= min_w or len(edges) <= 2:
                i += 1
                continue
            # 좌/우 공유 경계 점수 비교 → 낮은 쪽과 병합 (= 약한 경계를 지운다)
            left_ok, right_ok = i > 0, i + 2 < len(edges)
            if left_ok and right_ok:
                drop = i if bs[a] <= bs[b] else i + 1
            elif right_ok:
                drop = i + 1
            elif left_ok:
                drop = i
            else:
                i += 1
                continue
            edges.pop(drop)
            stat["merge"] += 1
            changed = True
        if not changed:
            break
    return edges, stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+",
                    default=["m5__rate0.15", "m5__frozen", "m5__rate0.3"])
    ap.add_argument("--merge-iters", type=int, default=10)
    ap.add_argument("--suffix", default="_b1")
    # ── 체크포인트 세대 분리 (2026-08-01, 하류 재검증) ──────────────────────
    ap.add_argument("--scores-dir", default=None,
                    help="점수 캐시 디렉터리 절대경로. 생략하면 구 W0 캐시.")
    ap.add_argument("--out-base", default=None, help="청크·리포트 산출 루트")
    ap.add_argument("--report-out", default=None,
                    help="리포트 파일명. 경로 고정이면 2회차가 1회차 통계를 덮어쓴다(실제 사고 기록됨).")
    a = ap.parse_args()

    global CACHE
    out_base = Path(a.out_base) if a.out_base else BASE
    if a.scores_dir:
        CACHE = Path(a.scores_dir)
    out_chunks = out_base / "chunks"
    out_chunks.mkdir(parents=True, exist_ok=True)
    print(f"[b1] 점수캐시={CACHE} · 산출={out_chunks}")

    if not BOUNDS.exists():
        raise SystemExit(f"🔴 {BOUNDS} 없음 — `07b_declare_bounds.py` 를 먼저 돌릴 것. "
                         "(MAX = 검색 인코더 용량, MIN = dev gold span p95. 둘 다 튜닝 아님. "
                         "앵커에서 뽑던 07 은 폐기됨 — scripts/gutenqa/_discarded/ 참조)")
    bd = json.loads(BOUNDS.read_text(encoding="utf-8"))
    MIN_W, MAX_W = int(bd["MIN_WORDS"]), int(bd["MAX_WORDS"])
    ms, ns = bd.get("max_source", {}), bd.get("min_source", {})
    print(f"[b1] 🔒 MIN_WORDS={MIN_W} · MAX_WORDS={MAX_W}  (튜닝 아님)")
    print(f"     MAX ← 인코더 {ms.get('encoder','?')} "
          f"{ms.get('max_seq_length_tokens','?')}토큰 ÷ {ms.get('tokens_per_word_measured','?')}토큰/단어")
    print(f"     MIN ← {ns.get('split','?')} gold span p{ns.get('percentile','?')} "
          f"(n={ns.get('n_questions','?')}, test 미열람)")

    books = {r["book_id"]: r["paragraphs"] for r in load_jsonl(DATA / "books.jsonl")}
    bscore, pwords = {}, {}
    for bid, paras in books.items():
        z = np.load(CACHE / f"book{bid:03d}.npz")
        bscore[bid] = para_boundary_scores(paras, z["scores"])
        pwords[bid] = [len(p.split()) for p in paras]

    report = {"bounds": {"MIN_WORDS": MIN_W, "MAX_WORDS": MAX_W}, "cells": {}}
    for cell in a.cells:
        # 신 실험 디렉터리를 먼저 보고, 없으면 구 실험(W0 셀)으로 폴백한다
        src = out_chunks / f"{cell}.jsonl"
        if not src.exists():
            src = DATA / f"chunks/{cell}.jsonl"
        if not src.exists():
            print(f"  skip {cell} (없음)")
            continue
        out_name = f"{cell}{a.suffix}"
        agg = {"split": 0, "merge": 0, "unsplittable": 0}
        per_book, wl_before, wl_after = {}, [], []
        for r in load_jsonl(src):
            bid = r["book_id"]
            paras = books[bid]
            # 기존 청크 → 문단 경계열 복원 (청크는 문단들의 연속 결합이므로 누적 길이로 역산)
            edges, n = [0], 0
            for c in r["chunks"]:
                k = len([x for x in c.split("\n") if x.strip()])
                n += k
                edges.append(min(n, len(paras)))
            edges = sorted(set(e for e in edges if 0 <= e <= len(paras)))
            if edges[-1] != len(paras):
                edges.append(len(paras))
            wl_before += [len(c.split()) for c in r["chunks"]]

            new_edges, st = normalize(edges, pwords[bid], bscore[bid],
                                      MIN_W, MAX_W, a.merge_iters)
            for k in agg:
                agg[k] += st[k]
            chunks = ["\n".join(paras[x:y]).strip()
                      for x, y in zip(new_edges[:-1], new_edges[1:])]
            chunks = [c for c in chunks if c]
            per_book[bid] = chunks
            wl_after += [len(c.split()) for c in chunks]

        outp = out_chunks / f"{out_name}.jsonl"
        with outp.open("w", encoding="utf-8") as f:
            for bid in sorted(per_book):
                f.write(json.dumps({"book_id": bid, "chunks": per_book[bid]},
                                   ensure_ascii=False) + "\n")

        def stats(w):
            w = np.array(w, dtype=float)
            return {"n": int(w.size), "mean": float(w.mean()), "cv": float(w.std() / w.mean()),
                    "p5": float(np.percentile(w, 5)), "p50": float(np.percentile(w, 50)),
                    "p95": float(np.percentile(w, 95)), "max": float(w.max()),
                    "over_max": float((w > MAX_W).mean()), "under_min": float((w < MIN_W).mean())}

        sb, sa = stats(wl_before), stats(wl_after)
        report["cells"][out_name] = {"source": cell, "ops": agg, "before": sb, "after": sa}
        print(f"\n== {cell} -> {out_name} ==")
        print(f"  분할 {agg['split']:,} · 병합 {agg['merge']:,} · 분할불가(단일문단 초과) {agg['unsplittable']:,}")
        print(f"  {'':10s} {'n':>8s} {'mean':>7s} {'CV':>6s} {'p50':>6s} {'p95':>7s} {'max':>8s} "
              f"{'>MAX':>7s} {'<MIN':>7s}")
        for lab, s in (("before", sb), ("after", sa)):
            print(f"  {lab:10s} {s['n']:>8,} {s['mean']:>7.1f} {s['cv']:>6.2f} {s['p50']:>6.0f} "
                  f"{s['p95']:>7.0f} {s['max']:>8.0f} {s['over_max']*100:>6.1f}% {s['under_min']*100:>6.1f}%")
        ok = sa["cv"] < sb["cv"]
        print(f"  🔒 조작 점검(CV 감소): {'PASS' if ok else '🔴 FAIL — 점수 변화를 해석하지 말 것'}")

    rep = out_base / (a.report_out or "b1_normalize_report.json")
    rep.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {rep}")


if __name__ == "__main__":
    main()
