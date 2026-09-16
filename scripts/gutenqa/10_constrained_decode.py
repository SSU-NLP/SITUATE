#!/usr/bin/env python3
"""제약 하의 디코딩 — 길이 밴드가 **먼저**, 임계 규칙은 밴드 **안에서만** (D-B45).

## B1 과 무엇이 다른가
| | B1 (`08_b1_length_normalize.py`) | 이 스크립트 |
|---|---|---|
| 방식 | **사후 수리** — 자른 뒤 크면 쪼개고 작으면 붙임 | **제약 하 디코딩** — 나쁜 절단을 애초에 안 만듦 |
| 온라인 | ❌ 병합이 최종 청크 크기를 알아야 함 = 오프라인 | ✅ 단어 카운터 하나 든 전방 패스 |
| 경계 집합 | 약한 경계를 사후에 **삭제** | MIN 미달 구간에서 발화를 **억제** → 다음 자격 봉우리가 경계 |

M5 의 핵심 주장이 **온라인**인데 B1 의 병합이 그걸 깨뜨렸다. 이 방식은 유지한다.

## 알고리즘
`w` = 마지막 절단 이후 누적 단어수. 후보 `C` = 임계 규칙이 낸 경계를 문단으로 스냅한 집합.

    w < MIN         → 후보여도 자르지 않는다 (강제 계속)
    MIN ≤ w < MAX   → 후보면 자른다 (임계 규칙이 결정)
    w ≥ MAX         → 강제 절단. 위치는 **그 구간에서 점수 최고인 합법 문단 경계로 소급**

마지막 줄이 설계문서 §4.8 의 *"delayed commit = K′-문장 입자 스무딩"* 이다 — 창 안에서 소급해
최적점을 고르는 것. 임의 추가가 아니라 문서 승인 범위다. 강제 절단은 **thresholded 후보가 아니라
원신호 최댓값**을 쓴다(그 구간에 합법 후보가 없었기 때문에 강제 절단에 이른 것이므로).

🔴 문단 중간은 절대 자르지 않는다(gold 부분문자열 보존 = 측정 타당성, D-B39 §6).
바운드는 `b1_length_bounds.json`(MIN = dev gold p95, MAX = 인코더 용량) 그대로 — 새 튜닝 없음.

    uv run python scripts/gutenqa/10_constrained_decode.py --cells adaptive ewma20 ewma50 fixedq
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

FIXEDQ_RATE = 0.0642        # ⛔ 폐기 규칙 (애블레이션용)


def load_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def constrained_cuts(cand: set[int], pw: list[int], bs: np.ndarray,
                     min_w: int, max_w: int) -> tuple[list[int], dict]:
    """길이 밴드 우선 · 임계 후보는 밴드 안에서만. 반환 = 절단 문단 인덱스 + 통계."""
    n = len(pw)
    cum = np.concatenate([[0], np.cumsum(pw)])
    words = lambda a, b: int(cum[b] - cum[a])            # noqa: E731
    st = {"by_threshold": 0, "forced_max": 0, "suppressed_min": 0}
    cuts: list[int] = []
    a, j = 0, 1
    while j <= n - 1:
        w = words(a, j)
        if w >= max_w:                                    # 강제 절단 + delayed commit
            legal = [k for k in range(a + 1, j + 1) if words(a, k) >= min_w]
            k = max(legal, key=lambda x: bs[x]) if legal else j
            cuts.append(k); st["forced_max"] += 1
            a, j = k, k + 1
            continue
        if j in cand:
            if w >= min_w:
                cuts.append(j); st["by_threshold"] += 1
                a, j = j, j + 1
                continue
            st["suppressed_min"] += 1                     # 후보였지만 너무 작아 억제
        j += 1
    return cuts, st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["adaptive", "ewma20", "ewma50", "fixedq"],
                    help="후보 생성 규칙: adaptive | ewma<span> | fixedq")
    ap.add_argument("--device", default=None)
    # ── 체크포인트 세대 분리 (2026-08-01, 하류 재검증) ──────────────────────
    ap.add_argument("--scores-dir", default=None,
                    help="점수 캐시 디렉터리 절대경로. 생략하면 구 W0 캐시를 쓴다.")
    ap.add_argument("--cell-prefix", default="m5__", help="셀 접두. 예: m5i2__ / m5w3__")
    ap.add_argument("--out-base", default=None, help="청크·리포트 산출 루트")
    a = ap.parse_args()

    global CACHE
    out_base = Path(a.out_base) if a.out_base else BASE
    if a.scores_dir:
        CACHE = Path(a.scores_dir)
    out_chunks = out_base / "chunks"
    out_chunks.mkdir(parents=True, exist_ok=True)
    print(f"[cd] 점수캐시={CACHE} · 접두={a.cell_prefix} · 산출={out_chunks}")

    if not BOUNDS.exists():
        raise SystemExit("🔴 b1_length_bounds.json 없음 — 07b_declare_bounds.py 를 먼저 돌릴 것")
    bd = json.loads(BOUNDS.read_text(encoding="utf-8"))
    MIN_W, MAX_W = int(bd["MIN_WORDS"]), int(bd["MAX_WORDS"])
    cfg = json.loads((REPO_ROOT / "configs/m5_default.json").read_text(encoding="utf-8"))
    ad_mode = cfg["threshold"].get("mode", "online_nf")
    ad_c = float(cfg["threshold"].get("c", 1.0))
    print(f"[cd] 🔒 MIN={MIN_W} · MAX={MAX_W} (b1_length_bounds.json, 새 튜닝 없음)")
    print(f"[cd] config 임계 = {ad_mode} c={ad_c} (읽기만)")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "seg", REPO_ROOT / "scripts/gutenqa/03_segment_m5.py")
    seg = importlib.util.module_from_spec(spec); spec.loader.exec_module(seg)

    books = list(load_jsonl(DATA / "books.jsonl"))
    out: dict[str, dict[int, list[str]]] = {c: {} for c in a.cells}
    agg = {c: {"by_threshold": 0, "forced_max": 0, "suppressed_min": 0} for c in a.cells}
    stats = {c: [] for c in a.cells}

    for bi, bk in enumerate(books):
        bid, paras = bk["book_id"], bk["paragraphs"]
        sents, starts = seg.paragraphs_to_sentences(paras)
        if len(sents) < 2:
            continue
        sc = np.load(CACHE / f"book{bid:03d}.npz")["scores"]
        bs = seg.para_boundary_scores(paras, sc) if hasattr(seg, "para_boundary_scores") else None
        if bs is None:                                    # 08 에만 있으므로 직접 계산
            bs = np.full(len(paras), -np.inf)
            for j in range(1, len(paras)):
                i = starts[j] - 1
                if 0 <= i < len(sc):
                    bs[j] = float(sc[i])
        pw = [len(p.split()) for p in paras]

        for cell in a.cells:
            if cell == "adaptive":
                cand = seg.cuts_adaptive(sc, starts, ad_c, ad_mode)
            elif cell.startswith("ewma"):
                cand = seg.cuts_adaptive(sc, starts, ad_c, "ewma", warmup=int(cell[4:]))
            elif cell == "fixedq":
                cand = seg.cuts_fixedq(sc, starts, FIXEDQ_RATE)
            else:
                raise SystemExit(f"알 수 없는 셀 {cell!r}")
            cuts, st = constrained_cuts(set(cand), pw, bs, MIN_W, MAX_W)
            for k in agg[cell]:
                agg[cell][k] += st[k]
            edges = [0] + cuts + [len(paras)]
            chunks = ["\n".join(paras[x:y]).strip() for x, y in zip(edges[:-1], edges[1:])]
            chunks = [c for c in chunks if c]
            out[cell][bid] = chunks
            stats[cell] += [len(c.split()) for c in chunks]
        if (bi + 1) % 20 == 0:
            print(f"  {bi+1}/{len(books)} books", flush=True)

    report = {"bounds": {"MIN_WORDS": MIN_W, "MAX_WORDS": MAX_W}, "cells": {}}
    print()
    for cell in a.cells:
        name = f"{a.cell_prefix}cd_{cell}"
        p = out_chunks / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for bid in sorted(out[cell]):
                f.write(json.dumps({"book_id": bid, "chunks": out[cell][bid]},
                                   ensure_ascii=False) + "\n")
        w = np.array(stats[cell], dtype=float)
        s = {"n": int(w.size), "mean": float(w.mean()), "cv": float(w.std() / w.mean()),
             "p5": float(np.percentile(w, 5)), "p50": float(np.percentile(w, 50)),
             "p95": float(np.percentile(w, 95)), "max": float(w.max()),
             "over_max": float((w > MAX_W).mean()), "under_min": float((w < MIN_W).mean()),
             "chunks_per_book": round(w.size / len(out[cell]), 1), "ops": agg[cell]}
        report["cells"][name] = s
        o = agg[cell]
        tot = o["by_threshold"] + o["forced_max"]
        print(f"== {name} ==")
        print(f"  절단 {tot:,} = 임계 {o['by_threshold']:,} ({o['by_threshold']/max(1,tot)*100:.1f}%) "
              f"+ 강제MAX {o['forced_max']:,} ({o['forced_max']/max(1,tot)*100:.1f}%) "
              f"· MIN 억제 {o['suppressed_min']:,}")
        print(f"  {s['chunks_per_book']}/book · 평균 {s['mean']:.1f}단어 · CV {s['cv']:.2f} · "
              f"p50 {s['p50']:.0f} · p95 {s['p95']:.0f} · >MAX {s['over_max']*100:.1f}% · "
              f"<MIN {s['under_min']*100:.1f}%")
    rep_name = "cd_report.json" if a.cell_prefix == "m5__" else f"cd_report_{a.cell_prefix.rstrip('_')}.json"
    (out_base / rep_name).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"\nsaved -> {out_base/rep_name}")


if __name__ == "__main__":
    main()
