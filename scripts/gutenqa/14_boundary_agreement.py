#!/usr/bin/env python3
"""체크포인트 세대 간 경계 일치도 (하류 재검증 1차 진단).

## 왜 이걸 보는가
"체크포인트가 정말 다른 분절기인가, 아니면 같은 걸 미세하게 흔든 것인가"를
**하류에 돈을 쓰기 전에** 판별한다. 참조 눈금 2개가 이미 있다:

- **M5 ↔ M6 경계 F1 0.951~0.970** → LongBench 4태스크 점수가 **소수점까지 동일**했다 (D-I14/15)
- **W0 ↔ W3 경계 F1 0.771** → "seed 노이즈가 아니라 다른 분절기" (D-L15)

⇒ 🔒 **사전 선언: F1 ≥ 0.95 면 하류 재실행이 무의미하다는 신호**로 읽는다 (PREREG §1 진단 항목).

## 무엇을 재는가
1. **경계 F1** — 임계 규칙을 고정하고 체크포인트만 바꿨을 때의 문단 절단점 일치도.
   규칙이 고정이므로 차이는 전부 신호에서 온다.
2. **신호 상관** — 전이별 s 의 Spearman ρ. 임계와 완전히 독립인 순위 일치도.
3. **밀도** — 절단 수. 입도 교란(하류 지배 변수)을 점수 열람 **전에** 기록해 둔다.

    uv run python scripts/gutenqa/14_boundary_agreement.py \
        --out-base outputs/experiments/2026-07-31_gutenqa_ckpt_sweep
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/eacl_phase0"))
sys.path.insert(0, str(REPO_ROOT / "scripts/eacl_phase1"))
OLD = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"

REF = {"M5↔M6 (하류 점수 동일)": "0.951~0.970", "W0↔W3 (다른 분절기)": "0.771"}


def f1(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    tp = len(a & b)
    if tp == 0:
        return 0.0
    p, r = tp / len(a), tp / len(b)
    return 2 * p * r / (p + r)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / d) if d else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--rules", nargs="+", default=["adaptive", "ewma20"],
                    help="경계 F1 을 잴 임계 규칙 (규칙 고정, 체크포인트만 변화)")
    a = ap.parse_args()
    base = Path(a.out_base)

    spec = importlib.util.spec_from_file_location(
        "seg", REPO_ROOT / "scripts/gutenqa/03_segment_m5.py")
    seg = importlib.util.module_from_spec(spec); spec.loader.exec_module(seg)
    cfg = json.loads((REPO_ROOT / "configs/m5_default.json").read_text(encoding="utf-8"))
    ad_mode = cfg["threshold"].get("mode", "online_nf")
    ad_c = float(cfg["threshold"].get("c", 1.0))

    CACHES = {"w0": OLD / "m5_scores",
              "w3": base / "m5_scores_w3",
              "i2": base / "m5_scores_i2"}
    gens = [g for g, p in CACHES.items() if p.exists() and any(p.iterdir())]
    print(f"[진단] 세대 {gens} · 임계 {ad_mode} c={ad_c} (고정)\n")

    books = list(seg.load_books())
    cuts: dict[tuple[str, str], dict[int, set[int]]] = {}
    sig: dict[str, dict[int, np.ndarray]] = {g: {} for g in gens}
    ncut: dict[tuple[str, str], list[int]] = {}

    for bk in books:
        bid, paras = bk["book_id"], bk["paragraphs"]
        sents, starts = seg.paragraphs_to_sentences(paras)
        if len(sents) < 2:
            continue
        for g in gens:
            f = CACHES[g] / f"book{bid:03d}.npz"
            if not f.exists():
                continue
            sc = np.load(f)["scores"]
            sig[g][bid] = sc
            for rule in a.rules:
                if rule == "adaptive":
                    c = seg.cuts_adaptive(sc, starts, ad_c, ad_mode)
                else:
                    c = seg.cuts_adaptive(sc, starts, ad_c, "ewma", warmup=int(rule[4:]))
                cuts.setdefault((g, rule), {})[bid] = set(c)
                ncut.setdefault((g, rule), []).append(len(c))

    print("=== 참조 눈금 ===")
    for k, v in REF.items():
        print(f"  {k:32s} F1 = {v}")
    print("  🔒 F1 ≥ 0.95 → 하류 재실행 무의미 신호 (사전 선언)\n")

    out = {"reference": REF, "threshold": {"mode": ad_mode, "c": ad_c}, "pairs": {}, "density": {}}

    print("=== 경계 F1 (규칙 고정, 체크포인트만 변화) ===")
    hdr = f"{'규칙':10s} {'쌍':12s} {'경계F1':>8s} {'신호 Spearman':>14s}"
    print(hdr); print("─" * len(hdr))
    pairs = [(x, y) for i, x in enumerate(gens) for y in gens[i + 1:]]
    for rule in a.rules:
        for x, y in pairs:
            common = sorted(set(cuts.get((x, rule), {})) & set(cuts.get((y, rule), {})))
            if not common:
                continue
            fs = [f1(cuts[(x, rule)][b], cuts[(y, rule)][b]) for b in common]
            rs = [spearman(sig[x][b], sig[y][b]) for b in common
                  if b in sig[x] and b in sig[y] and sig[x][b].shape == sig[y][b].shape]
            mf, mr = float(np.mean(fs)), float(np.nanmean(rs)) if rs else float("nan")
            flag = "  ← 0.95↑ 재실행 무의미" if mf >= 0.95 else ""
            print(f"{rule:10s} {x}↔{y:8s} {mf:>8.3f} {mr:>14.3f}{flag}")
            out["pairs"][f"{rule}|{x}-{y}"] = {"boundary_f1": round(mf, 4),
                                               "signal_spearman": round(mr, 4),
                                               "n_books": len(common)}

    print("\n=== 밀도 (입도 교란 — 점수 열람 전에 기록) ===")
    hdr2 = f"{'규칙':10s} {'세대':5s} {'절단/book':>10s} {'vs W0':>8s} {'신호 s 평균':>12s}"
    print(hdr2); print("─" * len(hdr2))
    for rule in a.rules:
        base_n = np.mean(ncut[("w0", rule)]) if ("w0", rule) in ncut else None
        for g in gens:
            if (g, rule) not in ncut:
                continue
            m = float(np.mean(ncut[(g, rule)]))
            smean = float(np.mean([s.mean() for s in sig[g].values()])) if sig[g] else float("nan")
            rel = f"{m/base_n*100-100:+.1f}%" if base_n else "—"
            print(f"{rule:10s} {g:5s} {m:>10.1f} {rel:>8s} {smean:>12.3f}")
            out["density"][f"{rule}|{g}"] = {"cuts_per_book": round(m, 2),
                                             "signal_mean": round(smean, 4)}

    p = base / "boundary_agreement.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
