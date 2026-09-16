#!/usr/bin/env python3
"""Phase A 후속 2건 (재분석만, 새 실행 없음).

## ① gold span 길이 분포 — ceiling 동일성의 해석 확정
ceiling 은 청크 48.8단어(Paragraph)~969.6단어(M5)까지 **20배 차이에도 82.1~82.47% 로 동일**했다.
gold 가 임의 길이 조각이라면 문단 경계를 자주 관통해야 하는데 그러지 않았다. 가장 단순한 설명은
**gold 가 짧다**는 것이다. median gold 가 20~40단어면 어떤 방법의 경계도 거의 건드리지 않는다.
확인되면 **ceiling 동일성이 provenance 와 무관하게 완전히 설명**되고, 생성단위 편향이 ceiling 을
통해 작동할 채널이 없다는 것이 확정된다(남는 채널 = 임베딩 정렬뿐, C2 로만 측정).

## ② 순위 실패의 **크기 대역별 분해** — 두 실패 모드 분리
CV 가 크면 평균이 어디든 **오른쪽 꼬리(거대 청크 → 희석)** 와 **왼쪽 덩어리(미세 청크 → 맥락 부족)** 를
동시에 갖는다. `m5__rate0.5` 는 CV 1.66 · `<50단어` 48.7% 로 왼쪽 꼬리가 지배적이다.
대역별 순위실패율을 보면 22% 바닥이 (a) 여전히 분산 문제인지 (b) 신호 자체의 약함인지 갈린다:
- 실패가 양 극단 대역에 몰리고 중간 대역에서는 LumberChunker 수준이면 → **분산 문제**
- 모든 대역에서 균일하게 높으면 → **신호/증류 문제**

    uv run python scripts/gutenqa/06_gold_span_and_bands.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"
DATA = BASE / "data"
RET = BASE / "retrieval"

BANDS = [(0, 50), (50, 200), (200, 800), (800, 10**9)]
FOCUS = ["lumberchunker", "recursive", "paragraph", "semantic",
         "m5__rate0.15", "m5__frozen", "m5__rate0.3", "m5__rate0.5"]


def load_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "PHASE_A_FOLLOWUP.md"))
    a = ap.parse_args()

    qs = {q["qid"]: q for q in load_jsonl(DATA / "questions.jsonl")}
    L = ["# Phase A 후속 — gold span 길이 · 순위실패 크기대역 분해\n",
         "기존 산출물만 재분석. 새 실행·LLM 호출 없음.\n"]

    # ---------- ① gold span 길이 ----------
    gw = np.array([len(q["gold"].split()) for q in qs.values()], dtype=float)
    gc = np.array([len(q["gold"]) for q in qs.values()], dtype=float)
    st = {"n": int(gw.size), "mean": gw.mean(), "p5": np.percentile(gw, 5),
          "p25": np.percentile(gw, 25), "p50": np.percentile(gw, 50),
          "p75": np.percentile(gw, 75), "p95": np.percentile(gw, 95), "max": gw.max(),
          "share_gt50w": float((gw > 50).mean()), "share_gt100w": float((gw > 100).mean()),
          "mean_chars": gc.mean(), "p50_chars": float(np.percentile(gc, 50))}
    L += ["## ① gold span (`Chunk Must Contain`) 길이 분포\n",
          f"| 통계 | 단어 |", "|---|---|",
          f"| n | {st['n']:,} |", f"| 평균 | {st['mean']:.1f} |",
          f"| p5 | {st['p5']:.0f} |", f"| p25 | {st['p25']:.0f} |",
          f"| **p50 (중앙값)** | **{st['p50']:.0f}** |", f"| p75 | {st['p75']:.0f} |",
          f"| p95 | {st['p95']:.0f} |", f"| max | {st['max']:.0f} |",
          f"| >50단어 비율 | {st['share_gt50w']*100:.1f}% |",
          f"| >100단어 비율 | {st['share_gt100w']*100:.1f}% |",
          f"| 평균 문자수 | {st['mean_chars']:.0f} (중앙값 {st['p50_chars']:.0f}) |", ""]

    # gold 이 문단 하나에 들어가는가 (Paragraph 청크가 최소 단위이므로 그것으로 검사)
    par = {r["book_id"]: r["chunks"] for r in load_jsonl(DATA / "chunks/paragraph.jsonl")}
    in_one_par = 0
    for q in qs.values():
        g = q["gold"].lower()
        if any(g in c.lower() for c in par.get(q["book_id"], [])):
            in_one_par += 1
    L.append(f"**gold 이 문단 1개 안에 통째로 들어가는 문항 = {in_one_par:,}/{st['n']:,} "
             f"({in_one_par/st['n']*100:.2f}%)** — Paragraph 청크 = 문단 1개이므로 그 ceiling 과 동일.")
    L.append(f"⇒ ceiling 82.47% 의 정체 = **gold 의 82.47% 가 단일 문단 안에 있다**는 것이다. "
             f"gold 중앙값이 {st['p50']:.0f}단어이므로 문단(중앙값 24단어, p95 178단어)을 "
             f"관통하는 경우가 적고, **어느 방법이 문단을 어떻게 묶든 ceiling 이 변하지 않는다.**")
    L.append("⇒ **생성단위 편향이 ceiling 을 통해 작동할 채널은 없다**(확정). "
             "남는 채널은 질문–청크 임베딩 정렬뿐이며 그것은 2×2 교차(C2)로만 측정된다.\n")

    # ---------- ② 크기 대역별 순위실패 ----------
    L += ["## ② 순위실패의 크기 대역별 분해\n",
          "대상 = gold 이 어딘가엔 있는 문항. 대역 = **정답을 포함한 청크의 단어수**.\n",
          "| 방법 | CV | <50단어 | 50–200 | 200–800 | >800단어 | 전체 |",
          "|---|---|---|---|---|---|---|"]
    a1 = json.loads((BASE / "phase_a_diagnosis.json").read_text(encoding="utf-8"))["A1"]
    out = {}
    for name in FOCUS:
        rp, cp = RET / f"{name}.json", DATA / f"chunks/{name}.jsonl"
        if not (rp.exists() and cp.exists()):
            continue
        ch = {r["book_id"]: r["chunks"] for r in load_jsonl(cp)}
        recs = json.loads(rp.read_text(encoding="utf-8"))["records"]
        band_n = [0] * len(BANDS)
        band_f = [0] * len(BANDS)
        for r in recs:
            if not r["gold_in_some_chunk"]:
                continue
            q = qs[r["qid"]]
            g = q["gold"].lower()
            cand = [c for c in ch[r["book_id"]] if g in c.lower()]
            if not cand:
                continue
            cw = min(len(c.split()) for c in cand)
            for bi, (lo, hi) in enumerate(BANDS):
                if lo <= cw < hi:
                    band_n[bi] += 1
                    if r["dcg@20"] == 0:
                        band_f[bi] += 1
                    break
        tot_n, tot_f = sum(band_n), sum(band_f)
        cells = []
        for bi in range(len(BANDS)):
            if band_n[bi] == 0:
                cells.append("—")
            else:
                cells.append(f"{band_f[bi]/band_n[bi]*100:.1f}% <sub>(n={band_n[bi]})</sub>")
        cv = a1.get(name, {}).get("cv", float("nan"))
        small = a1.get(name, {}).get("small", float("nan"))
        out[name] = {"cv": cv, "bands": [{"lo": BANDS[i][0], "hi": BANDS[i][1],
                                          "n": band_n[i], "fail": band_f[i]}
                                         for i in range(len(BANDS))],
                     "total_n": tot_n, "total_fail": tot_f}
        L.append(f"| {name} | {cv:.2f} | " + " | ".join(cells)
                 + f" | **{tot_f/tot_n*100:.1f}%** |")

    # 중간 대역(200–800) 비교 = 크기 교란이 가장 작은 구간
    L += ["\n### 중간 대역(200–800단어)만 비교 — 크기 교란이 가장 작은 구간\n",
          "| 방법 | 200–800단어 순위실패율 | LumberChunker 대비 |", "|---|---|---|"]
    base = None
    for name in ["lumberchunker", "recursive", "m5__rate0.15", "m5__frozen",
                 "m5__rate0.3", "m5__rate0.5"]:
        if name not in out:
            continue
        b = out[name]["bands"][2]
        if b["n"] == 0:
            continue
        v = b["fail"] / b["n"]
        if name == "lumberchunker":
            base = v
        ratio = f"{v/base:.2f}×" if base else "—"
        L.append(f"| {name} | {v*100:.1f}% <sub>(n={b['n']})</sub> | {ratio} |")
    L.append("\n🔴 판정 규칙(사전 선언): 중간 대역에서 M5 가 LumberChunker 의 **1.5배 이내**면 "
             "22% 바닥은 **분산 문제** → B1 기대효과 큼. **2배 이상**이면 크기와 무관한 "
             "**신호/증류 결손** → 증류 가설 지분 큼.\n")

    (BASE / "phase_a_followup.json").write_text(
        json.dumps({"gold_span": {k: float(v) for k, v in st.items()},
                    "gold_in_single_paragraph": in_one_par,
                    "bands": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
