#!/usr/bin/env python3
"""Phase-1 권→편 분할기 프로토타입 — **탐색 측정 전용 (판정·동결 없음)**.

질문: "1만 편"이 가능한가. 이는 **컬렉션 비율 × 분할 성공률 × 길이필터 생존율**에 달렸는데
셋 다 미측정이다. 3,662 는 *권* 수이고 단편집 한 권에 10~30편이 들어 있을 수 있다.

측정(=산출물):
  ① 층화 표본 20권 (SF 10 / non-SF 10, seed 고정)
  ② 규칙 기반 분할 (LLM 폴백 없음 — 비용·비결정성)
  ③ 컬렉션 비율 · 권당 편수 분포 · 실패 사유 · 길이필터 생존 · SF vs non-SF · 3,662 외삽(불확실성 포함)
  ④ 실패 사례 육안용 덤프

🔴 순서: **편 분할이 Stage-0(F/G)보다 먼저.** 단편집 구분선은 (a) 편 사이 = A 경계,
(b) 편 안 장면전환 = B 경계 둘 다를 표시한다. Stage-0 을 먼저 돌리면 A 경계가 gold 로 섞여
D-B4(GRANULARITY) 오류가 재발한다. 편을 자른 뒤 **각 편 내부에서만** Stage-0.

⚠️ 여기서 앵커 라벨링(5축 추출)은 하지 않는다 — 장르 정책·k 미정.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_narrative_pack import split_sentences  # noqa: E402
from build_pg_shorts import (CAT, candidates, fetch, split_volume,  # noqa: E402
                             stage0_piece)
from fetch_repcorpus import density, strip_pg  # noqa: E402

SEED = 20260725
MIN_S, MAX_S = 30, 700          # D-B11 선례
MIN_MED_W = 5.0                 # 서지/목록 잔재 배제


def is_sf(rec: dict) -> bool:
    return "science fiction" in rec["subjects"].lower()


def analyse(rec: dict, delay: float) -> dict:
    txt = fetch(rec["gid"], delay)
    if not txt:
        return {**rec, "status": "fetch_fail", "n_pieces": 0, "n_kept": 0}
    pieces, reason = split_volume(txt)
    body_sents = len(split_sentences(strip_pg(txt)))
    out = {**rec, "split_reason": reason, "n_pieces": len(pieces), "vol_sents": body_sents,
           "pieces": [], "n_kept": 0}
    for i, (title, body) in enumerate(pieces):
        sents, _ops = stage0_piece(split_sentences(body))       # ⑤ 편 내부에서만
        med = st.median(len(x.split()) for x in sents) if sents else 0
        keep = MIN_S <= len(sents) <= MAX_S and med >= MIN_MED_W
        out["pieces"].append({"i": i, "title": title, "n_sent": len(sents),
                              "median_words": med, "keep": keep,
                              "dd": round(density(sents), 3) if sents else 0.0,
                              "head": sents[0][:110] if sents else ""})
        out["n_kept"] += int(keep)
    # 상태 분류
    if reason == "chaptered_work":
        out["status"] = "chaptered_work"          # 주제어는 단편집인데 실제로는 단일 장편
    elif len(pieces) >= 3:
        out["status"] = "collection"
    elif len(pieces) == 0:
        out["status"] = "no_pieces"
    elif body_sents > MAX_S:
        out["status"] = "single_long"             # 헤딩 못 찾음 + 긴 본문 = 분할 실패 의심
    else:
        out["status"] = "single_story"            # 진짜 단일 단편
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-stratum", type=int, default=10)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "outputs/experiments/2026-07-25_phase1_split_proto")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    cands = candidates(CAT, set())     # ⑥ 격리(QC·Tier2·제외작가) 내장
    sf = [c for c in cands if is_sf(c)]
    non = [c for c in cands if not is_sf(c)]
    rng = random.Random(SEED)
    samp = rng.sample(sf, a.n_per_stratum) + rng.sample(non, a.n_per_stratum)
    print(f"후보 {len(cands)}권 (SF {len(sf)} / non-SF {len(non)}) · seed={SEED} · "
          f"층화표본 {len(samp)}권\n", flush=True)

    res = []
    for k, c in enumerate(samp):
        r = analyse(c, a.delay)
        r["stratum"] = "SF" if is_sf(c) else "nonSF"
        res.append(r)
        print(f"  [{k+1:>2}/{len(samp)}] pg{c['gid']:<6} {r['stratum']:<5} "
              f"{r['status']:<14} 편={r['n_pieces']:>3} 생존={r['n_kept']:>3}  "
              f"{c['title'][:44]}", flush=True)

    (a.out_dir / "sample_detail.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))

    # ------------------------------------------------------------- ③ 측정
    L = []
    L.append("## ③ 측정\n")
    L.append(f"표본 = 층화 무작위 20권 (SF 10 / non-SF 10), `seed={SEED}`, "
             f"후보 풀 {len(cands)}권(SF {len(sf)} / non-SF {len(non)}).\n")
    L.append("| 지표 | 전체 | SF | non-SF |")
    L.append("|---|---|---|---|")

    def col(rows, f):
        return f(rows)
    groups = [("전체", res), ("SF", [r for r in res if r["stratum"] == "SF"]),
              ("non-SF", [r for r in res if r["stratum"] == "nonSF"])]
    def row(name, fn):
        L.append(f"| {name} | " + " | ".join(str(fn(g)) for _, g in groups) + " |")
    row("권 수", lambda g: len(g))
    row("컬렉션(≥3편) 비율", lambda g: f"{sum(1 for r in g if r['status']=='collection')}/{len(g)}"
        f" = {sum(1 for r in g if r['status']=='collection')/len(g):.0%}")
    row("권당 편수 중앙값", lambda g: st.median([r["n_pieces"] for r in g]) if g else 0)
    row("권당 편수 범위", lambda g: f"{min(r['n_pieces'] for r in g)}~{max(r['n_pieces'] for r in g)}" if g else "-")
    row("총 원시 편", lambda g: sum(r["n_pieces"] for r in g))
    row("길이필터 생존 편", lambda g: sum(r["n_kept"] for r in g))
    row("권당 생존 편 평균", lambda g: f"{sum(r['n_kept'] for r in g)/len(g):.2f}" if g else 0)
    row("생존율(생존/원시)", lambda g: f"{sum(r['n_kept'] for r in g)/max(sum(r['n_pieces'] for r in g),1):.0%}")

    L.append("\n### 상태 분류 (실패 사유)\n")
    L.append("| status | 전체 | SF | non-SF | 뜻 |")
    L.append("|---|---|---|---|---|")
    MEAN = {"collection": "≥3편으로 분할 성공",
            "single_story": "헤딩 <2 이고 본문 짧음 = 진짜 단일 단편",
            "single_long": "헤딩 못 찾음 + 본문 700문장 초과 = **분할 실패 의심**",
            "chaptered_work": "章 번호 다수 = 주제어는 단편집이나 **실제로는 단일 장편**",
            "no_pieces": "편 0", "fetch_fail": "다운로드 실패"}
    for s in sorted({r["status"] for r in res}):
        cnt = [sum(1 for r in g if r["status"] == s) for _, g in groups]
        L.append(f"| `{s}` | {cnt[0]} | {cnt[1]} | {cnt[2]} | {MEAN.get(s,'')} |")

    # 외삽 (권 단위 부트스트랩)
    rng2 = random.Random(SEED)
    boots = []
    kept = [r["n_kept"] for r in res]
    for _ in range(10000):
        s = [rng2.choice(kept) for _ in kept]
        boots.append(sum(s) / len(s) * len(cands))
    boots.sort()
    lo, hi = boots[250], boots[9750]
    point = sum(kept) / len(kept) * len(cands)
    L.append(f"\n### 3,662권 → 예상 확보 편수 (외삽)\n")
    L.append(f"- 권당 생존 편 평균 **{sum(kept)/len(kept):.2f}** → 점추정 **{point:,.0f} 편**")
    L.append(f"- 권 단위 부트스트랩 95% 구간 **[{lo:,.0f}, {hi:,.0f}]** (10,000회, seed={SEED})")
    L.append(f"- ⚠️ n=20 표본이라 구간이 넓다. 층화 비율(SF {len(sf)/len(cands):.0%})을 모집단에 맞춰")
    L.append(f"  가중하지 않은 **단순 평균**이므로, SF/non-SF 생존율이 크게 다르면 편향된다(아래 표 참조).")
    sfk = [r["n_kept"] for r in res if r["stratum"] == "SF"]
    nsk = [r["n_kept"] for r in res if r["stratum"] == "nonSF"]
    w = (sum(sfk)/len(sfk)) * len(sf) + (sum(nsk)/len(nsk)) * len(non)
    L.append(f"- **층화 가중 점추정 = {w:,.0f} 편** (SF {sum(sfk)/len(sfk):.2f}/권 × {len(sf)}권 + "
             f"non-SF {sum(nsk)/len(nsk):.2f}/권 × {len(non)}권)")

    (a.out_dir / "_measure.md").write_text("\n".join(L))
    print("\n" + "\n".join(L))

    # ------------------------------------------------- ④ 실패 사례 덤프
    fails = [r for r in res if r["status"] in ("single_long", "chaptered_work", "no_pieces", "fetch_fail")]
    D = ["## ④ 분할 실패 사례 (육안 확인용)\n"]
    for r in fails[:10]:
        D.append(f"### pg{r['gid']} — `{r['status']}` ({r['stratum']})")
        D.append(f"- 제목: {r['title']}")
        D.append(f"- 저자: {r['author']}")
        D.append(f"- 주제: {r['subjects'][:120]}")
        D.append(f"- 권 본문 {r.get('vol_sents','?')} 문장 · 탐지 편 {r['n_pieces']} · 생존 {r['n_kept']}")
        for p in r["pieces"][:3]:
            D.append(f"  - 편{p['i']} `{p['title'][:40]}` {p['n_sent']}문장 med={p['median_words']} "
                     f"keep={p['keep']}\n    첫: {p['head'][:100]}")
        D.append("")
    (a.out_dir / "_failures.md").write_text("\n".join(D))
    print(f"\n실패 {len(fails)}건 → {a.out_dir/'_failures.md'}")


if __name__ == "__main__":
    main()
