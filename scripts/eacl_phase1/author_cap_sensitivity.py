#!/usr/bin/env python3
"""작가당 상한 N 민감도 분석 — 무상한 풀 대비 N별 코퍼스 구성 변화.

🔴 **"대표성 검정"이 아니라 민감도 분석이다** (codex 2026-08-02):
   무상한 풀은 다작 작가에 큰 가중치를 주므로 바람직한 기준분포가 아니다. 따라서
   "무상한 풀과 같음을 증명"하는 게 목적이 아니라, **N 선택이 코퍼스 구성을 얼마나
   바꾸는지를 곡선으로 공개**하는 것이 목적이다.

지표 (codex 권고 반영):
  A. 작가 구조 — 상한이 직접 조작하는 단위. 가장 중요
  B. 서지    — LoCC 분류 · PG 등록연도 · SF 여부(**sentinel**: 사전 분석에서 상한 민감성 관측)
  C. 형식    — 편 길이 · 대사 비율 · 서술 시점
  D. 문체    — 문장 길이 · 어휘 다양성(TTR)
  E. 5축 대리 — 인물/장소/시간/인과/의도 표지어 밀도

⚠️ `is_sf` 는 PG Subjects 기반이며 주제(subject)이지 장르(genre/form)가 아니다.
   장르 대표성의 검정이 아니라 **상한 민감 sentinel** 로만 해석한다.
⚠️ `issued` 는 PG 등록 시기이지 작품 출판연도가 아니다.

효과크기는 **원 단위**로 보고한다(codex): 범주형 = 범주별 최대 절대 비율차(%p) + 총변동거리,
연속형 = 중앙값 차이(원 단위) + KS D. 신뢰구간은 **작가 단위 cluster bootstrap**.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/eacl_phase0"), str(REPO / "scripts/eacl_phase1")]
from build_narrative_pack import split_sentences  # noqa: E402
from build_pg_shorts import split_volume, stage0_piece  # noqa: E402
from fetch_repcorpus import density, register  # noqa: E402

CACHE = REPO / "data/eacl_phase1/pg_cache"
CAT = REPO / "data/eacl_phase1/pgcat.csv"
OUT = REPO / "outputs/experiments/2026-08-02_author_cap"
MIN_SENT, MAX_SENT, MIN_MED = 30, 900, 5.0
CAPS = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48]
EXCL_AUTH_NEW = ["Pohl, Frederik"]          # dev-3 tunnel 저자
BOOT, SEED = 400, 20260802

# E. 5축 대리 표지어 (편당 문장 대비 출현율)
MARK = {
    "인물": re.compile(r"\b(he|she|they|him|her|them|his|their)\b", re.I),
    "장소": re.compile(r"\b(in|at|into|inside|outside|toward|across|through|room|house|street|city|road)\b", re.I),
    "시간": re.compile(r"\b(then|later|afterward|meanwhile|suddenly|next day|morning|night|hour|minute|year)\b", re.I),
    "인과": re.compile(r"\b(because|since|therefore|thus|so that|as a result|consequently|hence)\b", re.I),
    "의도": re.compile(r"\b(want|wanted|intend|intended|decide|decided|plan|planned|try|tried|in order to)\b", re.I),
}


def build_pool() -> list[dict]:
    meta = {int(r["Text#"]): r for r in csv.DictReader(CAT.open(encoding="utf-8"))}
    pool, seen = [], set()
    files = sorted(CACHE.glob("*.txt"), key=lambda f: int(f.stem))
    for i, f in enumerate(files):
        m = meta.get(int(f.stem))
        if m is None:
            continue
        au = m.get("Authors", "")
        if any(a in au for a in EXCL_AUTH_NEW):
            continue
        pieces, reason = split_volume(f.read_text(encoding="utf-8", errors="ignore"))
        if reason == "chaptered_work":
            continue
        for title, body in pieces:
            key = (au, title.upper())
            if title and key in seen:
                continue
            sents, _ = stage0_piece(split_sentences(body))
            if not (MIN_SENT <= len(sents) <= MAX_SENT):
                continue
            wl = [len(x.split()) for x in sents]
            if statistics.median(wl) < MIN_MED:
                continue
            seen.add(key)
            blob = " ".join(sents)
            toks = re.findall(r"[A-Za-z']+", blob.lower())
            loc = (m.get("LoCC") or "").strip()
            iss = (m.get("Issued") or "")[:4]
            rec = {"author": au, "gid": int(f.stem),
                   "locc1": loc.split(",")[0].strip()[:2] if loc else "NA",
                   "is_sf": "science fiction" in (m.get("Subjects") or "").lower(),
                   "issued_year": int(iss) if iss.isdigit() else None,
                   "n_sent": len(sents), "dialogue_density": density(sents),
                   "register": register(sents)[0],
                   "sent_words": float(np.median(wl)),
                   "ttr": len(set(toks)) / max(len(toks), 1)}
            for k, rx in MARK.items():
                rec[f"표지_{k}"] = sum(1 for s in sents if rx.search(s)) / len(sents)
            pool.append(rec)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)}권 · 누적 {len(pool):,}편", flush=True)
    return pool


def cap(pool: list[dict], n: int) -> list[dict]:
    c, out = Counter(), []
    for p in pool:
        if c[p["author"]] >= n:
            continue
        c[p["author"]] += 1
        out.append(p)
    return out


CATS = ["locc1", "is_sf", "register"]
NUMS = ["issued_year", "n_sent", "dialogue_density", "sent_words", "ttr"] + [f"표지_{k}" for k in MARK]


def diffs(sub: list[dict], full: list[dict]) -> dict:
    from scipy import stats as st
    r = {}
    for k in CATS:
        a, b = Counter(str(x[k]) for x in sub), Counter(str(x[k]) for x in full)
        keys = set(a) | set(b)
        pp = {x: (a[x] / len(sub) - b[x] / len(full)) * 100 for x in keys}
        r[k] = {"max_pp": max(abs(v) for v in pp.values()),
                "tvd": sum(abs(v) for v in pp.values()) / 200,
                "worst": max(pp, key=lambda x: abs(pp[x]))}
    for k in NUMS:
        a = [x[k] for x in sub if x[k] is not None]
        b = [x[k] for x in full if x[k] is not None]
        r[k] = {"med_sub": float(np.median(a)), "med_full": float(np.median(b)),
                "med_diff": float(np.median(a) - np.median(b)),
                "ks_D": float(st.ks_2samp(a, b).statistic)}
    return r


def author_concentration(sub: list[dict]) -> dict:
    c = Counter(p["author"] for p in sub)
    tot = sum(c.values())
    top10 = sum(v for _, v in c.most_common(10)) / tot * 100
    share = np.array(sorted(c.values(), reverse=True)) / tot
    gini = float(1 - 2 * np.sum(np.cumsum(share) - share / 2) / len(share))
    return {"n_authors": len(c), "top10_share_pct": top10, "gini": abs(gini),
            "max_author_pieces": c.most_common(1)[0][1]}


def boot_ci(pool: list[dict], n: int, key: str, sub_stat) -> tuple[float, float]:
    """작가 단위 cluster bootstrap — 편이 작가 내에서 종속이므로 작가를 재표집한다."""
    rng = np.random.default_rng(SEED)
    by = defaultdict(list)
    for p in pool:
        by[p["author"]].append(p)
    auth = list(by)
    vals = []
    for _ in range(BOOT):
        pick = rng.choice(len(auth), len(auth), replace=True)
        rp = [x for i in pick for x in by[auth[i]]]
        vals.append(sub_stat(cap(rp, n), rp))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1] 무상한 전체 편 풀 구축 (후보 전권)", flush=True)
    pool = build_pool()
    full_conc = author_concentration(pool)
    print(f"    전체 풀 {len(pool):,}편 · 작가 {full_conc['n_authors']:,}명 · "
          f"상위10 점유 {full_conc['top10_share_pct']:.1f}% · 최다작가 {full_conc['max_author_pieces']}편\n",
          flush=True)

    rows = []
    print(f"{'N':>4s} {'편수':>8s} {'작가':>6s} {'상위10%':>8s} {'Gini':>6s} "
          f"{'LoCC최대%p':>10s} {'SF%p':>7s} {'문장수Δ':>8s} {'어휘Δ':>8s} {'인과표지Δ':>10s}")
    for n in CAPS:
        sub = cap(pool, n)
        d = diffs(sub, pool)
        conc = author_concentration(sub)
        rows.append({"cap": n, "n_pieces": len(sub), **conc, "diffs": d})
        print(f"{n:4d} {len(sub):8,} {conc['n_authors']:6,} {conc['top10_share_pct']:7.1f}% "
              f"{conc['gini']:6.3f} {d['locc1']['max_pp']:9.2f}p {d['is_sf']['max_pp']:6.2f}p "
              f"{d['n_sent']['med_diff']:+8.1f} {d['ttr']['med_diff']:+8.4f} "
              f"{d['표지_인과']['med_diff']:+10.5f}", flush=True)

    print("\n[2] 작가 단위 cluster bootstrap 95% CI (LoCC 최대 비율차, %p)", flush=True)
    for n in [8, 16, 32]:
        lo, hi = boot_ci(pool, n, "locc1", lambda s, f: diffs(s, f)["locc1"]["max_pp"])
        print(f"  N={n:3d}  [{lo:.2f}, {hi:.2f}] %p", flush=True)

    (OUT / "sensitivity.json").write_text(json.dumps(
        {"pool_size": len(pool), "full": full_conc, "caps": rows,
         "note": "대표성 검정 아님 — 민감도 분석. 무상한 풀은 다작 작가 가중이 큼."},
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n-> {OUT/'sensitivity.json'}")


if __name__ == "__main__":
    main()


def fine_grid_and_knee() -> None:
    """세밀 격자 + 무릎점(knee). `python -c "from author_cap_sensitivity import *; fine_grid_and_knee()"`

    knee = 두 끝점을 잇는 직선에서 가장 멀리 떨어진 점(Kneedle). 임의 문턱이 없다.
    trade-off 축 = x: 상위10 저자 점유율(↑ 나쁨) · y: LoCC 최대 구성변화 %p(↓ 좋음).
    """
    import json
    OUT.mkdir(parents=True, exist_ok=True)
    pool = build_pool()
    (OUT / "pool.json").write_text(json.dumps(pool, default=float), encoding="utf-8")
    full = author_concentration(pool)
    rows = []
    for n in range(1, 49):
        s = cap(pool, n)
        c = author_concentration(s)
        d = diffs(s, pool)
        rows.append({"cap": n, "n_pieces": len(s), "top10": c["top10_share_pct"],
                     "gini": c["gini"], "locc_pp": d["locc1"]["max_pp"],
                     "sf_pp": d["is_sf"]["max_pp"], "med_sent": d["n_sent"]["med_diff"]})
    x = np.array([r["top10"] for r in rows])
    y = np.array([r["locc_pp"] for r in rows])
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    # 두 끝점을 잇는 직선에서의 수직 거리
    p0, p1 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    v = p1 - p0
    v = v / np.linalg.norm(v)
    dist = [abs(np.cross(v, np.array([a, b]) - p0)) for a, b in zip(xn, yn)]
    knee = rows[int(np.argmax(dist))]
    for r, dd in zip(rows, dist):
        r["knee_dist"] = float(dd)
    (OUT / "fine_grid.json").write_text(json.dumps(
        {"full_pool": {"n_pieces": len(pool), **full}, "rows": rows,
         "knee_cap": knee["cap"]}, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"전체 풀 {len(pool):,}편 · 상위10 {full['top10_share_pct']:.1f}%")
    print(f"{'N':>3s} {'편수':>8s} {'상위10%':>8s} {'LoCC%p':>8s} {'SF%p':>7s} {'무릎거리':>8s}")
    for r in rows:
        if r["cap"] <= 24 or r["cap"] % 4 == 0:
            print(f"{r['cap']:3d} {r['n_pieces']:8,} {r['top10']:7.2f}% {r['locc_pp']:7.2f}p "
                  f"{r['sf_pp']:6.2f}p {r['knee_dist']:8.4f}"
                  + ("   ← knee" if r["cap"] == knee["cap"] else ""))
    print(f"\n🔒 무릎점 N = {knee['cap']}  ({knee['n_pieces']:,}편 · 상위10 {knee['top10']:.2f}% · "
          f"LoCC {knee['locc_pp']:.2f}%p)")
