#!/usr/bin/env python3
"""최종 풀 기준 작가당 상한 스윕 — figure_01 과 다양성 표의 원자료를 만든다.

최종 조건 (2026-08-02 확정)
  격리        dev-3 저자(Frederik Pohl)만
  다운로드    권별 누적 다운로드 상위 5% 제외 — 컷은 **빌드가 쓴 값**을 매니페스트에서 읽어 재사용한다
  선집        상한 면제 (`build_pg_shorts.is_anthology`) · 유효 저자 수 계산에서 제외 · 코퍼스에는 포함

⚠️ 상한을 소프트웨어로 훑으려면 **상한 없는 풀**이 필요하다. 상한이 걸린 코퍼스에서는 N 을 올릴 수 없다.
   그래서 입력은 `pg_shorts_dev3_nocap_nofilter.jsonl` 이고, 다운로드 컷은 여기서 다시 적용한다.

⚠️ 시대 십분위 경계는 **컷 적용 후 풀**에서 정의한다. 컷 전 풀에서 잡으면 표에 싣는 코퍼스와
   경계를 정의한 모집단이 달라진다.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eacl_phase1"))
from build_pg_shorts import is_anthology  # noqa: E402  — 빌드와 같은 판별을 쓴다
from author_review_list import parse_person  # noqa: E402  — 인물 키(이름+생몰연도) 파싱

POOL = REPO / "data/eacl_phase1/pg_shorts_dev3_nocap_nofilter.jsonl"
FINAL_MF = REPO / "data/eacl_phase1/pg_shorts_dev3_dl95_cap19_anthfree_manifest.json"
CAT = REPO / "data/eacl_phase1/pgcat.csv"
DOWNLOADS = REPO / "data/eacl_phase1/pg_downloads.json"
OUTS = {  # 조건별 네임스페이스 — 같은 파일에 덮어쓰면 어느 키로 만든 표인지 사라진다
    "string": REPO / "outputs/experiments/2026-08-02_author_cap/table_final_dev3.json",
    "person": REPO / "outputs/experiments/2026-08-02_author_cap/table_final_dev3_personkey.json",
}

ALPHA = re.compile(r"^([A-Z]+)")
BIRTH = re.compile(r",\s*(\d{4})\s*-")


def lc_class(locc: str | None) -> str:
    """LCC 강목 — 알파벳 접두만 취하고, P 류(문학)가 있으면 그것을 대표로 삼는다."""
    cs = [ALPHA.match(x.strip()).group(1) for x in (locc or "").split(";") if ALPHA.match(x.strip())]
    return ([c for c in cs if c.startswith("P")] or cs or ["NA"])[0]


def inv_simpson(c: Counter) -> float:
    q = np.array(list(c.values()), float)
    q /= q.sum()
    return float(1 / np.sum(q ** 2))


def author_key(author: str, mode: str) -> object:
    """상한을 무엇 단위로 걸 것인가.

    string = 서지 레코드의 저자 문자열 그대로(2026-08-02 최초 스윕). 같은 인물이라도 삽화가·서문
             저자가 함께 표기되면 다른 키가 되어 **한 인물이 상한을 여러 번 받는다**
             (2026-08-03 실측: 인물 1,320명 중 71명이 20편 초과, 초과분 925편).
    person = 첫 저자의 (정규화 이름, 생몰연도). 연도가 없거나 충돌하면 병합하지 않는다.
    """
    if mode == "string":
        return author
    name, years, _role = parse_person(author.split(";")[0])
    return (name, years)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--author-key", choices=("string", "person"), default="person",
                    help="상한 적용 단위. person 이 기본(문자열 키는 인물 편중을 통제하지 못한다)")
    args = ap.parse_args()
    KEY = args.author_key
    OUT = OUTS[KEY]
    cut = json.loads(FINAL_MF.read_text())["download_cut"]
    dl = json.loads(DOWNLOADS.read_text())
    cat = {r["Text#"]: r for r in csv.DictReader(CAT.open(encoding="utf-8"))}

    rows = []
    for line in POOL.open(encoding="utf-8"):
        r = json.loads(line)
        g = str(r["gutenberg_id"])
        if dl.get(g, 0) >= cut:                     # 다운로드 상위 5% 권 제외
            continue
        m = BIRTH.search(r["author"])
        rows.append({"author": r["author"], "gid": r["gutenberg_id"], "n_sent": r["n_sent"],
                     "lc": lc_class(cat.get(g, {}).get("LoCC")),
                     "by": int(m.group(1)) if m else None,
                     "anth": is_anthology(r["author"])})

    anth = [r for r in rows if r["anth"]]
    named = [r for r in rows if not r["anth"]]
    edges = np.quantile([r["by"] for r in rows if r["by"]], np.linspace(0, 1, 11))[1:-1]
    for r in rows:
        r["key"] = author_key(r["author"], KEY)
    max_n = Counter(r["key"] for r in named).most_common(1)[0][1]

    print(f"풀 {len(rows):,}편 (권 {len({r['gid'] for r in rows}):,}) · 다운로드 컷 {cut:,.0f}회")
    print(f"  선집 {len(anth):,}편(상한 면제) · 저자 특정가능 {len(named):,}편 / "
          f"{len({r['key'] for r in named}):,}명 · 최다 {max_n}편  [키={KEY}]")
    print(f"  출생연도 있음 {sum(1 for r in rows if r['by']):,} · 십분위 {edges.astype(int)}")

    auth, pieces, sent, era, lc, top100 = [], [], [], [], [], []
    for n in range(1, max_n + 1):
        c, keep = Counter(), []
        for r in named:
            if c[r["key"]] < n:
                c[r["key"]] += 1
                keep.append(r)
        corpus = keep + anth
        b = [r["by"] for r in corpus if r["by"]]
        auth.append(inv_simpson(c))
        pieces.append(len(corpus))
        sent.append(sum(r["n_sent"] for r in corpus))
        era.append(inv_simpson(Counter(np.digitize(b, edges).tolist())))
        lc.append(inv_simpson(Counter(r["lc"] for r in corpus)))
        top100.append(sum(v for _, v in c.most_common(100)) / len(keep) * 100)

    a = np.array(auth)
    p = np.array(pieces, float)
    rate = (np.diff(p) / -np.diff(a)).tolist()
    sm = np.convolve(rate, np.ones(3) / 3, "valid")
    nn = np.arange(2, len(rate))
    peak = int(nn[int(np.argmax(sm))])
    band = nn[sm >= sm.max() * 0.95]

    OUT.write_text(json.dumps({"author_key": KEY, "download_cut": cut, "max_n": max_n, "auth": auth,
                               "pieces": pieces, "sent": sent, "era": era, "lc": lc,
                               "top100": top100, "rate": rate, "peak": peak,
                               "band": [int(band.min()), int(band.max())]}))
    print(f"\n한계 교환비 정점 N={peak} ({sm.max():.2f}) · 최대의 95% 구간 N={band.min()}~{band.max()}")
    print(f"LC 전 구간 {min(lc):.2f}~{max(lc):.2f} (폭 {max(lc) - min(lc):.2f})")
    print(f"\n{'N':>6s} {'유효저자':>8s} {'편수':>8s} {'문장':>10s} {'편/명':>7s} "
          f"{'유효시대':>8s} {'유효LC':>7s} {'상위100':>8s}")
    for n in (1, 5, 9, 12, 16, peak, 24, 48):
        i = n - 1
        print(f"{n:6d} {auth[i]:8.0f} {pieces[i]:8,d} {sent[i]:10,d} {rate[i]:7.2f} "
              f"{era[i]:8.2f} {lc[i]:7.2f} {top100[i]:7.1f}%")
    print(f"{'없음':>6s} {auth[-1]:8.0f} {pieces[-1]:8,d} {sent[-1]:10,d} {'–':>7s} "
          f"{era[-1]:8.2f} {lc[-1]:7.2f} {top100[-1]:7.1f}%")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
