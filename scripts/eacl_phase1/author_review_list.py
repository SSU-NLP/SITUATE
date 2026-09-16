"""인물 키 정규화를 위한 **수동 검수 목록** 생성 — 재빌드 전 선행 작업.

왜: 현행 빌드는 `per_author[author 문자열 전체]` 를 상한 키로 써서, 같은 인물이 삽화가·
서문저자가 붙으면 다른 키가 되어 상한을 새로 받는다(2026-08-03 실측: 인물 1,320명 중 71명이
20편 초과, 초과분 925편 = 코퍼스의 7.15%). 이를 인물 단위로 고치려면 키 병합이 필요한데,
자동 병합은 동명이인을 잘못 합칠 수 있다(실측: `Austen, Jane, 1775-1817` 과 `Austen, Sidney`).

규칙 (보수적):
  · 기본 키 = 정규화된 이름 + 생몰연도.
  · 생몰연도가 **충돌**하면 별도 인물로 유지한다.
  · 생몰연도가 **없거나 부분만 일치**하면 자동 병합하지 않고 검수로 넘긴다.
  · 이니셜·필명·역순 표기는 명시적 alias table 이 있을 때만 병합한다.

읽기 전용이다 — 판단은 사람이 한다. 산출은 검수용 표 3개.

    python scripts/eacl_phase1/author_review_list.py <corpus.jsonl> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ANTH_PLACEHOLDER = re.compile(r"\b(various|anonymous|unknown|unidentified)\b", re.I)
ANTH_EDITOR = re.compile(r"\[(editor|compiler|contributor)\]", re.I)
ROLE = re.compile(r"\s*\[([^\]]*)\]\s*$")
# "Doyle, Arthur Conan, 1859-1930" / "Defoe, Daniel, 1661?-1731" / "Fitzgerald, M. (Artist)"
YEARS = re.compile(r",\s*(\d{3,4}\??)\s*-\s*(\d{3,4}\??)?\s*$")
BIRTH_ONLY = re.compile(r",\s*(\d{3,4}\??)\s*-?\s*$")


def is_anthology(author: str) -> bool:
    first = author.split(";")[0]
    return bool(ANTH_PLACEHOLDER.search(first) or ANTH_EDITOR.search(first))


def parse_person(seg: str) -> tuple[str, str, str]:
    """저자 1명 문자열 → (이름부, 연도부, 역할). 연도부가 ''면 미상."""
    seg = seg.strip()
    m = ROLE.search(seg)
    role = m.group(1).strip() if m else ""
    seg = ROLE.sub("", seg).strip()
    ym = YEARS.search(seg)
    if ym:
        years = f"{ym.group(1)}-{ym.group(2) or ''}"
        name = seg[: ym.start()].strip()
    else:
        bm = BIRTH_ONLY.search(seg)
        if bm:
            years = f"{bm.group(1)}-"
            name = seg[: bm.start()].strip()
        else:
            years, name = "", seg
    return name, years, role


def norm(name: str) -> str:
    """비교용 이름 정규화 — 병합 판단이 아니라 **후보 묶기**에만 쓴다."""
    n = name.lower()
    n = re.sub(r"\([^)]*\)", " ", n)      # (Lyman Frank) 같은 확장형 제거
    n = re.sub(r"[^\w\s,]", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=20)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
    cap_rows = [r for r in rows if not is_anthology(r["author"])]

    # 인물(이름+연도) → 편수 · 등장한 author 문자열 · 역할
    pieces: Counter = Counter()
    variants: dict[tuple[str, str], set[str]] = defaultdict(set)
    roles: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in cap_rows:
        name, years, role = parse_person(r["author"].split(";")[0])
        key = (name, years)
        pieces[key] += 1
        variants[key].add(r["author"])
        roles[key][role or "-"] += 1

    # ① 상한 초과 인물 — 문자열이 몇 개로 갈렸는지
    over = [(k, n) for k, n in pieces.items() if n > a.cap]
    over.sort(key=lambda x: -x[1])
    p1 = a.out / "review_01_cap_violations.md"
    with open(p1, "w", encoding="utf-8") as fh:
        fh.write(f"# ① 상한 초과 인물 ({len(over)}명 · 초과분 "
                 f"{sum(n - a.cap for _, n in over):,}편)\n\n")
        fh.write("병합 판단이 필요한 이유 = 같은 인물이 여러 author 문자열로 갈려 각각 상한을 받았다.\n\n")
        fh.write("| 편수 | 문자열 | 이름 | 생몰 | 검수 |\n|---|---|---|---|---|\n")
        for (name, years), n in over:
            flag = "연도없음 ⚠️" if not years else ""
            fh.write(f"| {n} | {len(variants[(name, years)])} | {name} | {years or '—'} | {flag} |\n")
        fh.write("\n## 문자열 내역\n\n")
        for (name, years), n in over:
            fh.write(f"### {name} ({years or '연도없음'}) — {n}편\n")
            for v in sorted(variants[(name, years)]):
                fh.write(f"- `{v}`\n")
            fh.write("\n")

    # ② 같은 이름인데 연도가 다르거나 없는 묶음 = 동명이인 위험 → 자동병합 금지
    by_name: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for name, years in pieces:
        by_name[norm(name)].add((name, years))
    amb = {k: v for k, v in by_name.items() if len(v) > 1}
    p2 = a.out / "review_02_same_name_diff_years.md"
    with open(p2, "w", encoding="utf-8") as fh:
        fh.write(f"# ② 이름은 같은데 생몰연도가 다르거나 없는 묶음 ({len(amb)}건)\n\n")
        fh.write("**자동 병합 금지.** 동명이인일 수 있다(실측 사례: Austen, Jane vs Austen, Sidney).\n\n")
        fh.write("| 정규화 이름 | 후보 | 편수 |\n|---|---|---|\n")
        for k, vs in sorted(amb.items(), key=lambda x: -sum(pieces[v] for v in x[1])):
            cand = " / ".join(f"{nm} [{yr or '연도없음'}] {pieces[(nm, yr)]}편"
                              for nm, yr in sorted(vs, key=lambda v: -pieces[v]))
            fh.write(f"| {k} | {cand} | {sum(pieces[v] for v in vs)} |\n")

    # ③ 연도 미상 인물 — 병합 근거가 없어 보수적으로 분리 유지
    noyear = [(k, n) for k, n in pieces.items() if not k[1]]
    noyear.sort(key=lambda x: -x[1])
    p3 = a.out / "review_03_missing_years.md"
    with open(p3, "w", encoding="utf-8") as fh:
        fh.write(f"# ③ 생몰연도 미상 인물 ({len(noyear)}명 · {sum(n for _, n in noyear):,}편)\n\n")
        fh.write("연도가 없으면 동일인 판정 근거가 없다. **기본은 분리 유지**, 필요 시 alias table 로만 병합.\n\n")
        fh.write("| 편수 | 이름 | 주 역할 |\n|---|---|---|\n")
        for (name, _), n in noyear[:200]:
            fh.write(f"| {n} | {name} | {roles[(name, '')].most_common(1)[0][0]} |\n")
        if len(noyear) > 200:
            fh.write(f"\n… 외 {len(noyear) - 200}명\n")

    print(f"상한 대상 {len(cap_rows):,}편 · 인물(이름+연도) {len(pieces):,}명")
    print(f"  ① 상한 초과      {len(over)}명 → {p1}")
    print(f"  ② 동명이인 위험  {len(amb)}건 → {p2}")
    print(f"  ③ 연도 미상      {len(noyear)}명 → {p3}")


if __name__ == "__main__":
    main()
