#!/usr/bin/env python3
"""Gate 0-b — GutenQA 100편을 평가 부분집합으로 분류(**점수 보기 전 동결**).

## 왜
M5 의 scope 는 **영어 서사 픽션**(D-M8)이다. GutenQA 100편에는 철학·정치학 논고
(Meditations / The Republic / The Prince / Beyond Good and Evil …)와 자서전
(Frederick Douglass / Wagner …)이 섞여 있어 도메인 밖이다.
사용자 결정 = **전체 100편 주표 + 서사 부분집합 병기**. 그러려면 분류가 필요하고,
분류는 `no-postresult-criterion-change` 에 따라 **점수를 보기 전에** 확정돼야 한다.

## 기준 — 내 주관이 아니라 PG 메타데이터
1차 = `pgcat.csv` 의 `Subjects` 에 **픽션 표지**(`fiction` / `stories` / `tales`)가 있는가.
   이 한 줄이 100편을 85 / 15 로 정확히 가른다 (경계 사례 없음 — 각 편의 Subjects 를
   눈으로 확인함).
2차 = 픽션 표지가 없는 15편만 `nonfiction_narrative`(자서전·회고록 = 서사이되 픽션 아님)와
   `non_narrative`(철학·에세이·운문)로 손으로 나눈다. 사유를 각 항목에 적어 동결한다.

## 산출 부분집합
| 이름 | 정의 | 용도 |
|---|---|---|
| `all100` | 전체 100편 | **주표** |
| `narrative_fiction` | 픽션 표지 있는 85편 | **M5 scope 병기표** |
| `nonfiction_narrative` | 자서전·회고록 4편 | 참고 |
| `non_narrative` | 철학·에세이·운문 11편 | 참고(도메인 밖) |
| `contaminated` | Dubliners, The King in Yellow (2편) | 오염 효과 직접 측정 |
| `all100_minus_contaminated` | 98편 | 오염 민감도 |
| `dev` / `test` | Book ID 0–19 / 20–99 | 봉인 (D-B37) |

    uv run python scripts/gutenqa/00b_book_subsets.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PGCAT = REPO_ROOT / "data/eacl_phase1/pgcat.csv"
OUTDIR = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"
CONTAM = OUTDIR / "contamination.json"

# 1차 자동 기준
FICTION_MARKER = re.compile(r"\bfiction\b|\bstories\b|\btales\b", re.IGNORECASE)

# 2차 — 픽션 표지가 없는 15편의 **사람 판정**(사유 포함). 점수 보기 전 동결.
NONFICTION_NARRATIVE = {          # 서사이되 픽션 아님 (자서전·회고록). M5 scope 경계.
    27: "노예 서사 자서전 (Jacobs, 1인칭 회고)",
    36: "작곡가 자서전 (Wagner)",
    37: "노예 서사 자서전 (Douglass, 1인칭 회고)",
    55: "영적 자서전 (Augustine — 후반부는 신학 논고)",
}
NON_NARRATIVE = {                 # 도메인 밖 — 철학·에세이·운문
    13: "철학 논고 (Nietzsche, 아포리즘)",
    33: "스토아 철학 잠언 (Marcus Aurelius)",
    43: "정치철학 논고 (Locke)",
    67: "운문 서사시 — 서사이나 산문 픽션 아님 (Homer)",
    69: "정치철학 논고 (Machiavelli)",
    71: "철학 개론 (Russell)",
    72: "산문시 (Gibran)",
    74: "철학 대화편 (Plato)",
    78: "사회비평 에세이집 (Du Bois)",
    88: "철학적 산문 — 서사 액자가 있으나 논고 (Nietzsche)",
    94: "에세이·자연기 (Thoreau)",
}

# ---- dev/test 봉인 (D-B37) --------------------------------------------------
# 🔴 Book ID 는 **제목 알파벳 순**이다. 단순 ID 구간(0–19)으로 자르면 dev 가 A~D 로 시작하는
# 책만 모이고, 비서사 15편 중 14편이 test 로 몰린다(dev 95% 서사 vs test 89%).
# dev 에서 크기 매칭 r* 를 정할 것이므로 dev 는 대표성이 있어야 한다.
# → **class 층화 + 시드 고정** 추출. 시드는 레포 관례값(20260711).
DEV_N = 20
SPLIT_SEED = 20260711


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTDIR / "book_subsets.json"))
    a = ap.parse_args()

    books = json.loads(CONTAM.read_text(encoding="utf-8"))["books"]
    cat = {}
    with PGCAT.open(encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                cat[int(r["Text#"])] = r
            except (ValueError, KeyError, TypeError):
                continue

    rows = []
    for b in books:
        bid = b["book_id"]
        rec = cat.get(b["pg_ids"][0]) if b["pg_ids"] else None
        subjects = (rec or {}).get("Subjects", "")
        locc = (rec or {}).get("LoCC", "")
        has_marker = bool(FICTION_MARKER.search(subjects))

        if has_marker:
            cls, why = "narrative_fiction", "Subjects 픽션 표지"
        elif bid in NONFICTION_NARRATIVE:
            cls, why = "nonfiction_narrative", NONFICTION_NARRATIVE[bid]
        elif bid in NON_NARRATIVE:
            cls, why = "non_narrative", NON_NARRATIVE[bid]
        else:  # 자동/수동이 어긋남 = 분류표가 낡았다는 뜻. 조용히 넘어가면 안 된다.
            raise SystemExit(
                f"🔴 미분류: book_id={bid} {b['book_name']!r}\n"
                f"   Subjects={subjects[:120]!r}\n"
                f"   → NONFICTION_NARRATIVE / NON_NARRATIVE 에 사유와 함께 추가할 것")

        rows.append({"book_id": bid, "book_name": b["book_name"], "pg_ids": b["pg_ids"],
                     "locc": locc, "subjects": subjects, "class": cls, "reason": why,
                     "contaminated": b["contaminated"], "split": None})

    # 수동 표에 있는데 실제로는 픽션 표지가 붙은 경우(= 표가 낡음)도 잡는다
    for bid in set(NONFICTION_NARRATIVE) | set(NON_NARRATIVE):
        r = next((x for x in rows if x["book_id"] == bid), None)
        if r and r["class"] == "narrative_fiction":
            raise SystemExit(f"🔴 수동표 충돌: book_id={bid} 가 Subjects 픽션 표지를 가짐 — 표를 갱신할 것")

    # ---- class 층화 dev/test 분할 (시드 고정) --------------------------------
    import random
    rng = random.Random(SPLIT_SEED)
    dev_ids: set[int] = set()
    for cls in sorted({r["class"] for r in rows}):
        pool = sorted(r["book_id"] for r in rows if r["class"] == cls)
        # 각 class 에서 비율만큼 (반올림), 최소 1편은 dev 로
        k = max(1, round(len(pool) * DEV_N / len(rows)))
        dev_ids.update(rng.sample(pool, k))
    # 반올림 오차 보정 — 모자라면/넘치면 전체 풀에서 결정적으로 조정
    spare = [r["book_id"] for r in sorted(rows, key=lambda x: x["book_id"])
             if r["book_id"] not in dev_ids]
    while len(dev_ids) < DEV_N:
        dev_ids.add(spare.pop(rng.randrange(len(spare))))
    while len(dev_ids) > DEV_N:
        dev_ids.discard(sorted(dev_ids)[rng.randrange(len(dev_ids))])
    for r in rows:
        r["split"] = "dev" if r["book_id"] in dev_ids else "test"

    by = lambda c: sorted(r["book_id"] for r in rows if r["class"] == c)  # noqa: E731
    subsets = {
        "all100": sorted(r["book_id"] for r in rows),
        "narrative_fiction": by("narrative_fiction"),
        "nonfiction_narrative": by("nonfiction_narrative"),
        "non_narrative": by("non_narrative"),
        "contaminated": sorted(r["book_id"] for r in rows if r["contaminated"]),
        "all100_minus_contaminated": sorted(r["book_id"] for r in rows if not r["contaminated"]),
        "dev": sorted(r["book_id"] for r in rows if r["split"] == "dev"),
        "test": sorted(r["book_id"] for r in rows if r["split"] == "test"),
    }
    payload = {
        "frozen_at": "2026-07-29 (D-B37 사전등록, 점수 보기 전)",
        "criterion": "1차 = pgcat Subjects 의 fiction/stories/tales 표지; 2차 = 표지 없는 15편만 수동 분류(사유 동결)",
        "split_rule": f"class 층화 무작위, seed={SPLIT_SEED}, dev {DEV_N}편 "
                      f"(Book ID 는 제목 알파벳순이라 ID 구간 분할은 편향됨)",
        "counts": {k: len(v) for k, v in subsets.items()},
        "subsets": subsets,
        "books": rows,
    }
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("== 부분집합 (동결) ==")
    for k, v in payload["counts"].items():
        print(f"  {k:28s} {v:>4d}")
    print("\n== 서사 픽션이 아닌 15편 ==")
    for r in rows:
        if r["class"] != "narrative_fiction":
            print(f"  [{r['split']:4s}] {r['book_id']:>3d} {r['class']:22s} "
                  f"{r['book_name'][:48]:48s} — {r['reason']}")
    print("\n== 오염 2편이 어느 split 인가 ==")
    for r in rows:
        if r["contaminated"]:
            print(f"  {r['book_id']:>3d} {r['book_name'][:50]:50s} split={r['split']} class={r['class']}")
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
