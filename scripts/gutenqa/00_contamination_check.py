#!/usr/bin/env python3
"""Gate 0 — GutenQA 100편이 Phase-1 앵커 코퍼스(M5 학습 데이터)와 겹치는가?

## 왜 필요한가
D-B15 Phase-1 앵커 = PG 영어 단편 10,598편. M5 는 이걸로 학습했다.
GutenQA 100편도 Project Gutenberg. 겹치면 **학습 데이터로 평가**하는 꼴이 된다.

## 겹칠 확률이 낮다고 보는 구조적 근거 (`scripts/eacl_phase1/build_pg_shorts.py`)
① Subjects 에 `"short stories"` 없으면 후보에서 탈락 → 장편 대부분 배제
② `split_volume()` 이 챕터 헤딩 다수인 권을 `chaptered_work` 로 통째 폐기 → 장편 배제
③ 길이 상한 900 문장 → 권당 ~12만 토큰인 GutenQA 도서는 통과 불가
④ `EXCL_AUTH` 14명 (Twain, Poe, Doyle, London, Melville, Maupassant, Chekhov, O.Henry …) 사전 제외

🔴 그래도 **단편집**은 위험하다 — Dubliners / Grimms' Fairy Tales / The King in Yellow /
White Nights and Other Stories 는 Subjects 에 "short stories" 가 붙어 있을 수 있고 저자가
EXCL_AUTH 에 없다. 이 스크립트는 그걸 잡으라고 있다.

## 방법 — 2중 방어
GutenQA `Book Name` 형식 = `Title_-_Author` (밑줄 구분). 제목·저자 둘 다 파싱된다.

- **Prong A (직접):** 앵커 메타(`all_master.meta.jsonl`)의 `title` + `author` 에 직접 대조.
  앵커의 `title` 은 *권* 제목이므로 단편집이 들어갔다면 여기서 바로 잡힌다.
- **Prong B (경유):** GutenQA 제목 → `pgcat.csv`(90,353행)로 PG ID 역매핑 → 앵커의
  `gutenberg_id` 고유값 집합과 교집합. 앵커 제목 문자열이 달라도 잡는다.

**자동 fuzzy 채택 금지.** 완전일치 실패분은 후보 top-5 를 덤프해 사람이 판정한다.

    uv run python scripts/gutenqa/00_contamination_check.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

PGCAT = REPO_ROOT / "data/eacl_phase1/pgcat.csv"
ANCHOR_META = REPO_ROOT / "data/eacl_phase1/all_master.meta.jsonl"
OUTDIR = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"
QUESTIONS = OUTDIR / "data/questions.parquet"
QUESTIONS_URL = "https://huggingface.co/datasets/LumberChunker/GutenQA/resolve/main/questions.parquet"

# build_pg_shorts.py ⑥ 격리 목록 — 리포트에 "이미 배제됨"을 표시하기 위해 그대로 복사
EXCL_AUTH = ["Henry, O.", "Maupassant", "Chekhov", "Chekhof", "Crane, Stephen", "Poe, Edgar",
             "London, Jack", "Bierce, Ambrose", "Gilman, Charlotte", "Doyle, Arthur Conan",
             "Twain", "Clemens", "Jewett, Sarah", "Melville"]

# 제목 자동매칭이 실패한 5편의 **사람 판정** 결과 (2026-07-29).
# 자동 fuzzy 채택을 금지했으므로 후보 top-5 를 보고 손으로 확정한 뒤 여기에 못박는다.
# 근거 = 각 후보의 Jaccard 최상위 + 저자 완전일치. 재실행해도 같은 결론이 나오도록 코드로 남긴다.
MANUAL_TITLE_MAP: dict[str, list[int]] = {
    # GutenQA Book Name                                  : 확정된 PG ID(들)
    "Incidents_in_the_Life_of_a_Slave_Girl_-_Harriet_A._Jacobs": [11030],
    "Les_Miserables_-_Victor_Hugo": [135],                  # 악센트 'Misérables'
    "Louise_de_la_Valliere_-_Alexandre_Dumas": [2710],      # 악센트 'Vallière'
    "The_Adventures_of_Tom_Sawyer_-_Mark_Twain": [74, 7193, 7194, 7195, 7196],  # Complete + 4분책
    "The_Extraordinary_Adventures_of_Arsene_Lupin_-_Maurice_Leblanc": [6133],   # 부제 'Gentleman-Burglar'
}

_ARTICLES = {"the", "a", "an"}
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def norm_title(s: str) -> str:
    """소문자·구두점 제거·공백 정규화·선행 관사 제거. 유니코드 아포스트로피(’) 포함."""
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("_", " ").lower()
    s = _PUNCT.sub(" ", s)
    toks = _WS.sub(" ", s).strip().split()
    while toks and toks[0] in _ARTICLES:
        toks.pop(0)
    return " ".join(toks)


def norm_author(s: str) -> frozenset[str]:
    """저자를 토큰 집합으로. pgcat/앵커는 'Twain, Mark', GutenQA 는 'Mark Twain' 이라
    순서·구분자에 의존할 수 없다. 연도(1835-1910)와 1글자 이니셜은 버린다."""
    s = unicodedata.normalize("NFKD", str(s or "")).replace("_", " ").lower()
    s = _PUNCT.sub(" ", s)
    toks = [t for t in _WS.sub(" ", s).strip().split()
            if len(t) > 1 and not t.isdigit()]
    return frozenset(toks)


def title_variants(t: str) -> set[str]:
    """PG 제목은 부제를 달고 다닌다. 전체/앞부분 둘 다 후보로 낸다.
    'Moby Dick; Or, The Whale' -> {'moby dick or the whale', 'moby dick'}"""
    out = {norm_title(t)}
    raw = str(t or "").replace("_", " ")
    for sep in [";", ":", "\r\n", "\n", " — ", " -- "]:
        if sep in raw:
            out.add(norm_title(raw.split(sep)[0]))
    # 'X Or The Y' / 'X, and Y' 형태
    m = re.split(r"\bor,?\b|\band on the\b", norm_title(t))
    if len(m) > 1 and m[0].strip():
        out.add(m[0].strip())
    return {v for v in out if v}


def load_gutenqa_books() -> list[dict]:
    import pandas as pd
    if not QUESTIONS.exists():
        import urllib.request
        QUESTIONS.parent.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] {QUESTIONS_URL}")
        urllib.request.urlretrieve(QUESTIONS_URL, QUESTIONS)
    df = pd.read_parquet(QUESTIONS)
    books = []
    for bid, name in (df[["Book ID", "Book Name"]].drop_duplicates()
                      .sort_values("Book ID").itertuples(index=False)):
        # 'Title_-_Author' — 마지막 '_-_' 로 쪼갠다 (제목 안에 '-' 가 있을 수 있음)
        if "_-_" in name:
            title_raw, author_raw = name.rsplit("_-_", 1)
        else:
            title_raw, author_raw = name, ""
        books.append({
            "book_id": int(bid), "book_name": name,
            "title": title_raw.replace("_", " "),
            "author": author_raw.replace("_", " "),
        })
    return books


def load_anchor() -> tuple[list[dict], set[int]]:
    rows, gids = [], set()
    with ANCHOR_META.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gids.add(int(r["gutenberg_id"]))
            rows.append(r)
    return rows, gids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTDIR / "contamination.md"))
    a = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    books = load_gutenqa_books()
    anchor_rows, anchor_gids = load_anchor()
    print(f"[load] GutenQA {len(books)} books | anchor {len(anchor_rows)} pieces, "
          f"{len(anchor_gids)} unique PG ids")

    # ---- Prong A: 앵커 메타의 (title, author) 에 직접 대조 -------------------
    # 앵커는 권 하나가 여러 piece 로 쪼개져 있으므로 gutenberg_id 로 접는다.
    anchor_by_gid: dict[int, dict] = {}
    for r in anchor_rows:
        g = int(r["gutenberg_id"])
        d = anchor_by_gid.setdefault(g, {"gid": g, "title": r.get("title", ""),
                                         "author": r.get("author", ""), "n_pieces": 0})
        d["n_pieces"] += 1
    anchor_title_idx: dict[str, list[dict]] = defaultdict(list)
    for d in anchor_by_gid.values():
        for v in title_variants(d["title"]):
            anchor_title_idx[v].append(d)

    # ---- Prong B: pgcat.csv 로 제목 -> PG ID 역매핑 --------------------------
    pgcat_idx: dict[str, list[dict]] = defaultdict(list)
    pgcat_rows = []
    with PGCAT.open(encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r.get("Type") != "Text":
                continue
            try:
                gid = int(r["Text#"])
            except (ValueError, KeyError, TypeError):
                continue
            rec = {"gid": gid, "title": r.get("Title", ""), "authors": r.get("Authors", ""),
                   "subjects": r.get("Subjects", ""), "lang": r.get("Language", "")}
            pgcat_rows.append(rec)
            for v in title_variants(rec["title"]):
                pgcat_idx[v].append(rec)
    print(f"[load] pgcat {len(pgcat_rows)} Text rows, {len(pgcat_idx)} title keys")

    results = []
    for b in books:
        gq_auth = norm_author(b["author"])
        gq_titles = title_variants(b["title"])

        # Prong A
        a_hits = []
        for v in gq_titles:
            for d in anchor_title_idx.get(v, []):
                if not gq_auth or not norm_author(d["author"]) or (gq_auth & norm_author(d["author"])):
                    a_hits.append(d)
        a_hits = {d["gid"]: d for d in a_hits}

        # Prong B
        cand = []
        for v in gq_titles:
            cand.extend(pgcat_idx.get(v, []))
        # 저자 토큰이 겹치는 것만 채택 (동명이제 방지)
        cand_auth = [c for c in cand if not gq_auth or (gq_auth & norm_author(c["authors"]))]
        pg_ids = sorted({c["gid"] for c in cand_auth})

        # 매핑 신뢰도 — 자동매칭 결과를 먼저 확정하고, 실패분만 사람 판정으로 덮는다
        if cand_auth:
            conf = "exact(title+author)"
        elif cand:
            conf = "title-only(저자 불일치)"
        elif b["book_name"] in MANUAL_TITLE_MAP:
            pg_ids = sorted(MANUAL_TITLE_MAP[b["book_name"]])
            conf = "manual(사람 판정)"
        else:
            conf = "UNMATCHED"

        b_hits = sorted(set(pg_ids) & anchor_gids)

        # 이미 EXCL_AUTH 로 배제된 저자인가
        excl = [e for e in EXCL_AUTH if norm_author(e) & gq_auth]

        results.append({
            **b, "pg_ids": pg_ids, "n_pg_candidates": len(cand),
            "confidence": conf,
            "prongA_anchor_gids": sorted(a_hits),
            "prongB_anchor_gids": b_hits,
            "contaminated": bool(a_hits or b_hits),
            "excl_auth": excl,
            "subjects": (cand_auth[0]["subjects"] if cand_auth else
                         (cand[0]["subjects"] if cand else "")),
        })

    contaminated = [r for r in results if r["contaminated"]]
    unmatched = [r for r in results if r["confidence"] == "UNMATCHED"]
    title_only = [r for r in results if r["confidence"] == "title-only(저자 불일치)"]
    manual = [r for r in results if r["confidence"] == "manual(사람 판정)"]

    # 오염된 책이 앵커에서 정확히 몇 조각으로 학습됐는지 (영향 규모)
    for r in contaminated:
        hit = set(r["prongA_anchor_gids"]) | set(r["prongB_anchor_gids"])
        pcs = [x for x in anchor_rows if int(x["gutenberg_id"]) in hit]
        r["anchor_pieces"] = len(pcs)
        r["anchor_sents"] = sum(int(x.get("n_sent", 0)) for x in pcs)
        r["anchor_piece_titles"] = [x.get("piece_title", "") for x in pcs]

    # UNMATCHED 는 후보 top-5 를 사람이 보게 덤프 (자동 fuzzy 채택 금지)
    suggestions = {}
    for r in unmatched:
        key_toks = set(norm_title(r["title"]).split())
        scored = []
        for c in pgcat_rows:
            if not (gq := norm_author(r["author"])) or not (gq & norm_author(c["authors"])):
                continue
            ct = set(norm_title(c["title"]).split())
            if not ct:
                continue
            j = len(key_toks & ct) / len(key_toks | ct)
            scored.append((j, c))
        scored.sort(key=lambda x: -x[0])
        suggestions[r["book_name"]] = [
            {"jaccard": round(j, 3), "gid": c["gid"], "title": c["title"][:90],
             "authors": c["authors"][:60], "in_anchor": c["gid"] in anchor_gids}
            for j, c in scored[:5]]

    json_out = OUTDIR / "contamination.json"
    json_out.write_text(json.dumps(
        {"n_books": len(results), "n_contaminated": len(contaminated),
         "n_unmatched": len(unmatched), "anchor_unique_pg_ids": len(anchor_gids),
         "books": results, "unmatched_suggestions": suggestions},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 리포트 -----------------------------------------------------------
    L = []
    L.append("# Gate 0 — GutenQA × Phase-1 앵커 오염 체크\n")
    L.append(f"- GutenQA 도서: **{len(results)}편**")
    L.append(f"- 앵커(D-B15) 고유 PG ID: **{len(anchor_gids)}개** / {len(anchor_rows)} pieces")
    L.append(f"- 🔴 **오염(교집합): {len(contaminated)}편**")
    L.append(f"- 자동매칭 실패 → 사람 판정으로 확정: {len(manual)}편 · "
             f"끝내 미매칭(UNMATCHED): {len(unmatched)}편 · "
             f"저자 불일치(title-only): {len(title_only)}편\n")

    if contaminated:
        L.append("## 🔴 오염된 책 (평가에서 제외 대상)\n")
        L.append("| Book ID | Book Name | PG ID | 앵커 조각 수 | 앵커 문장 수 |")
        L.append("|---|---|---|---|---|")
        for r in contaminated:
            L.append(f"| {r['book_id']} | {r['book_name']} | {r['pg_ids']} | "
                     f"{r['anchor_pieces']} | {r['anchor_sents']:,} |")
        L.append("")
        L.append("두 권 모두 **단편집**이다 — 앵커 빌더의 `\"short stories\" in Subjects` 필터를 "
                 "정상 통과했고, 저자(Joyce / Chambers)가 `EXCL_AUTH` 14명에 없다. "
                 "예측대로 장편은 한 편도 새지 않았고, 샌 것은 단편집뿐이다.\n")
        for r in contaminated:
            L.append(f"- **{r['book_name']}** (PG {r['pg_ids']}) 앵커 수록 조각: "
                     + ", ".join(f"`{t}`" for t in r["anchor_piece_titles"]))
        L.append("")
    else:
        L.append("## ✅ 오염 0편 — 두 prong 모두 교집합 없음\n")

    if manual:
        L.append("## 사람 판정으로 확정한 매핑 (자동 fuzzy 채택 금지 규칙에 따름)\n")
        L.append("악센트(Misérables/Vallière)·부제·분책 때문에 문자열 자동매칭이 실패한 건들이다. "
                 "`MANUAL_TITLE_MAP` 에 코드로 못박아 재실행 시에도 동일 결론이 나온다.\n")
        L.append("| Book ID | Book Name | 확정 PG ID | 앵커? |")
        L.append("|---|---|---|---|")
        for r in manual:
            L.append(f"| {r['book_id']} | {r['book_name']} | {r['pg_ids']} | "
                     f"{'🔴 YES' if r['contaminated'] else '✅ no'} |")
        L.append("")

    if unmatched:
        L.append("## ⚠️ 제목→PG ID 매핑 실패 (사람 판정 필요, 자동 채택 안 함)\n")
        for r in unmatched:
            L.append(f"**{r['book_id']} · {r['book_name']}**")
            sug = suggestions.get(r["book_name"], [])
            if not sug:
                L.append("- 저자 토큰이 겹치는 pgcat 항목이 하나도 없음")
            for s in sug:
                flag = " 🔴앵커포함" if s["in_anchor"] else ""
                L.append(f"- J={s['jaccard']} · PG {s['gid']} · {s['title']} — {s['authors']}{flag}")
            L.append("")

    L.append("## 전체 100편\n")
    L.append("| ID | Book Name | PG ID | 앵커? | 신뢰도 | EXCL_AUTH 사전배제 |")
    L.append("|---|---|---|---|---|---|")
    for r in results:
        mark = "🔴 YES" if r["contaminated"] else "✅ no"
        ids = ",".join(map(str, r["pg_ids"][:3])) + ("…" if len(r["pg_ids"]) > 3 else "")
        L.append(f"| {r['book_id']} | {r['book_name'][:60]} | {ids or '—'} | {mark} | "
                 f"{r['confidence']} | {', '.join(r['excl_auth']) or '—'} |")

    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"오염: {len(contaminated)}편 / {len(results)}")
    print(f"UNMATCHED: {len(unmatched)}편 · title-only: {len(title_only)}편")
    print(f"saved -> {a.out}")
    print(f"saved -> {json_out}")
    if contaminated:
        for r in contaminated:
            print(f"  🔴 {r['book_id']} {r['book_name']} -> anchor gids "
                  f"A{r['prongA_anchor_gids']} B{r['prongB_anchor_gids']}")


if __name__ == "__main__":
    main()
