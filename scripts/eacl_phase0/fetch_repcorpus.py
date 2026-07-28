#!/usr/bin/env python3
"""①B 대표 코퍼스 획득·전처리 (사전등록 D-B11).

유명 6 = 연구자 지정(id/heading 하드코딩, §1.1). 무명 6 = PG RDF 덤프의 download_count로
기계 선정(Gutendex 403 우회 — 같은 지표, §1.2). register 층(1인칭 2 / 3인칭 4) + 밀도·길이 필터.
**작가 hand-pick 없음** — 무명은 (download asc, sha256) 정렬로 조건 충족분 자동 선택.

산출:
- `outputs/eacl_phase0/repcorpus_master.jsonl` — 12편 transition rows(doc_id·transition·previous·current·boundary=0).
- `outputs/experiments/2026-07-22_gate1b_repcorpus/manifest.json` — 확정 12편(작가·id·밀도·register·집단).
- 동 dir `selection_log.json` — 무명 후보 검토 전량(선택/스킵 사유) = 재현·감사용.

전제: /tmp/pgcat.csv (pg_catalog), /tmp/pg_downloads.json ({id:downloads}). 없으면 먼저 받아 만들 것.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from build_narrative_pack import split_sentences  # frozen sentence splitter  # noqa: E402

OUT_MASTER = REPO / "outputs/eacl_phase0/repcorpus_master.jsonl"
EXP = REPO / "outputs/experiments/2026-07-22_gate1b_repcorpus"
CACHE = Path("/tmp/repcorpus_txt")
CACHE.mkdir(exist_ok=True)

MIN_SENT, MAX_SENT = 30, 700          # §1.2 길이 필터 (collection/too-long 배제)
CAP = 250                              # 추출 문장 상한 (비용; target-only라 truncation=per-sentence 라벨 불변)
MAX_DENSITY = 0.65                     # §2 밀도>65% = 희곡화 교체
N_FIRST, N_THIRD = 2, 4                # §1.2 register 층 (1인칭 2 / 3인칭 4)

# §1.1 유명 6 (연구자 지정 = 약근거). (doc_id, gutenberg_id, heading|None=whole, author, note)
FAMOUS = [
    ("fam_london_fire",      2429,  "TO BUILD A FIRE",                  "Jack London",              "극저 추정·3인칭"),
    ("fam_bierce_owlcreek",  375,   None,                               "Ambrose Bierce",           "저 추정·3인칭"),
    ("fam_gilman_wallpaper", 1952,  None,                               "Charlotte Perkins Gilman", "1인칭 고백체(시험지 register)"),
    ("fam_doyle_speckled",   1661,  "THE ADVENTURE OF THE SPECKLED BAND","Arthur Conan Doyle",      "1인칭 Watson"),
    ("fam_jewett_heron",     74980, "A WHITE HERON",                    "Sarah Orne Jewett",        "3인칭"),
    ("fam_twain_frog",       3189,  "THE JUMPING FROG",                 "Mark Twain",               "1인칭 구어 액자"),
]
# §1.1 Gilman 차단 시 사전지정 교체 = Melville "Bartleby"
GILMAN_REPLACEMENT = ("fam_melville_bartleby", 11231, "BARTLEBY", "Herman Melville")

EXCL_ID = {2776, 3080, 13415, 45524}   # Tier2 gutenberg ids
EXCL_AUTH = ["Henry, O.", "Maupassant", "Chekhov", "Chekhof", "Crane, Stephen", "Poe, Edgar",
             "London, Jack", "Bierce, Ambrose", "Gilman, Charlotte", "Doyle, Arthur Conan",
             "Twain", "Clemens", "Jewett, Sarah", "Melville"]  # Tier2 + 유명슬레이트 (12작가 전원 다름)

UA = {"User-Agent": "Mozilla/5.0 (research; hi-ontop gate1b corpus)"}


def fetch(gid: int) -> str:
    p = CACHE / f"pg{gid}.txt"
    if not p.exists():
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        req = urllib.request.Request(url, headers=UA)
        p.write_bytes(urllib.request.urlopen(req, timeout=30).read())
        time.sleep(0.4)  # politeness
    return p.read_text(encoding="utf-8", errors="replace")


def strip_pg(text: str) -> str:
    lines = text.splitlines()
    try:
        s = next(i for i, l in enumerate(lines) if "START OF THE PROJECT GUTENBERG" in l) + 1
    except StopIteration:
        s = 0
    try:
        e = next(i for i, l in enumerate(lines) if "END OF THE PROJECT GUTENBERG" in l)
    except StopIteration:
        e = len(lines)
    return "\n".join(lines[s:e])


def is_headingish(line: str) -> bool:
    """Story-heading line = short, mostly-uppercase letters (handles 'IX. THE ...', tolerates '[1865]' suffix)."""
    s = line.strip()
    letters = [c for c in s if c.isalpha()]
    if not (4 <= len(s) <= 70) or len(letters) < 3:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) >= 0.8


def slice_story(text: str, heading: str | None) -> str:
    """Whole stripped text if heading None; else from the heading's body occurrence to next story head."""
    lines = text.splitlines()
    if heading is None:
        return text
    hidx = [i for i, l in enumerate(lines) if l.strip().upper().startswith(heading.upper())]
    if not hidx:
        hidx = [i for i, l in enumerate(lines) if heading.upper() in l.strip().upper() and is_headingish(l)]
    if not hidx:
        raise ValueError(f"heading '{heading}' not found")
    start = hidx[-1] + 1  # last occurrence = story body (earlier = TOC)
    nxt = next((i for i in range(start + 3, len(lines)) if is_headingish(lines[i])), len(lines))
    return "\n".join(lines[start:nxt])


def first_story_block(text: str) -> str:
    """For a collection: text from the first story heading to the second."""
    lines = text.splitlines()
    heads = [i for i, l in enumerate(lines) if is_headingish(l)
             and not any(k in l.upper() for k in ("CONTENTS", "GUTENBERG", "PREFACE", "ILLUSTRATION"))]
    if len(heads) < 2:
        return text
    return "\n".join(lines[heads[0] + 1:heads[1]])


QUOTE_RE = re.compile(r'["“”‘’]')
QSPAN_RE = re.compile(r'["“][^"“”]*["”]')


def density(sents: list[str]) -> float:
    """대사밀도 = 직접인용부호 포함 문장 비율 (§2 조작화)."""
    if not sents:
        return 0.0
    return sum(1 for s in sents if QUOTE_RE.search(s)) / len(sents)


def register(sents: list[str]) -> tuple[str, float]:
    """1인칭/3인칭 = 서술부(대사 제외) 대명사 우세. ratio = I / (I+he+she)."""
    narration = QSPAN_RE.sub(" ", " ".join(sents))
    toks = re.findall(r"\b\w+\b", narration.lower())
    n_i = sum(1 for t in toks if t in ("i", "me", "my", "myself"))
    n_3 = sum(1 for t in toks if t in ("he", "him", "his", "she", "her", "hers"))
    ratio = n_i / (n_i + n_3) if (n_i + n_3) else 0.0
    return ("first" if ratio >= 0.40 else "third"), ratio


def bin_of(d: float) -> str:
    return "극저" if d < .10 else "저" if d < .25 else "중" if d < .45 else "고" if d <= .65 else "희곡"


def make_rows(doc_id: str, sents: list[str]) -> list[dict]:
    return [{"doc_id": doc_id, "transition": i, "previous": sents[i - 1], "current": sents[i],
             "boundary": 0, "idm_score": None} for i in range(1, len(sents))]


def measure(doc_id: str, gid: int, heading: str | None, author: str, group: str) -> dict:
    raw = strip_pg(fetch(gid))
    body = slice_story(raw, heading)
    sents = split_sentences(body)
    if len(sents) > MAX_SENT and heading is None:   # likely a collection → first story
        sents = split_sentences(first_story_block(raw))
    full_n = len(sents)
    sents = sents[:CAP]                             # 비용 상한 truncation
    d = density(sents)
    reg, ratio = register(sents)
    return {"doc_id": doc_id, "gutenberg_id": gid, "author": author, "group": group,
            "n_sent": len(sents), "full_n_sent": full_n, "density": round(d, 3), "bin": bin_of(d),
            "register": reg, "reg_ratio": round(ratio, 3), "_sents": sents}


def select_obscure(log: list[dict]) -> list[dict]:
    dl = {int(k): v for k, v in json.load(open("/tmp/pg_downloads.json")).items()}
    rows = list(csv.DictReader(open("/tmp/pgcat.csv", encoding="utf-8")))
    P = []
    for r in rows:
        if r.get("Type") != "Text" or r.get("Language", "") != "en":  # exact en only
            continue
        s = (r.get("Subjects", "") + " " + r.get("Bookshelves", "")).lower()
        if "short stories" not in s:
            continue
        tid = int(r["Text#"])
        auth = r.get("Authors", "")
        if tid in EXCL_ID or ";" in auth or any(x in auth for x in EXCL_AUTH):
            continue
        d = dl.get(tid, 0)
        if d <= 0:
            continue
        h = hashlib.sha256(f"{tid}|{r['Title']}".encode()).hexdigest()
        P.append((d, h, tid, r["Title"].strip(), auth.strip()))
    P.sort()  # (download asc, sha256 asc)  §2 tie-break
    decile = P[:max(1, len(P) // 10)]  # §1.2 bottom decile
    print(f"[obscure] P={len(P)} decile={len(decile)} (dl {decile[0][0]}..{decile[-1][0]})", flush=True)
    need = {"first": N_FIRST, "third": N_THIRD}
    picked: list[dict] = []
    for d, h, tid, title, auth in decile:
        if sum(need.values()) == 0:
            break
        rec = {"gutenberg_id": tid, "download": d, "title": title, "author": auth}
        try:
            m = measure(f"obs_{tid}", tid, None, auth, "obscure")
        except Exception as e:
            rec["decision"] = f"skip:fetch/slice ({type(e).__name__})"; log.append(rec); continue
        rec.update({"n_sent": m["n_sent"], "full_n_sent": m["full_n_sent"], "density": m["density"],
                    "register": m["register"], "bin": m["bin"]})
        if m["full_n_sent"] > MAX_SENT or m["n_sent"] < MIN_SENT:   # collection/too-short 배제
            rec["decision"] = f"skip:len full={m['full_n_sent']}"; log.append(rec); continue
        if m["density"] > MAX_DENSITY:
            rec["decision"] = f"skip:density>{MAX_DENSITY}"; log.append(rec); continue
        if need[m["register"]] <= 0:
            rec["decision"] = f"skip:register-quota-full ({m['register']})"; log.append(rec); continue
        need[m["register"]] -= 1
        rec["decision"] = f"SELECT ({m['register']}, {m['bin']})"
        log.append(rec)
        picked.append(m)
        print(f"  SELECT {tid} dl={d} n={m['n_sent']} dens={m['density']} {m['register']} {m['bin']} | {auth[:25]}", flush=True)
    return picked


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    docs: list[dict] = []
    # famous
    for doc_id, gid, heading, auth, note in FAMOUS:
        try:
            m = measure(doc_id, gid, heading, auth, "famous")
        except Exception as e:
            print(f"[FAMOUS FAIL] {doc_id} ({gid}): {type(e).__name__} {e}", flush=True)
            if doc_id == "fam_gilman_wallpaper":
                rid, rgid, rhead, rauth = GILMAN_REPLACEMENT
                print(f"  -> 사전지정 교체 {rauth}", flush=True)
                m = measure(rid, rgid, rhead, rauth, "famous"); m["replaced_from"] = "gilman"
            else:
                raise
        m["note"] = note
        docs.append(m)
        print(f"[famous] {m['doc_id']:<22} n={m['n_sent']:<4} dens={m['density']} {m['register']}({m['reg_ratio']}) {m['bin']} | {auth}", flush=True)
    # obscure
    log: list[dict] = []
    docs += select_obscure(log)
    (EXP / "selection_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))

    # write master + manifest
    all_rows = []
    for m in docs:
        all_rows += make_rows(m["doc_id"], m.pop("_sents"))
    OUT_MASTER.parent.mkdir(parents=True, exist_ok=True)
    OUT_MASTER.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows))
    (EXP / "manifest.json").write_text(json.dumps(docs, indent=2, ensure_ascii=False))

    print(f"\n=== {len(docs)} docs, {len(all_rows)} transitions -> {OUT_MASTER.name} ===")
    for g in ("famous", "obscure"):
        gg = [m for m in docs if m["group"] == g]
        fp = sum(1 for m in gg if m["register"] == "first")
        print(f" {g}: {len(gg)} docs, 1인칭 {fp} | bins {[m['bin'] for m in gg]}")


if __name__ == "__main__":
    main()
