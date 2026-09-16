#!/usr/bin/env python3
"""Phase-1 학습 코퍼스 구축 — PG 단편 수집 (권 → 편 분할).

사용자 지정 6단계 (2026-07-25):
  ① PG 덤프에서 영어 단편 레코드          — 카탈로그 필터
  ② 권 → 편 분할 (목차·제목 헤딩)          — **신규**
  ③ 길이 필터 30~700 문장
  ④ 작가당 상한 N 편
  ⑤ 편 내부에서 Stage-0 (F/G)              — **반드시 ② 다음**
  ⑥ 격리 확인 (QC 12편·Tier2·시험지 제외)

D-M8: 서사 픽션 단일(스크린플레이 혼합 금지). 선정 기준에 대사 비중 스펙트럼 다양화 포함
→ 편별 대사 밀도를 메타로 기록해 하류에서 층화 가능하게 한다.

⚠️ ⑤ 를 ② 앞에 두면 안 된다 — 권 단위 front-matter 제거는 편 경계를 모르는 상태의 추측이 된다.
편으로 자른 뒤에야 "이 편의 머리 잡물"이 정의된다.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from build_narrative_pack import split_sentences  # frozen sentence splitter  # noqa: E402
from fetch_repcorpus import (EXCL_AUTH, EXCL_ID, UA, density, is_headingish,  # noqa: E402
                             register, strip_pg)

CAT = REPO / "data/eacl_phase1/pgcat.csv"
CACHE = REPO / "data/eacl_phase1/pg_cache"
DOWNLOADS = REPO / "data/eacl_phase1/pg_downloads.json"   # {gutenberg_id(str): 누적 다운로드 수}
# ⑥ 격리: QC 12편(repcorpus) gutenberg id + Tier2 id. 시험지(Born a Crime)는 PG 밖이라 id 충돌 없음.
QC_IDS = {2429, 375, 1952, 1661, 74980, 3189}          # 유명 6 (fam_*)
# 무명 6(obs_*)은 doc_id 가 gutenberg id 그대로다 → manifest 에서 읽어 합친다.
FRONT_JUNK = re.compile(
    r"(project gutenberg|transcriber|etext|copyright|illustrat|contents|produced by|"
    r"proofread|distributed proofread|\[note|footnote|printed in|first published|"
    r"all rights reserved|typographical error)", re.I)
HEAD_SKIP = ("CONTENTS", "GUTENBERG", "PREFACE", "ILLUSTRATION", "INTRODUCTION",
             "APPENDIX", "INDEX", "TRANSCRIBER", "FOOTNOTE", "COPYRIGHT", "DEDICATION",
             "TITLE PAGE", "HALF-TITLE", "COLOPHON", "ERRATA", "BIBLIOGRAPH", "NOTE ON")
# 구조 구분(작품 아님): 로마숫자·아라비아 단독, CHAPTER/PART/BOOK/SECTION/CANTO + 번호
STRUCT_HEAD = re.compile(
    r"^\s*(?:(?:CHAPTER|PART|BOOK|SECTION|CANTO|ACT|SCENE|VOLUME|LETTER|NO)\b[\s.:—-]*)?"
    r"(?:[IVXLCDM]+|\d+|FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)"
    r"[\s.:—-]*$", re.I)
# 편 제목으로 인정할 최소 실질 어휘 수 (숫자·구조어 제외)
MIN_HEAD_WORDS = 2


# ------------------------------------------------------------------ ① 후보
DEV3_AUTH = ["Pohl, Frederik"]     # dev-3 tunnel 저자. pieman·monkey 는 PG 밖이라 id 충돌 없음

# ④ 상한 면제: 저자 필드가 총칭이거나 편집자 크레딧뿐인 선집.
# 이런 권의 각 편은 서로 다른 작가가 썼는데 카탈로그에는 편집자 한 명만 적혀 있다. 여기에 상한을
# 걸면 `Various` 736편을 한 작가로 세어 잘라내게 되어, 상한이 막으려던 "한 작가의 문체 지배"와
# 정반대로 코퍼스에서 가장 다양한 부분을 버린다.
ANTH_PLACEHOLDER = re.compile(r"\b(various|anonymous|unknown|unidentified)\b", re.I)
ANTH_EDITOR = re.compile(r"\[(editor|compiler|contributor)\]", re.I)


def is_anthology(author: str) -> bool:
    """저자 미상 선집인가 — 저자 필드 첫 항목이 총칭이거나 편집자 크레딧."""
    first = author.split(";")[0]
    return bool(ANTH_PLACEHOLDER.search(first) or ANTH_EDITOR.search(first))


def candidates(cat: Path, extra_excl: set[int], excl_auth: list[str]) -> list[dict]:
    rows = list(csv.DictReader(cat.open(encoding="utf-8")))
    out = []
    for r in rows:
        if r.get("Language") != "en" or r.get("Type") != "Text":
            continue
        if "short stories" not in r.get("Subjects", "").lower():
            continue
        gid = int(r["Text#"])
        if gid in extra_excl:                              # ⑥
            continue
        if any(a in r.get("Authors", "") for a in excl_auth):   # ⑥
            continue
        out.append({"gid": gid, "title": r.get("Title", ""), "author": r.get("Authors", ""),
                    "subjects": r.get("Subjects", ""), "issued": r.get("Issued", ""),
                    "locc": r.get("LoCC", ""), "bookshelves": r.get("Bookshelves", "")})
    return out


def fetch(gid: int, delay: float) -> str | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{gid}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8", errors="ignore")
    for url in (f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
                f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt"):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as f:
                t = f.read().decode("utf-8", errors="ignore")
            p.write_text(t, encoding="utf-8")
            time.sleep(delay)
            return t
        except Exception:
            continue
    time.sleep(delay)
    return None


# --------------------------------------------------- ② 권 → 편 분할 (신규)
def story_heading(line: str) -> bool:
    """편 제목인가. 章 번호(XXIII, PART SECOND)·서지 헤딩은 배제 — 이것들은 작품 경계가 아니다."""
    s = line.strip()
    if not is_headingish(s) or any(k in s.upper() for k in HEAD_SKIP):
        return False
    if STRUCT_HEAD.match(s):                      # 'XXIII', 'PART SECOND', 'CHAPTER IV'
        return False
    core = re.sub(r"^(?:CHAPTER|PART|BOOK|SECTION|CANTO)\b[\s.:—-]*[IVXLCDM\d]*[\s.:—-]*", "",
                  s, flags=re.I)
    words = [w for w in re.findall(r"[A-Za-z']+", core) if len(w) > 1]
    return len(words) >= MIN_HEAD_WORDS


def split_volume(text: str) -> tuple[list[tuple[str, str]], str]:
    """본문 헤딩 기준으로 권을 편으로 자른다. ([(제목, 본문), ...], 사유)

    TOC 는 앞쪽에 헤딩이 몰려 나오므로 **각 헤딩의 마지막 출현**을 본문 위치로 본다
    (fetch_repcorpus.slice_story 와 같은 규약).

    ⚠️ 章 번호가 다수인 권(= 단일 중·장편)은 자르지 않고 **통째로 버린다** — 챕터를 '편'으로 세면
    단편 코퍼스가 아니라 장편 조각 코퍼스가 된다(스모크에서 톨스토이 'XXIII'·'PART SECOND' 로 확인).
    """
    lines = strip_pg(text).splitlines()
    heads = [(i, l.strip()) for i, l in enumerate(lines) if story_heading(l)]
    struct = sum(1 for l in lines if is_headingish(l) and STRUCT_HEAD.match(l.strip()))
    if struct >= max(3, 2 * len(heads)):
        return [], "chaptered_work"               # 단일 장편/중편 → 제외
    cand = heads
    if len(cand) < 2:
        return [("", "\n".join(lines))], "single"
    last: dict[str, int] = {}
    for i, h in cand:
        last[h.upper()] = i                      # 같은 헤딩의 마지막 출현 = 본문
    pos = sorted((i, h) for h, i in last.items())
    if len(pos) < 2:
        return [("", "\n".join(lines))], "single"
    out = []
    for k, (i, h) in enumerate(pos):
        end = pos[k + 1][0] if k + 1 < len(pos) else len(lines)
        body = "\n".join(lines[i + 1:end]).strip()
        if body:
            out.append((h, body))
    return out, "split"


# ------------------------------------------- ⑤ 편 내부 Stage-0 (F/G, 일반형)
def stage0_piece(sents: list[str]) -> tuple[list[str], dict]:
    """편 단위 F(잡물)·G(구분선) 처리. 12문서 HEAD_SPEC 대신 휴리스틱(1만 편 규모라 명시 스펙 불가)."""
    ops = Counter()
    out: list[str] = []
    for s in sents:
        t = s.strip()
        if not t:
            continue
        if re.fullmatch(r"[\W_]{3,}", t):                     # G: 구분선 단독 문장
            ops["split_marker"] += 1
            continue
        parts = re.split(r"\s*(?:\*\s*){3,}\s*|\s*[_=~\-]{3,}\s*", t)  # G: 문장 내부 융합 구분선
        if len(parts) > 1:
            ops["split_inline"] += 1
        for q in parts:
            q = q.strip()
            if not q:
                continue
            if FRONT_JUNK.search(q) and len(q) < 200:          # F: boilerplate
                ops["drop_junk"] += 1
                continue
            out.append(q)
    # F: 머리쪽 잔여 잡물 + 줄바꿈된 헤딩 잔재(대문자 연속) 제거
    while out and (FRONT_JUNK.search(out[0]) or _is_heading_remnant(out[0])):
        out.pop(0); ops["drop_head"] += 1
    return out, dict(ops)


def _is_heading_remnant(s: str) -> bool:
    """줄바꿈으로 잘린 헤딩 꼬리(예: 'WRITING IT: AND THENCE PASSES TO ...')."""
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(1 for c in letters if c.isupper()) / len(letters) >= 0.8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=10000, help="목표 편 수")
    ap.add_argument("--per-author", type=int, required=True, help="④ 작가당 상한 N 편")
    ap.add_argument("--min-sent", type=int, default=30)
    ap.add_argument("--max-sent", type=int, default=700)
    ap.add_argument("--min-median-words", type=float, default=5.0,
                    help="문장 중앙 어휘수 하한 — 서지/목록 잔재 배제(스모크: 잡물 1.0 vs 본문 11+)")
    ap.add_argument("--isolate", choices=["legacy", "dev3"], default="legacy",
                    help="legacy = 예비 점검 12편·Tier2·유명 저자 14인·명작 앤솔로지 전부 제외. "
                         "dev3 = 평가셋(dev-3) 저자만 제외 — Tier2 의 held-out 지위가 D-M13 에서 "
                         "폐기됐으므로 나머지를 학습에서 뺄 이유가 없다")
    ap.add_argument("--exclude-download-pct", type=float, default=0.0,
                    help="다운로드 수 상위 (100-P)%% 권 제외. 90 이면 90분위 이상 제외(0=끄기). "
                         "LLM 이 널리 읽힌 작품의 줄거리를 외운 채 라벨을 다는 위험을 줄인다")
    ap.add_argument("--max-volumes", type=int, default=0, help="권 수 상한(0=무제한, 스모크용)")
    ap.add_argument("--delay", type=float, default=0.5, help="PG 요청 간 대기(초)")
    ap.add_argument("--out", type=Path, default=REPO / "data/eacl_phase1/pg_shorts.jsonl")
    ap.add_argument("--manifest", type=Path, default=REPO / "data/eacl_phase1/pg_shorts_manifest.json")
    a = ap.parse_args()

    # ⑥ 격리
    extra: set[int] = set()
    excl_auth = DEV3_AUTH
    if a.isolate == "legacy":
        # 무명 6편 id 를 repcorpus manifest 에서 회수해 격리 목록에 추가
        mf = REPO / "outputs/experiments/2026-07-22_gate1b_repcorpus/manifest.json"
        if mf.exists():
            try:
                m = json.loads(mf.read_text())
                for d in (m if isinstance(m, list) else m.get("docs", [])):
                    g = d.get("gutenberg_id")
                    if g:
                        extra.add(int(g))
            except Exception:
                pass
        extra |= QC_IDS | EXCL_ID
        excl_auth = EXCL_AUTH
    print(f"[⑥ 격리 {a.isolate}] 제외 id {len(extra)}개 · 제외 작가 {len(excl_auth)}명 {excl_auth}",
          flush=True)

    cands = candidates(CAT, extra, excl_auth)
    print(f"[① 후보] {len(cands)} 권", flush=True)

    dl_cut = None
    if a.exclude_download_pct:
        # 분위는 **후보 권 집합** 위에서 정의한다(편 분할 전이라 편 가중은 계산 불가).
        dl = json.loads(DOWNLOADS.read_text())
        have = [dl[str(c["gid"])] for c in cands if str(c["gid"]) in dl]
        import numpy as np
        dl_cut = float(np.percentile(have, a.exclude_download_pct))
        before = len(cands)
        cands = [c for c in cands if dl.get(str(c["gid"]), 0) < dl_cut]
        print(f"[다운로드 필터] {a.exclude_download_pct:.0f}분위 = {dl_cut:,.0f}회 이상 제외 · "
              f"{before} → {len(cands)} 권 (-{before - len(cands)})", flush=True)
    if a.max_volumes:
        cands = cands[:a.max_volumes]

    per_author: Counter = Counter()
    stats = Counter()
    rows: list[dict] = []
    seen_titles: set[tuple[str, str]] = set()
    for vi, c in enumerate(cands):
        if len(rows) >= a.target:
            break
        anth = is_anthology(c["author"])                 # ④ 선집은 상한 면제
        if not anth and per_author[c["author"]] >= a.per_author:   # ④ 권 단위 조기 스킵
            stats["skip_author_cap"] += 1
            continue
        txt = fetch(c["gid"], a.delay)
        if not txt:
            stats["fetch_fail"] += 1
            continue
        pieces, reason = split_volume(txt)                # ②
        stats[f"vol_{reason}"] += 1
        stats["volumes"] += 1
        stats["pieces_raw"] += len(pieces)
        for pi, (title, body) in enumerate(pieces):
            if len(rows) >= a.target:
                break
            if not anth and per_author[c["author"]] >= a.per_author:   # ④
                stats["skip_author_cap_piece"] += 1
                continue
            key = (c["author"], title.upper())
            if title and key in seen_titles:
                stats["dup_title"] += 1
                continue
            sents = split_sentences(body)
            sents, ops = stage0_piece(sents)              # ⑤ (② 다음)
            if not (a.min_sent <= len(sents) <= a.max_sent):   # ③
                stats["len_filter"] += 1
                continue
            # 서지·목록 잔재 배제: 스모크에서 중앙 어휘수 1.0 vs 본문 11+ 로 완전 분리
            med = statistics.median(len(x.split()) for x in sents)
            if med < a.min_median_words:
                stats["median_words_filter"] += 1
                continue
            d = density(sents)
            rows.append({
                "doc_id": f"pg{c['gid']}_{pi:03d}", "gutenberg_id": c["gid"],
                "piece_index": pi, "piece_title": title, "author": c["author"],
                "title": c["title"], "subjects": c["subjects"],
                # 층화는 지금 하지 않는다(SF 46% 기록만). 메타를 저장해 두면 나중에 부분추출 가능,
                # 저장 안 하면 복구 불가 — 사용자 지시 2026-07-25.
                "issued": c["issued"], "locc": c["locc"], "bookshelves": c["bookshelves"],
                "is_sf": "science fiction" in c["subjects"].lower(),
                "n_sent": len(sents), "dialogue_density": round(d, 4),
                "register": register(sents)[0], "stage0_ops": ops, "sentences": sents,
            })
            if anth:
                stats["anthology_exempt"] += 1
            else:
                per_author[c["author"]] += 1
            seen_titles.add(key)
        if vi % 25 == 0:
            print(f"  권 {vi}/{len(cands)}  편 {len(rows)}/{a.target}  "
                  f"(len필터 탈락 {stats['len_filter']})", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"n_pieces": len(rows), "n_authors": len(per_author),
           "per_author_cap": a.per_author, "len_range": [a.min_sent, a.max_sent],
           "exclude_download_pct": a.exclude_download_pct, "download_cut": dl_cut,
           "isolate": a.isolate,
           "stats": dict(stats),
           "sent_total": sum(r["n_sent"] for r in rows),
           "min_median_words": a.min_median_words,
           "register_bins": dict(Counter(r["register"] for r in rows)),
           "dialogue_density_bins": dict(Counter(
               ("low" if r["dialogue_density"] < .15 else
                "mid" if r["dialogue_density"] < .40 else "high") for r in rows)),
           "top_authors": per_author.most_common(15),
           "sf_share": round(sum(1 for r in rows if r["is_sf"]) / max(len(rows), 1), 4),
           "issued_bins": dict(Counter((r["issued"] or "?")[:4] for r in rows).most_common(20))}
    a.manifest.write_text(json.dumps(man, ensure_ascii=False, indent=2))
    print(f"\n[완료] {len(rows)} 편 · {man['sent_total']} 문장 · 작가 {len(per_author)}명")
    print(f"  인칭 분포:     {man['register_bins']}")
    print(f"  대사밀도 분포: {man['dialogue_density_bins']}")
    print(f"  통계: {dict(stats)}")
    print(f"  -> {a.out}\n  -> {a.manifest}")


if __name__ == "__main__":
    main()
