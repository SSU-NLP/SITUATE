#!/usr/bin/env python3
"""Stage-0 코퍼스 정제 (F: boilerplate/front-matter strip / G: 구분선 강제분할).

사전등록 D-B11 master(`repcorpus_master.jsonl`)는 `strip_pg`(PG 헤더/푸터)만 걸러
**본문 안팎 잡물이 잔존**한다:
- (G) 장면 구분선 `* * * * *`이 문장 중간에 융합 → 두 장면이 한 transition.
- (F1) 문서 **머리 front-matter**: 제목·바이라인(저자명)·판본 헤더·저자 약력·풍자 부제·섹션번호가
  서사 첫 문장에 융합되거나 앞에 붙음. Stage-1 cast/protagonist 추출이 문서 전체를 읽으므로 **바이라인
  저자명이 등장인물로 오염**될 수 있어 특히 해롭다.
- (F2) 문서 **뒤/중간 back-matter**: PG 제작 크레딧·Transcriber note·저작권 갱신 안내·서점 광고.

정제 = 동결 `split_sentences` 불변(문장화 *후처리*로만). 세 연산:
- **DROP**  : 순수 boilerplate 문장 통째 제거.
- **STRIP** : 서사에 융합된 front-matter 접두 · `[Transcriber ...]` 접미 제거(서사 잔여 보존).
- **SPLIT** : `* * * * *` 구분선에서 분할·글리프 제거 → 분할점 transition = **gold scene boundary=1**
              (구분선=저자 지정 장면경계, D-B11 개정).

12문서 고정 코퍼스라 **문서 머리 front-matter 는 문서별 명시 스펙(HEAD_SPEC)** 으로 처리(regex 추측보다
감사 가능·견고). 전 문서 머리는 STAGE0_AUDIT.txt 에 before/after 전량 기록.

원본 master 보존, 산출 `repcorpus_master_stage0.jsonl`. dev·test 전 문서 균일 적용(seal 규율).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IN_MASTER = REPO / "outputs/eacl_phase0/repcorpus_master.jsonl"
OUT_MASTER = REPO / "outputs/eacl_phase0/repcorpus_master_stage0.jsonl"

# --- G: 장면 구분선 (asterisk run) ---------------------------------------
DIVIDER_RE = re.compile(r"\*(?:\s*\*)+")

# --- F1: 문서 머리 front-matter — 문서별 명시 스펙 -------------------------
# doc_id -> (n_drop, head_prefix_regex|None)
#   n_drop          = 머리에서 통째 제거할 문장 수(제목/약력/부제/섹션번호 등)
#   head_prefix_re  = 그 다음(첫 잔여) 문장에서 제거할 접두(제목·바이라인·판본이 서사에 융합된 경우)
# 근거 = 각 문서 머리 실사(STAGE0_AUDIT.txt). narrative-start 는 감사 dump 로 확인.
HEAD_SPEC: dict[str, tuple[int, re.Pattern | None]] = {
    # 판본 헤더 융합: "... THE MILLENNIUM FULCRUM EDITION, 1988 I A man stood ..."
    "fam_bierce_owlcreek": (0, re.compile(r"^.*?EDITION,\s*\d{4}\s+[IVX]+\s+")),
    "fam_doyle_speckled":  (0, None),                                   # 머리 깨끗
    # 제목+바이라인 융합: "The Yellow Wallpaper By Charlotte Perkins Gilman It is very seldom ..."
    "fam_gilman_wallpaper": (0, re.compile(r"^The Yellow Wallpaper\s+By\s+Charlotte Perkins Gilman\s+")),
    "fam_jewett_heron":    (0, None),                                   # "I." = 섹션번호(generic DROP)
    "fam_london_fire":     (0, None),                                   # 머리 깨끗
    # Twain 풍자 title-page 3줄: "IN ENGLISH. / THEN IN FRENCH. / THEN CLAWED BACK ..." → 서사 #3부터
    "fam_twain_frog":      (3, None),
    # PG 크레딧+라이선스박스+제목+저작권 융합 → "Marine Captain Jase Thompson ..."
    "obs_25744":           (0, re.compile(r"^.*?\bCE\s+(?=[A-Z])", re.S)),
    # PG 크레딧(#0) + 저자 약력(#1~#3 "Smith has been writing SF...") → 서사 #4 "Smith_ There are many ways..."
    "obs_28922":           (4, re.compile(r"^Smith_\s+")),
    # PG 크레딧(#0~#1) + 제목 "THE STAG-HUNT."(#2) → 서사 #3부터
    "obs_37511":           (3, None),
    # 제목+바이라인: "THE STRIKE AT TOO DRY By Willis Brindley Young Percival came out ..."
    "obs_74140":           (0, re.compile(r"^[A-Z][A-Z0-9 ,'\"&.\-]{4,}?\s+By\s+[A-Z][a-zA-Z.]+\s+[A-Z][a-zA-Z.]+\s+")),
    # 제목+바이라인 융합: "The Recalcitrant By Evelyn Goldstein He was proud ..."
    "obs_74162":           (0, re.compile(r"^The Recalcitrant\s+By\s+Evelyn Goldstein\s+")),
    # 제목+바이라인(#0 "IN MID-AIR By R.", #1 "J.") + 부제 융합(#2 "McSwiney ... Gloriously Six thousand feet ...")
    "obs_77176":           (2, re.compile(r"^McSwiney\b.*?\bGloriously\s+", re.S)),
}

# --- F2: 순수 boilerplate 시그니처 (문장 통째 DROP, 머리/중간/뒤 공통) ------
BOILERPLATE_DROP = [
    re.compile(r"^Produced by\b"),                        # PG 제작 크레딧
    re.compile(r"^Transcriber'?s? Note\b"),               # 순수 Transcriber note
    re.compile(r"\bDistributed Proofreading Team\b"),     # 크레딧 연속(splitter 분리분)
    re.compile(r"copyright on this publication was renewed"
               r"|did not uncover any evidence that"),    # PG 저작권 갱신 안내(obs_74162 편집 blurb)
    re.compile(r"ACCOUNT BOOKS|MISCELLANEOUS BOOKS|TOY BOOKS"
               r"|Manufacturers of every description|JUST PUBLISHED"),  # 서점 광고(obs_37511)
    re.compile(r"^[IVX]{1,4}\.?\s*$"),                    # 홀로 선 섹션번호 "I." "II."
]

# --- F/STRIP 접미: 서사 끝에 붙은 대괄호 Transcriber note ------------------
SUFFIX_TRANSCRIBER_RE = re.compile(r"\s*\[Transcriber.*$", re.S)


class Op:
    """단일 문장에 적용된 변환 기록 (감사용)."""

    def __init__(self, doc: str, idx: int, kind: str, before: str, after: list[str]):
        self.doc, self.idx, self.kind, self.before, self.after = doc, idx, kind, before, after


def clean_sentence(doc: str, idx: int, s: str, ops: list[Op]) -> tuple[list[str], list[bool]]:
    """문장 하나 → (정제 문장 리스트, scene_start 플래그). 빈 리스트 = DROP."""
    orig = s
    # 1) STRIP 접미: 대괄호 Transcriber note
    s2 = SUFFIX_TRANSCRIBER_RE.sub("", s).strip()
    if s2 != s:
        ops.append(Op(doc, idx, "STRIP_SUFFIX", orig, [s2]))
        s = s2
    # 2) DROP: 순수 boilerplate (접미 strip 후에도 서사 없음)
    if s and any(p.search(s) for p in BOILERPLATE_DROP):
        ops.append(Op(doc, idx, "DROP", orig, []))
        return [], []
    # 3) SPLIT: 장면 구분선
    if DIVIDER_RE.search(s):
        pieces = [p.strip() for p in DIVIDER_RE.split(s) if p.strip()]
        if len(pieces) >= 2:
            ops.append(Op(doc, idx, "SPLIT", orig, pieces))
            return pieces, [False] + [True] * (len(pieces) - 1)
        s = pieces[0] if pieces else ""
    return ([s], [False]) if s else ([], [])


def reconstruct_sents(doc_rows: list[dict]) -> list[str]:
    """transition rows → 원 문장 리스트 (previous[0] + 각 current)."""
    rs = sorted(doc_rows, key=lambda r: r["transition"])
    return [rs[0]["previous"]] + [r["current"] for r in rs]


def apply_head(doc: str, sents: list[str], ops: list[Op]) -> list[str]:
    """HEAD_SPEC 적용: 머리 n_drop 문장 제거 + 첫 잔여 문장 접두 strip."""
    n_drop, prefix = HEAD_SPEC.get(doc, (0, None))
    out = list(sents)
    for i in range(n_drop):
        ops.append(Op(doc, i, "DROP_HEAD", sents[i], []))
    out = out[n_drop:]
    if prefix and out:
        m = prefix.sub("", out[0], count=1).strip()
        if m and m != out[0]:
            ops.append(Op(doc, n_drop, "STRIP_PREFIX", out[0], [m]))
            out[0] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-0 corpus cleaning (F strip / G divider split).")
    ap.add_argument("--in-master", type=Path, default=IN_MASTER)
    ap.add_argument("--out-master", type=Path, default=OUT_MASTER)
    ap.add_argument("--audit", action="store_true", help="print every transform before/after")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.in_master.open()]
    bydoc: dict[str, list[dict]] = collections.defaultdict(list)
    order: list[str] = []
    for r in rows:
        if r["doc_id"] not in bydoc:
            order.append(r["doc_id"])
        bydoc[r["doc_id"]].append(r)

    ops: list[Op] = []
    out_rows: list[dict] = []
    stats = collections.Counter()
    per_doc_delta: dict[str, tuple[int, int]] = {}

    for doc in order:
        sents = reconstruct_sents(bydoc[doc])
        sents = apply_head(doc, sents, ops)           # F1: 머리 front-matter
        clean: list[str] = []
        starts: list[bool] = []
        for i, s in enumerate(sents):
            cs, cstart = clean_sentence(doc, i, s, ops)   # F2/STRIP/SPLIT
            clean.extend(cs)
            starts.extend(cstart)
        per_doc_delta[doc] = (len(bydoc[doc]) + 1, len(clean))
        for i in range(1, len(clean)):
            b = 1 if starts[i] else 0
            stats["boundary1"] += b
            out_rows.append({"doc_id": doc, "transition": i, "previous": clean[i - 1],
                             "current": clean[i], "boundary": b, "idm_score": None})

    for op in ops:
        stats[op.kind] += 1

    args.out_master.parent.mkdir(parents=True, exist_ok=True)
    args.out_master.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))

    if args.audit:
        print("=== Stage-0 transforms (before → after) ===")
        for op in ops:
            print(f"\n[{op.doc} #{op.idx}] {op.kind}")
            print(f"  BEFORE: {op.before!r}")
            for a in op.after:
                print(f"  AFTER : {a!r}")
            if not op.after:
                print("  AFTER : <dropped>")

    print("\n=== summary ===")
    print(f"in  : {len(rows)} transitions, {len(order)} docs")
    print(f"out : {len(out_rows)} transitions -> {args.out_master.name}")
    print(f"ops : DROP_HEAD={stats['DROP_HEAD']} DROP={stats['DROP']} STRIP_PREFIX={stats['STRIP_PREFIX']} "
          f"STRIP_SUFFIX={stats['STRIP_SUFFIX']} SPLIT={stats['SPLIT']} gold_boundary1={stats['boundary1']}")
    print("per-doc sentence count (orig -> clean):")
    for doc in order:
        o, c = per_doc_delta[doc]
        flag = "" if o == c else f"  Δ{c - o:+d}"
        print(f"  {doc:<22} {o:>4} -> {c:>4}{flag}")


if __name__ == "__main__":
    main()
