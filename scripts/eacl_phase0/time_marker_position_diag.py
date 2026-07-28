#!/usr/bin/env python3
"""time 축 진단 — temporal_shift_marker 라벨의 '표지 위치'(문두 vs 문중) 분포.

배경(사용자 2026-07-23): frozen 프롬프트가 sequence 에만 "opens with"(문두) 조건을 걸고 jump/simultaneity
엔 안 걸어 비대칭. 문중 'then'(sequence)은 absent 여야 하나 문중 'later'(jump)는 쳐질 수 있음. 같은 위치인데
표지 종류로 판정이 갈림 = 결함. 이 진단은 캐시된 라벨에서 그 문중 표지 비율을 재 사람(strict-초두)과 모델의
체계적 어긋남을 추정한다. 예산 0(캐시 읽기 + regex).

판정: non-absent 라벨 문장마다 카테고리 표지어의 최초 등장 '단어 인덱스'를 구해
  initial(0, 선행 접속사/따옴표 허용해 실질 문두) / near(1~2) / mid(>=3) / none(regex 미검출)
로 분류. mid 가 상당수면 strict-초두 규칙 하에서 absent 로 뒤집힐 라벨(=사람과 어긋남)이 많다는 뜻.
"""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/eacl_phase0/repcorpus_s5_eismv2_run1"
MASTER = ROOT / "outputs/eacl_phase0/repcorpus_master.jsonl"

# 문장이 시간부사(구)로 '시작'하는가만 판정 (opens-with 규칙 준수 여부). 문두만 검사하므로
# when/while/as 도 포함(문두면 대개 시간·종속 도입부). 표지어 탐지가 아니라 '문두 여부'가 목적.
_NUM = r'(an?|one|two|three|four|five|six|seven|eight|nine|ten|some|several|a few|many|the)'
_UNIT = r'(hours?|days?|weeks?|months?|years?|moments?|minutes?|seconds?|nights?|mornings?|winters?|summers?)'
START_TEMPORAL = re.compile(
    r'^(then|next|after that|afterwards?|later|soon|presently|eventually|meanwhile|suddenly|'
    r'shortly|immediately|instantly|instantaneously|directly|forthwith|momentarily|'
    r'at last|at length|at once|at (that|this) (moment|instant|time|point)|now|'
    r'by (nightfall|dawn|daybreak|morning|evening|night|noon|then|now|the (time|next)|and by)|'
    r'from (this|that)\b|the (next|following|same) (morning|day|night|week|month|year|instant|moment)|'
    r'that (night|morning|day|evening|instant|moment)|last (week|night|year|month|spring|summer|autumn|winter|evening|time)|'
    + _NUM + r' ' + _UNIT + r' (later|after|afterwards?|before|ago)|'
    r'when\b|while\b|as \b|before\b|after\b|once\b|in (a|an) (moment|instant|while)|'
    r'for a (moment|while|time)|all (this|the) (time|while)|during\b|thereupon|thereafter|'
    r'henceforth|the moment|no sooner|scarcely|hardly had)',
    re.IGNORECASE)
LEAD = re.compile(r'^[\s"“”\'‘’\-—(]+')                  # 선행 따옴표·대시·괄호
CONJ = re.compile(r'^(and|but|so|yet|or|nor)\b[\s,]*', re.IGNORECASE)  # 선행 등위접속사 1개 허용


def sequence(rows):
    rows = sorted(rows, key=lambda x: int(x["transition"]))
    return [rows[0]["previous"], *[x["current"] for x in rows]]


def opens_with_temporal(sent: str) -> bool:
    s = LEAD.sub("", sent.strip())
    s = CONJ.sub("", s)          # "And then..." -> "then..."
    s = LEAD.sub("", s)
    return bool(START_TEMPORAL.match(s))


def bucket(sent: str, _label: str) -> str:
    return "initial" if opens_with_temporal(sent) else "not_initial"


def main() -> None:
    by_doc = defaultdict(list)
    for r in (json.loads(x) for x in MASTER.read_text().splitlines() if x.strip()):
        by_doc[str(r["doc_id"])].append(r)
    sents = {d: sequence(rs) for d, rs in by_doc.items()}

    per_label = defaultdict(Counter)   # label -> bucket counts
    examples = defaultdict(list)
    tot = Counter()
    for ddir in sorted(p for p in RUN.iterdir() if p.is_dir()):
        doc = ddir.name
        tp = ddir / "time_states.jsonl"
        if not tp.exists() or doc not in sents:
            continue
        for st in (json.loads(x) for x in tp.read_text().splitlines() if x.strip()):
            lab = st["temporal_shift_marker"]
            tot[lab] += 1
            if lab == "absent":
                continue
            i = int(st["item_id"])
            if i >= len(sents[doc]):
                continue
            sent = sents[doc][i]
            b = bucket(sent, lab)
            per_label[lab][b] += 1
            if b == "not_initial" and len(examples[lab]) < 5:
                examples[lab].append((doc, i, sent[:115]))

    print(f"전체 라벨 분포: {dict(tot)}  (non-absent = {sum(v for k,v in tot.items() if k!='absent')})\n")
    print(f"{'label':13s} {'n':>4s} {'문두시작':>8s} {'문두아님':>8s}   문두아님%")
    ni = 0
    nonabs = 0
    for lab in ("sequence", "jump", "simultaneity"):
        c = per_label[lab]
        n = sum(c.values())
        if not n:
            continue
        nonabs += n
        ni += c["not_initial"]
        print(f"{lab:13s} {n:>4d} {c['initial']:>8d} {c['not_initial']:>8d}   {c['not_initial']/n:>7.1%}")
    print(f"\n문두로 시작 안 함 = {ni}/{nonabs} = {ni/nonabs:.1%} of non-absent time 라벨")
    print("  → 이만큼이 '문장이 시간부사로 시작하지 않는데도' 시간전환으로 라벨됨.")
    print("  → sequence 는 'opens with' 규칙이 있어 문두가 많아야 정상; jump/simultaneity 는 규칙에")
    print("     위치조건이 없어 문두아님이 많으면 = 비대칭 결함(문중 표지로 발화) 실증.\n")
    for lab in ("sequence", "jump", "simultaneity"):
        for doc, i, s in examples[lab]:
            print(f"  [{lab}/문두아님] {doc}#{i}: {s}")


if __name__ == "__main__":
    main()
