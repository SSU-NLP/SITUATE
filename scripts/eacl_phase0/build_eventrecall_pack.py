#!/usr/bin/env python3
"""Build the sealed held-out masters from EventRecall (Panela et al. 2025).

시험지 = *Born a Crime* 발췌 3편 (Run · GoHitler · MyMothersLife), 주석자 20명.
원자료 = `benchmarks/EventRecall/data/` (MIT 코드 / 텍스트는 저작권 — 재배포 금지).

## 문장 정책 (dev-3 와 동일하게 동결)
`build_narrative_pack.split_sentences` (문장부호 분리) + `build_pieman_pack.align_words_to_sentences`
(difflib 단어→문장 정렬). 경계 사상 규약도 dev-3(Michelmann)와 같다 —
**누른 단어가 속한 문장이 segment-final** 이고 경계는 **그 다음 전이**에 놓인다.

## gold 두 판 (🔒 채점 **전에** 사전 선언, `EACL/phase0-decisions-and-runs.md` D-B62)
1. **goldauthor = 원저자 규칙 복원.** `code/segmentation/gpt_segmentation_analysis.Rmd` 의
   "peak" 정의는 `avg_responses != 0` = **20명 중 1명이라도 누른 지점**. 그대로 복원한다.
2. **게이트** — goldauthor 의 경계율이 *그 문서 주석자 1인 평균 밀도의 2배*를 넘으면
   **gold 로 사용 불가** 판정. 판정은 gold 밀도만으로 하며 **어떤 방법의 점수도 보지 않는다.**
3. **goldtopn = 밀도 정합 top-N**(게이트 실패 시의 사전 확정 대체). 문장별 누름 수 상위 N개,
   `N = round(주석자 1인 평균 경계 수)`. 동점은 (누름 수 내림, 문장 번호 오름)으로 끊는다.
   문장 단위 집계 자체가 저자 평활(ksmooth)의 역할을 하며, 임의의 bandwidth 를 새로 정하지 않는다.

산출: `outputs/eacl_phase0/{doc}_eventrecall_{gold}_master.jsonl` · 리포트 = stdout + JSON.

    .venv/bin/python scripts/eacl_phase0/build_eventrecall_pack.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_narrative_pack import split_sentences                    # noqa: E402
from build_pieman_pack import align_words_to_sentences              # noqa: E402

SRC = REPO / "benchmarks/EventRecall/data"
OUT = REPO / "outputs/eacl_phase0"
WORD = re.compile(r"[a-z0-9']+")

# story_id -> (doc_id, 파일명). 첫 단어로 검증한다(아래 assert).
STORIES = {"1": ("run", "Run.txt"),
           "2": ("gohitler", "GoHitler.txt"),
           "3": ("mymotherslife", "MyMothersLife.txt")}
GATE_RATIO = 2.0        # 🔒 사전 선언: 주석자 1인 평균 밀도의 몇 배까지 gold 로 허용하나


def load_words(story_id: str) -> list[str]:
    """`visual_time_word_data.csv` 에서 story 의 단어 열 (word_number 순)."""
    seen: dict[int, str] = {}
    with (SRC / "segmentation/visual_time_word_data.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["story_id"] != story_id:
                continue
            w = int(r["word_number"])
            if w in seen and seen[w] != r["word"]:
                raise ValueError(f"story {story_id} word {w}: 단어 불일치 {seen[w]!r}/{r['word']!r}")
            seen[w] = r["word"]
    return [seen[i] for i in sorted(seen)]


def load_presses(story_id: str) -> dict[str, list[int]]:
    """`visual_segmentation_data.csv` 에서 주석자별 누름 word_number 목록."""
    out: dict[str, list[int]] = defaultdict(list)
    with (SRC / "segmentation/visual_segmentation_data.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["story_id"] == story_id:
                out[r["subject"]].append(int(r["word_number"]))
    return dict(out)


def build_doc(story_id: str) -> dict:
    """한 편의 문장·전이·두 gold 를 만든다."""
    doc_id, fname = STORIES[story_id]
    text = (SRC / "stories" / fname).read_text(encoding="utf-8")
    sents = split_sentences(text)
    n = len(sents)

    words = load_words(story_id)
    toks = [WORD.findall(w.lower().replace("’", "'")) for w in words]
    flat = [t[0] if t else "" for t in toks]
    assert flat[0] in sents[0].lower().replace("’", "'"), \
        f"{doc_id}: 첫 단어 {flat[0]!r} 가 첫 문장에 없다 — story_id 매핑 확인"
    word_sent = align_words_to_sentences(flat, sents)

    presses = load_presses(story_id)
    # 주석자별: 누름 -> 문장 -> 전이(그 문장이 segment-final). 마지막 문장 누름은 전이가 없어 버린다.
    per_subj: dict[str, set[int]] = {}
    for subj, ws in presses.items():
        per_subj[subj] = {word_sent[w - 1] for w in ws if 0 <= word_sent[w - 1] <= n - 2}
    subj_counts = [len(v) for v in per_subj.values()]
    mean_subj = sum(subj_counts) / len(subj_counts)

    votes = Counter()
    for v in per_subj.values():
        votes.update(v)

    gold_author = sorted(votes)                                   # ≥1명 = 저자 규칙
    n_top = round(mean_subj)
    gold_topn = sorted(sorted(votes, key=lambda t: (-votes[t], t))[:n_top])

    n_trans = n - 1
    return {"doc_id": doc_id, "story_id": story_id, "sents": sents, "n_sent": n,
            "n_trans": n_trans, "n_words": len(words), "n_subjects": len(per_subj),
            "subj_counts": sorted(subj_counts), "mean_subj": mean_subj,
            "subj_rate": mean_subj / n_trans, "votes": {int(k): int(v) for k, v in votes.items()},
            "gold_author": gold_author, "gold_topn": gold_topn, "n_top": n_top,
            "rate_author": len(gold_author) / n_trans, "rate_topn": len(gold_topn) / n_trans}


def write_master(doc: dict, gold_key: str, tag: str) -> Path:
    """dev-3 master 와 같은 스키마로 쓴다 (doc_id·transition·previous·current·boundary)."""
    gold = set(doc[gold_key])
    p = OUT / f"{doc['doc_id']}_eventrecall_{tag}_master.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for t in range(doc["n_trans"]):
            f.write(json.dumps({
                "doc_id": doc["doc_id"], "transition": t + 1,
                "previous": doc["sents"][t], "current": doc["sents"][t + 1],
                "boundary": int(t in gold),
                "n_votes": doc["votes"].get(t, 0), "n_subjects": doc["n_subjects"],
                "gold_rule": tag,
                "source": "ryanapanela/EventRecall (Panela et al. 2025) visual segmentation",
            }, ensure_ascii=False) + "\n")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="master jsonl 을 실제로 쓴다")
    args = ap.parse_args()

    docs = [build_doc(s) for s in sorted(STORIES)]
    print("=" * 96)
    print("시험지 gold 게이트 — 방법 점수는 아직 하나도 보지 않았다 (gold 밀도만으로 판정)")
    print("=" * 96)
    print(f"{'doc':16s} {'문장':>5s} {'전이':>5s} {'주석자':>4s} {'1인평균':>7s} {'1인밀도':>8s} "
          f"{'저자규칙':>8s} {'저자밀도':>8s} {'배수':>6s} {'top-N':>6s}")
    for d in docs:
        print(f"{d['doc_id']:16s} {d['n_sent']:5d} {d['n_trans']:5d} {d['n_subjects']:4d} "
              f"{d['mean_subj']:7.1f} {d['subj_rate']:8.4f} {len(d['gold_author']):8d} "
              f"{d['rate_author']:8.4f} {d['rate_author'] / d['subj_rate']:6.2f} {d['n_top']:6d}")

    ratios = [d["rate_author"] / d["subj_rate"] for d in docs]
    passed = all(r <= GATE_RATIO for r in ratios)
    print(f"\n게이트 (저자규칙 밀도 <= 1인 평균 밀도 x {GATE_RATIO}): "
          f"{'PASS' if passed else 'FAIL'} — 최대 배수 {max(ratios):.2f}")
    print(f"⇒ main gold = {'goldauthor (저자 규칙)' if passed else 'goldtopn (밀도 정합 top-N)'} · "
          f"다른 판은 부록 병기")

    rep = {"gate_ratio_limit": GATE_RATIO, "passed": passed,
           "main_gold": "goldauthor" if passed else "goldtopn",
           "docs": [{k: v for k, v in d.items() if k != "sents"} for d in docs]}
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        for d in docs:
            for key, tag in (("gold_author", "goldauthor"), ("gold_topn", "goldtopn")):
                print(f"[write] {write_master(d, key, tag)}")
        rp = OUT / "eventrecall_gold_gate.json"
        rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[write] {rp}")
    else:
        print("\n(--write 없이 실행 — 파일을 쓰지 않았다)")


if __name__ == "__main__":
    main()
