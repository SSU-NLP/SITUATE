#!/usr/bin/env python3
"""Stage-1 목록 재현율 감사 — 본문에 있는 장소·인물이 목록에서 빠졌는가.

배경: 1차 전문가 검수 FINDINGS A = *"space 어휘공백(비행기가 목록 밖)"*.
Stage-1 이 장소를 빠뜨리면 Stage-2 는 그 문장에서 무대를 고를 수 없어 `unspecified` 로 흘린다.
그 구멍이 실제로 얼마나 큰지 재지 않고는 보강 방식을 정할 수 없다.

측정
  ① 목록 밖 엔티티율 — spaCy 가 잡은 장소(GPE/LOC/FAC)·인물(PERSON) 중 Stage-1 목록에 없는 비율
  ② 그 누락이 라벨을 망치는가 — 장소 엔티티가 있는데 `unspecified` 로 라벨된 문장 중,
     그 엔티티가 목록 밖인 비율. 목록에 있었다면 고를 수 있었을 문장이다

🔴 매칭은 정규화 후 부분일치다(목록은 `studley_grange` 처럼 정규화돼 있고 본문은 `Studley Grange`).
   느슨한 매칭이므로 **누락률은 하한**이다 — 실제 구멍은 이보다 크거나 같다.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs/eacl_phase0/phase1_all_v31"
MASTER = REPO / "data/eacl_phase1/pg_shorts_10700.jsonl"
OUT = REPO / "outputs/experiments/2026-08-02_stage1_audit"
PLACE_ENT = {"GPE", "LOC", "FAC"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def covered(ent: str, lst: list[str]) -> bool:
    """정규화 후 어느 한쪽이 다른 쪽을 포함하면 '목록에 있다'로 본다(느슨 = 누락률 하한)."""
    e = norm(ent)
    if not e:
        return True
    toks = [t for t in e.split("_") if len(t) > 2]
    for x in lst:
        n = norm(x)
        if e == n or e in n or n in e:
            return True
        if toks and any(t in n.split("_") for t in toks):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260802)
    a = ap.parse_args()

    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])

    docs = {}
    for line in MASTER.open(encoding="utf-8"):
        d = json.loads(line)
        docs[d["doc_id"]] = d["sentences"]

    avail = [p.name for p in SRC.iterdir()
             if p.is_dir() and (p / "story_lists.json").exists()
             and (p / "space_states.jsonl").exists() and p.name in docs]
    random.Random(a.seed).shuffle(avail)
    sample = avail[:a.n]
    print(f"story_lists+space 라벨 보유 {len(avail):,}편 중 {len(sample)}편 표본", flush=True)

    S = Counter()
    miss_examples: list[dict] = []
    for i, doc_id in enumerate(sample):
        lists = json.loads((SRC / doc_id / "story_lists.json").read_text())["lists"]
        locs, cast = lists.get("locations", []), lists.get("cast", [])
        sents = docs[doc_id]
        space = {json.loads(l)["item_id"]: json.loads(l).get("location")
                 for l in (SRC / doc_id / "space_states.jsonl").open(encoding="utf-8")}
        for j, doc in enumerate(nlp.pipe(sents, batch_size=64)):
            pl = [e.text for e in doc.ents if e.label_ in PLACE_ENT]
            pe = [e.text for e in doc.ents if e.label_ == "PERSON"]
            S["sent"] += 1
            for e in pl:
                S["place_ent"] += 1
                if not covered(e, locs):
                    S["place_miss"] += 1
            for e in pe:
                S["person_ent"] += 1
                if not covered(e, cast):
                    S["person_miss"] += 1
            if pl and space.get(j) == "unspecified":
                S["unspec_with_place"] += 1
                out = [e for e in pl if not covered(e, locs)]
                if out:
                    S["unspec_place_missing"] += 1
                    if len(miss_examples) < 40:
                        miss_examples.append({"doc": doc_id, "sent": sents[j][:150],
                                              "ents": out, "locs": locs[:8]})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sample)}편", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(
        {"n_docs": len(sample), "counts": dict(S), "examples": miss_examples},
        indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n문장 {S['sent']:,}개")
    print(f"① 장소 엔티티 {S['place_ent']:,}개 중 목록 밖 {S['place_miss']:,} "
          f"({S['place_miss']/max(S['place_ent'],1)*100:.1f}%)")
    print(f"   인물 엔티티 {S['person_ent']:,}개 중 목록 밖 {S['person_miss']:,} "
          f"({S['person_miss']/max(S['person_ent'],1)*100:.1f}%)")
    print(f"② 장소 엔티티가 있는데 unspecified 로 라벨된 문장 {S['unspec_with_place']:,}개")
    print(f"   그중 그 엔티티가 목록 밖 = {S['unspec_place_missing']:,} "
          f"({S['unspec_place_missing']/max(S['unspec_with_place'],1)*100:.1f}%)")
    print(f"   → 전체 문장 대비 {S['unspec_place_missing']/max(S['sent'],1)*100:.2f}%")
    print(f"\n-> {OUT/'audit.json'}")


if __name__ == "__main__":
    main()
