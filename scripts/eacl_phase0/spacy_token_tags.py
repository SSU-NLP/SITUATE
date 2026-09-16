#!/usr/bin/env python3
"""dev-3 문장별 spaCy 토큰 태그 덤프 (M9 엔티티 필터 ablation 용).

⚠️ **별도 venv 에서 돌린다.** `spacy` 는 `.venv` 에 없고 `.venv-eacl-extractors`
(spacy 3.8.14 + en_core_web_sm, torch 없음)에만 있다. LM 과 같은 프로세스에서 못 돌리므로
2단계 분리는 선택이 아니라 환경 제약이다.

    .venv-eacl-extractors/bin/python scripts/eacl_phase0/spacy_token_tags.py \
        --split dev3 --out outputs/experiments/<name>/cache/dev3_spacy_tags.json

문장 로딩은 `baseline_compare.load_dataset` 를 **복제**한다(그 모듈은 torch 를 import 해서
이 venv 에서 못 쓴다). 복제가 어긋나면 조용히 다른 문서를 태깅하게 되므로 문장별 sha1 을
같이 저장하고, 소비 측(`run_attn_entity_dev3.py`)이 자기 로딩 결과와 대조해 강제한다.

출력 스키마:
    {"split": "dev3",
     "spacy": "3.8.14", "model": "en_core_web_sm",
     "docs": {"<doc_id>": {"sent_sha": ["<sha1>", ...],
                           "tags": [[{"s":int,"e":int,"pos":str,"ent":str,"lemma":str}, ...], ...]}}}
`s`/`e` 는 **문장-로컬 char offset** (원문 문장 기준. LM 쪽 `"\\n"+문장` 오프셋과는
`len(sep)` 만큼 다르다 — 소비 측에서 이미 sep 을 뺀 로컬 오프셋을 쓴다).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# 🔒 baseline_compare.load_dataset:54-74 복제. 바꾸면 소비 측 sha1 대조가 실패한다.
DEV3 = ["pieman", "tunnel", "monkey"]


def load_sents(split: str) -> dict[str, list[str]]:
    """(doc_id -> 문장 리스트). `baseline_compare.load_dataset` 와 동일 규약."""
    if split != "dev3":
        raise ValueError(f"현재 dev3 만 지원: {split!r}")
    out: dict[str, list[str]] = {}
    for d in DEV3:
        p = REPO / f"outputs/eacl_phase0/{d}_michelmann_master.jsonl"
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r.get("doc_id", d), []).append(r)
        for did, rs in by.items():
            rs.sort(key=lambda r: r["transition"])
            sents = [rs[0]["previous"]] + [r["current"] for r in rs]
            if len(sents) >= 3:
                out[did] = sents
    return out


def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description="dev-3 spaCy 토큰 태그 덤프 (M9 엔티티 필터용)")
    ap.add_argument("--split", default="dev3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="en_core_web_sm")
    args = ap.parse_args()

    import spacy

    nlp = spacy.load(args.model)                      # extract_axis_features.py:178-182 와 동일
    docs = load_sents(args.split)
    payload = {"split": args.split, "spacy": spacy.__version__, "model": args.model, "docs": {}}
    for did, sents in docs.items():
        tags = []
        for doc in nlp.pipe(sents, batch_size=256):
            ent_of = {}
            for ent in doc.ents:
                for t in ent:
                    ent_of[t.i] = ent.label_
            tags.append([{"s": t.idx, "e": t.idx + len(t.text), "pos": t.pos_,
                          "ent": ent_of.get(t.i, ""), "lemma": t.lemma_.lower()}
                         for t in doc if not t.is_space])
        payload["docs"][did] = {"sent_sha": [sha(s) for s in sents], "tags": tags}
        n_ent = sum(1 for st in tags for t in st if t["ent"])
        n_tok = sum(len(st) for st in tags)
        print(f"[{did}] {len(sents)}문장 {n_tok}토큰 (NER {n_ent})")

    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"[저장] {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
