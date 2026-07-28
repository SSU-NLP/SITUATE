#!/usr/bin/env python3
"""Build an S5 master from SHARE (movie-script long-term dialogue).

SHARE 구조: 최상위 dict[화자쌍] -> {movie, dialogue:[세션...]}, 세션 =
{session, dialogues:[{speaker,text,label,utterance}], <화자>'s persona, <화자>'s
temporary event}. label/persona/temporary event = **LLM 가공 메타데이터**.

타깃 실험 (사용자 제안 2026-07-14): 메타데이터(persona/event/label/session 마커)를
**전부 제거**하고 **순수 대사 스트림**만 입력해, 각본가가 저작한 **씬(세션) 전환**을
잡을 수 있는지 본다 = "서술문 없이 대화만으로 상황전환 감지" 하드 조건.

- 입력 단위 = 한 발화(utterance). previous/current = 발화 텍스트만(speaker 별도 필드).
- gold 경계 = 세션 전환(end-turn 규약): 세션 k의 마지막 발화가 segment-final,
  세션 k+1의 첫 발화가 새 세그먼트 시작. 즉 transition s에서 session(s-1)!=session(s) → boundary=1.
- doc_id = 화자쌍(대화) 1개. multi-doc master (322 docs/valid → bootstrap CI 확보).

⚠️ 층위 1 caveat: 세션 경계는 시나리오 구조에서 온 메타데이터(무료 gold)다. 각본가
저작 상황전환이라는 점에서 정당하나, 화자쌍이 2인 고정이라 protagonist 축은 구조적
으로 死할 것으로 예상. ledger에 기록.

usage: uv run python scripts/eacl_phase0/build_share_pack.py \
         --split valid --out outputs/eacl_phase0/share_valid_master.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--src", type=Path,
                    default=REPO / "benchmarks/SHARE/datasets")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-turns", type=int, default=4,
                    help="이 미만 발화 수의 대화는 제외")
    ap.add_argument("--sample-n", type=int, default=0,
                    help="0이면 전체. >0이면 조건 통과 doc에서 seed 샘플 N개(probe용).")
    ap.add_argument("--sample-min-turns", type=int, default=15,
                    help="--sample-n>0일 때, 이 이상 발화 수 doc에서만 샘플(degenerate score 회피).")
    ap.add_argument("--seed", type=int, default=20260714)
    args = ap.parse_args()
    out = args.out or REPO / f"outputs/eacl_phase0/share_{args.split}_master.jsonl"

    data = json.load(open(args.src / f"{args.split}.json"))
    doc_rows: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for pair_key, conv in data.items():
        movie = conv.get("movie", "unknown")
        sessions = conv["dialogue"]
        # flatten: (speaker, text, session_id) preserving order
        flat: list[tuple[str, str, int]] = []
        for s in sessions:
            sid = s["session"]
            for t in s["dialogues"]:
                txt = (t.get("text") or "").strip()
                if not txt:
                    continue
                flat.append((t.get("speaker", "").strip(), txt, sid))
        if len(flat) < args.min_turns:
            continue
        n_bnd = sum(1 for i in range(1, len(flat)) if flat[i][2] != flat[i - 1][2])
        if n_bnd == 0:
            continue
        # doc_id: movie + speaker-pair, 충돌 방지 suffix
        pair = re.sub(r"[()',]", "", pair_key)
        doc_id = f"{slug(movie)}__{slug(pair)}"
        if doc_id in seen:
            k = 2
            while f"{doc_id}_{k}" in seen:
                k += 1
            doc_id = f"{doc_id}_{k}"
        seen.add(doc_id)
        doc_rows[doc_id] = [{
            "doc_id": doc_id, "transition": i,
            "previous": flat[i - 1][1], "current": flat[i][1],
            "prev_speaker": flat[i - 1][0], "speaker": flat[i][0],
            "previous_ko_reference": "", "current_ko_reference": "",
            "boundary": int(flat[i][2] != flat[i - 1][2]),
            "prev_session": flat[i - 1][2], "cur_session": flat[i][2],
            "idm_score": None,
            "protagonist": None, "place": None, "time": None,
            "causality": None, "goal": None, "notes": "",
        } for i in range(1, len(flat))]

    # optional seeded subset (probe)
    if args.sample_n > 0:
        elig = sorted(d for d, rs in doc_rows.items()
                      if len(rs) + 1 >= args.sample_min_turns)
        chosen = set(random.Random(args.seed).sample(elig, min(args.sample_n, len(elig))))
        doc_rows = {d: rs for d, rs in doc_rows.items() if d in chosen}

    rows = [r for d in sorted(doc_rows) for r in doc_rows[d]]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                   encoding="utf-8")
    nb = sum(r["boundary"] for r in rows)
    tag = f" (sample-n={args.sample_n}, seed={args.seed})" if args.sample_n else ""
    print(f"SHARE {args.split}{tag}: {len(doc_rows)} docs, {len(rows)} transitions, "
          f"{nb} boundaries ({100*nb/max(1,len(rows)):.1f}% density) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
