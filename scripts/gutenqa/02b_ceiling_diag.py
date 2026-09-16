#!/usr/bin/env python3
"""ceiling 진단 — 이 벤치마크에서 **원천적으로 못 맞히는 질문**이 왜 생기는가 (GPU 불필요).

## 왜 필요한가
`02_eval_harness.py` 실행 중 lumberchunker 와 recursive 의 ceiling 이 **소수 넷째 자리까지 동일한
0.8247** 로 나왔다. 청크 크기가 327 단어 vs 397 단어로 다른데 상한이 같다는 건 **손실의 원인이 청크
절단이 아니라는 뜻**이다. 이 스크립트가 그 원인을 특정한다.

## 결론 (실측)
| 방법 | 단어/청크 | ceiling(원본) | ceiling(정규화) |
|---|---|---|---|
| paragraph | 48.8 | 82.47% | 86.10% |
| semantic | 63.1 | 82.10% | 86.10% |
| lumberchunker | 327.3 | 82.47% | 88.40% |
| recursive | 397.5 | 82.47% | 88.40% |
| **propositions** | 12.2 | **7.03%** | 7.17% |

1. **산문 4종은 청크 크기가 8배 차이나도 ceiling 이 82% 대로 같다.** ⇒ 17.5% 손실은 청킹 탓이
   아니라 **데이터셋 자체의 표기 불일치**다. 질문을 만든 LLM 이 gold 를 옮겨 적으면서 작은따옴표
   `’`→`'`, 줄표 `—`→`-` 등을 바꿨다. 실측: 미발견 526건 중 원문에 그대로 있는 건 57건뿐이고,
   공백만 정규화하면 172건까지 회복된다. **방법 무관이라 비교 공정성은 깨지 않지만**, 전 방법의
   점수가 같은 비율로 눌려 있다는 뜻이므로 절대값 해석 시 반드시 알아야 한다.
2. 🔴 **propositions 는 이 지표로 측정 불가다.** 명제는 LLM 이 원문을 **다시 쓴** 문장이라 원문
   부분문자열을 거의 포함하지 않는다(ceiling 7%). 명제를 개별로 펴든(929k) 레포처럼 청크 단위로
   묶든(36.9k) 7% 로 같다 — 구성 방식 문제가 아니다. 논문의 Propositions DCG@20 = 46.19 는 이
   판정 기준으로는 재현 불가능하며(상한이 7%), 논문이 명제에 대해 다른 gold 판정을 썼다고 봐야 한다.
   ⇒ **우리 표에서 propositions 행은 "측정 불가"로 표시하고 순위 비교에서 뺀다.**

정규화 ceiling 은 **진단일 뿐 주지표가 아니다.** 주지표는 레포 코드 그대로(원본 부분문자열)를 쓴다
(`baselines-stay-as-published`).

    uv run python scripts/gutenqa/02b_ceiling_diag.py
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"
DATA = BASE / "data"

_QUOTES = {"’": "'", "‘": "'", "“": '"', "”": '"',
           "—": "-", "–": "-", "…": "..."}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for a, b in _QUOTES.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip().lower()


def main() -> None:
    qs = [json.loads(l) for l in (DATA / "questions.jsonl").open(encoding="utf-8") if l.strip()]
    books = {r["book_id"]: r["paragraphs"] for r in
             (json.loads(l) for l in (DATA / "books.jsonl").open(encoding="utf-8") if l.strip())}

    rows = []
    for p in sorted((DATA / "chunks").glob("*.jsonl")):
        m = p.stem
        ch = {r["book_id"]: r["chunks"] for r in
              (json.loads(l) for l in p.open(encoding="utf-8") if l.strip())}
        if not ch:
            continue
        n = sum(len(v) for v in ch.values())
        w = sum(len(c.split()) for v in ch.values() for c in v)
        low = {b: [c.lower() for c in v] for b, v in ch.items()}
        nrm = {b: [norm(c) for c in v] for b, v in ch.items()}
        raw = sum(1 for x in qs if any(x["gold"].lower() in c for c in low.get(x["book_id"], [])))
        nor = sum(1 for x in qs if any(norm(x["gold"]) in c for c in nrm.get(x["book_id"], [])))
        rows.append({"method": m, "chunks_per_book": round(n / len(ch), 1),
                     "words_per_chunk": round(w / max(1, n), 1),
                     "ceiling_raw": raw / len(qs), "ceiling_norm": nor / len(qs)})

    # 손실의 원인 분해 — lumberchunker 기준
    lc = {r["book_id"]: [c.lower() for c in r["chunks"]] for r in
          (json.loads(l) for l in (DATA / "chunks/lumberchunker.jsonl").open(encoding="utf-8") if l.strip())}
    miss = [x for x in qs if not any(x["gold"].lower() in c for c in lc[x["book_id"]])]
    full_raw = {b: " ".join(p).lower() for b, p in books.items()}
    full_nrm = {b: norm(" ".join(p)) for b, p in books.items()}
    cause = {
        "gold_not_in_any_chunk": len(miss),
        "of_which_present_in_full_text_verbatim": sum(1 for x in miss if x["gold"].lower() in full_raw[x["book_id"]]),
        "of_which_present_after_normalization": sum(1 for x in miss if norm(x["gold"]) in full_nrm[x["book_id"]]),
    }
    cause["unrecoverable_even_normalized"] = cause["gold_not_in_any_chunk"] - cause["of_which_present_after_normalization"]

    out = {"n_questions": len(qs), "methods": rows, "loss_decomposition_lumberchunker": cause}
    (BASE / "ceiling_diag.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    print(f"{'method':22s} {'청크/책':>9s} {'단어/청크':>10s} {'ceiling(원본)':>14s} {'ceiling(정규화)':>16s}")
    for r in sorted(rows, key=lambda x: x["words_per_chunk"]):
        print(f"{r['method']:22s} {r['chunks_per_book']:9.1f} {r['words_per_chunk']:10.1f} "
              f"{r['ceiling_raw']*100:13.2f}% {r['ceiling_norm']*100:15.2f}%")
    print(f"\n손실 원인 분해 (lumberchunker 기준, 전체 {len(qs)}문항):")
    for k, v in cause.items():
        print(f"  {k:42s} {v:>5d}")
    print(f"\nsaved -> {BASE/'ceiling_diag.json'}")


if __name__ == "__main__":
    main()
