#!/usr/bin/env python3
"""pg_shorts.jsonl (편 단위) → llm_s5_states 가 먹는 master 형식으로 변환.

master 규약(기존 Phase-0 과 동일): 한 행 = 한 전이.
  {doc_id, transition(1-based), previous, current, boundary, idm_score}

⚠️ **boundary = 0 고정.** 이 코퍼스는 **학습 앵커 공급용**이지 평가셋이 아니다. 편 내부 장면경계(B)
gold 가 없으므로 0 으로 채우고, Score 계산에 쓰지 않는다(gold 밀도 가드가 막는다). 편 *사이* 경계는
A 경계라 애초에 타깃이 아니며(D-B4), 편을 분리 문서로 두어 구조적으로 배제된다.

--shard 로 카나리아/부분 실행 가능. 메타(작가·연도·주제·대사밀도)는 sidecar 로 보존한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", type=Path, default=REPO / "data/eacl_phase1/pg_shorts.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta-out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N 편만 (카나리아용)")
    ap.add_argument("--offset", type=int, default=0)
    # Stage-1 은 편 전체를 한 프롬프트에 넣는다. 서버 max-model-len 16384 에서 출력 예약분을 빼면
    # 입력 예산이 남는다. 초과 편은 400(BadRequestError)으로 런 전체를 죽인다 — ami 에서 실측
    # (입력 8385 + 출력 8000 = 16385 > 16384). 동결 스크립트를 고치지 않고 여기서 걸러 기록한다.
    ap.add_argument("--ctx", type=int, default=16384, help="서버 max-model-len")
    ap.add_argument("--stage1-out", type=int, default=2000, help="Stage-1 출력 예약(=--stage1-max-tokens)")
    ap.add_argument("--tok-per-word", type=float, default=1.45, help="보수적 토큰 환산")
    a = ap.parse_args()

    rows = [json.loads(x) for x in a.pieces.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = rows[a.offset:]
    if a.limit:
        rows = rows[:a.limit]

    budget = a.ctx - a.stage1_out - 512      # 512 = 프롬프트 지시문·여유
    kept, over = [], []
    for r in rows:
        est = sum(len(s.split()) for s in r["sentences"]) * a.tok_per_word
        (over if est > budget else kept).append((r, est))
    if over:
        print(f"⚠️ Stage-1 컨텍스트 초과로 제외 {len(over)}/{len(rows)} 편 "
              f"(입력 예산 {budget:,} tok). 최대 {max(e for _, e in over):,.0f} tok")
        print("   " + ", ".join(f"{r['doc_id']}({r['n_sent']}문장)" for r, _ in over[:8]))
    rows = [r for r, _ in kept]

    out, meta = [], []
    for r in rows:
        s = r["sentences"]
        for i in range(1, len(s)):
            out.append({"doc_id": r["doc_id"], "transition": i,
                        "previous": s[i - 1], "current": s[i],
                        "boundary": 0, "idm_score": None})
        meta.append({k: v for k, v in r.items() if k != "sentences"})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
                     encoding="utf-8")
    mp = a.meta_out or a.out.with_suffix(".meta.jsonl")
    mp.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta),
                  encoding="utf-8")
    print(f"{len(rows)} 편 · {len(out)} 전이 · 문장 {sum(r['n_sent'] for r in rows)}")
    print(f"  -> {a.out}\n  -> {mp} (메타 sidecar)")


if __name__ == "__main__":
    main()
