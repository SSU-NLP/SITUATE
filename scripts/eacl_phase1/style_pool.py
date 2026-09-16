#!/usr/bin/env python3
"""문체 다양성 측정용 풀 재구축 — 편별 기능어 프로파일 + 서술 시점.

🔴 왜 다시 만드나: 이전 풀은 편 길이·문장 길이·TTR 만 저장했는데 그건 **길이 통계이지
   문체가 아니다**. 문체의 표준 조작화는 **기능어 상대빈도**(the/of/and/but/which…)다 —
   내용어와 달리 주제에 좌우되지 않아 저자·장르 판별의 고전적 신호로 쓰인다.

산출: 편마다 (기능어 상대빈도 벡터, 서술 시점, 대사 비율, 길이 통계).
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/eacl_phase0"), str(REPO / "scripts/eacl_phase1")]
from build_narrative_pack import split_sentences  # noqa: E402
from build_pg_shorts import split_volume, stage0_piece  # noqa: E402
from fetch_repcorpus import density, register  # noqa: E402

CACHE = REPO / "data/eacl_phase1/pg_cache"
CAT = REPO / "data/eacl_phase1/pgcat.csv"
OUT = REPO / "outputs/experiments/2026-08-02_author_cap"
MIN_SENT, MAX_SENT, MIN_MED = 30, 900, 5.0
EXCL = ["Pohl, Frederik"]

# 기능어 70개 — 관사·전치사·접속사·대명사·조동사·한정사. 내용어 없음(주제 비의존).
FW = ("the a an and or but if while although though because since unless whereas "
      "of in on at by for with about against between into through during before after "
      "above below to from up down out off over under again further then once "
      "i me my we our you your he him his she her it its they them their "
      "is are was were be been being have has had do does did "
      "not no nor very so too such that this these those what which who whom").split()
FW_IDX = {w: i for i, w in enumerate(FW)}


def main() -> None:
    meta = {int(r["Text#"]): r for r in csv.DictReader(CAT.open(encoding="utf-8"))}
    pool, seen = [], set()
    files = sorted(CACHE.glob("*.txt"), key=lambda f: int(f.stem))
    for i, f in enumerate(files):
        m = meta.get(int(f.stem))
        if m is None:
            continue
        au = m.get("Authors", "")
        if any(a in au for a in EXCL):
            continue
        pieces, reason = split_volume(f.read_text(encoding="utf-8", errors="ignore"))
        if reason == "chaptered_work":
            continue
        for title, body in pieces:
            key = (au, title.upper())
            if title and key in seen:
                continue
            sents, _ = stage0_piece(split_sentences(body))
            if not (MIN_SENT <= len(sents) <= MAX_SENT):
                continue
            wl = [len(x.split()) for x in sents]
            if statistics.median(wl) < MIN_MED:
                continue
            seen.add(key)
            toks = re.findall(r"[a-z']+", " ".join(sents).lower())
            vec = np.zeros(len(FW))
            for t in toks:
                j = FW_IDX.get(t)
                if j is not None:
                    vec[j] += 1
            n_fw = vec.sum()
            loc = (m.get("LoCC") or "").strip()
            pool.append({
                "author": au, "gid": int(f.stem),
                "locc1": loc.split(",")[0].strip()[:2] if loc else "NA",
                "n_sent": len(sents), "sent_words": float(np.median(wl)),
                "dialogue_density": density(sents), "register": register(sents)[0],
                "fw_rate": float(n_fw / max(len(toks), 1)),          # 기능어 비율 자체도 문체 신호
                "fw": (vec / n_fw).round(5).tolist() if n_fw else [0.0] * len(FW),
            })
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)}권 · 누적 {len(pool):,}편", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pool_style.json").write_text(json.dumps(pool), encoding="utf-8")
    print(f"[완료] {len(pool):,}편 · 기능어 {len(FW)}개 -> {OUT/'pool_style.json'}")


if __name__ == "__main__":
    main()
