#!/usr/bin/env python3
"""서사 6편(dev3+test3)에 **온라인 베이스라인**을 돌려 예측 덤프 + 4지표 표를 낸다.

지표 4종 = Pk · WindowDiff · F1 · Score. 채점 = deployable.
방법 이름에 lookahead 등급을 박는다: `-online`(zero) / `-online-d{d}`(bounded).

    .venv/bin/python EACL/methods/common/run_narrative_online.py --methods texttiling,texttiling-stream
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

METHODS = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(METHODS))

from common.narrative_online import (DATASETS, PRED, dump, load_all,  # noqa: E402
                                     online_flags, score4,
                                     streaming_texttiling_flags, texttiling_offline_fn)

OUT = REPO / "outputs/experiments/2026-08-03_narrative_online"


def run_texttiling(data, w: int, k: int) -> tuple[dict, int]:
    fn = texttiling_offline_fn(w, k)
    preds, miss_all = {}, 0
    for doc, sents, _g in data:
        p, miss = online_flags(fn, sents, delay=0)
        preds[doc], miss_all = p, miss_all + miss
    return preds, miss_all


def run_texttiling_stream(data, w: int, k: int) -> tuple[dict, int]:
    return {doc: streaming_texttiling_flags(sents, w=w, k=k)
            for doc, sents, _g in data}, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="texttiling,texttiling-stream")
    ap.add_argument("-w", type=int, default=10, help="논문 기본값")
    ap.add_argument("-k", type=int, default=6, help="논문 기본값")
    ap.add_argument("--tag", default="", help="조건을 이름에 박을 때 (예: _w5k3)")
    args = ap.parse_args()

    all_data = load_all()
    rows = {}
    for m in args.methods.split(","):
        m = m.strip()
        for ds in DATASETS:
            data = all_data[ds]
            t0 = time.perf_counter()
            if m == "texttiling":
                name, (preds, miss) = f"TextTiling-online{args.tag}", run_texttiling(data, args.w, args.k)
            elif m == "texttiling-stream":
                name, (preds, miss) = f"TextTiling-online-stream{args.tag}", \
                    run_texttiling_stream(data, args.w, args.k)
            else:
                print(f"🔴 미지원 방법: {m}")
                continue
            dt = time.perf_counter() - t0
            dump(name, ds, preds)
            s = score4(preds, data)
            npred = sum(sum(v) for v in preds.values())
            ngold = int(sum(int(g.sum()) for _d, _s, g in data))
            rows.setdefault(name, {})[ds] = {**s, "n_pred": npred, "n_gold": ngold,
                                             "miss": miss, "sec": round(dt, 1)}
            print(f"  {name:<34} {ds:<6} Pk {s['pk']:.4f} WD {s['wd']:.4f} "
                  f"F1 {s['f1']:.4f} Score {s['score']:.4f} · 예측 {npred}/gold {ngold} "
                  f"· miss {miss} · {dt:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"online4{args.tag}.json"
    prev = json.loads(f.read_text()) if f.exists() else {}
    prev.update(rows)
    f.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    print(f"\n저장: {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
