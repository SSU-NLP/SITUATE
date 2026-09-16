#!/usr/bin/env python3
"""CSM 체크포인트별 **실제 분절 성능** 측정 — 학습이 진행될수록 나빠지는가?

## 왜 필요한가
`train.py` 의 검증 지표(`validation_metric`)는 pseudo-triple 의 **쌍 순위 정확도**이지
분절 성능이 아니다. 그리고 그 지표는 3쌍의 평균인데 그 중 하나가 **neg1 vs neg2** 다:

- `neg_1` (minor) = **같은 대화** 안, anchor 와 인접하지 않고 dial-act 가 다른 발화
- `neg_2` (major) = **다른 topic 대화**에서 뽑은 발화

pos > neg1 은 의미가 분명하지만 **neg1 > neg2 는 약한 가정**이다. 모델이 "pos vs 나머지"를
잘하게 되면서 neg1/neg2 구분을 포기하면 그 항이 0.5 근처로 떨어져 **전체 지표가 내려간다** —
분절 성능은 그대로거나 나아지는데도. 그래서 검증 지표 하락만으로 "학습이 해롭다"고 말할 수 없다.

⚠️ 또한 epoch 1 안에서는 88.9만 표본을 **각각 한 번씩만** 본다 — 반복이 없으므로
고전적 과적합은 성립하지 않는다.

## 무엇을 재나
각 체크포인트로 `CSMOnlineDelay2`(표에 실제로 들어갈 그 구현) 를 돌려
**dev split 의 분절 Score 를 하네스 (A)** 로 잰다. test 는 건드리지 않는다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
sys.path.insert(0, str(REPO / "methods/RoBERTa/offline"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

SDS = REPO / "benchmarks/superdialseg_data"
OUT = REPO / "outputs/experiments/2026-07-29_baseline_compare"


def load_dialogs(ds: str, split: str) -> list[tuple[list[str], list[int]]]:
    """SuperDialseg JSON -> [(utterances, labels)]. 라벨 규약 = 세그먼트의 **마지막** turn 이 1."""
    raw = json.loads((SDS / ds / f"segmentation_file_{split}.json").read_text(encoding="utf-8"))
    arr = raw["dial_data"][list(raw["dial_data"])[0]]
    out = []
    for d in arr:
        utts = [t["utterance"] for t in d["turns"]]
        yt = [int(t.get("segmentation_label", 0)) for t in d["turns"]]
        if yt:
            yt[-1] = 0
        if len(utts) >= 3:
            out.append((utts, yt))
    return out


def run_ckpt(ckpt: Path, dialogs, backbone: str, device: str) -> tuple[dict, float]:
    """⚠️ 인스턴스를 대화마다 새로 만들면 **440MB 체크포인트를 매번 다시 로드**한다.
    가중치는 한 번만 올리고 대화 사이에는 **스트리밍 상태만** 초기화한다."""
    from hi_ontop.baselines.csm_online import _Welford, CSMOnlineDelay2
    t0 = time.perf_counter()
    seg = CSMOnlineDelay2(ckpt_path=str(ckpt), backbone=backbone, device=device)
    seg._ensure_loaded()

    def reset() -> None:
        seg._utts, seg._t, seg._scores = [], 0, []
        seg._depths, seg._evaluated_until, seg._emitted = {}, -1, []
        seg._last_boundary_t, seg._welf = -(10 ** 9), _Welford()

    preds = []
    for utts, yt in dialogs:
        reset()
        emitted: set[int] = set()
        for u in utts:
            for b in seg.push(u):
                emitted.add(int(b))
        # baseline 규약: b = 새 세그먼트를 여는 발화 번호 → 그 직전 turn(b-1) 이 세그먼트 끝
        yp = [0] * len(yt)
        for b in emitted:
            if 1 <= b <= len(yt) - 1:
                yp[b - 1] = 1
        preds.append(yp)
    r = score_dialogues([yt for _, yt in dialogs], preds)
    return ({"score": r["score"], "pk": r["pk"], "wd": r["wd"], "f1": r["f1"],
             "n_pred": int(sum(sum(p) for p in preds)),
             "n_gold": int(sum(sum(yt) for _, yt in dialogs))},
            time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=str(REPO / "external/Dialogue-Topic-Segmenter/checkpoints"))
    ap.add_argument("--backbone", default="aws-ai/dse-bert-base")
    ap.add_argument("--dataset", default="superseg")
    ap.add_argument("--split", default="validation", help="🔒 dev 만. test 금지")
    ap.add_argument("--n_dial", type=int, default=150, help="앞 N개(결정적)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", default="", help="쉼표구분 step 목록. 빈값=있는 것 전부")
    args = ap.parse_args()
    assert args.split != "test", "test split 금지 — 체크포인트 선택은 dev 에서만"

    dialogs = load_dialogs(args.dataset, args.split)[: args.n_dial]
    n_turn = sum(len(u) for u, _ in dialogs)
    print(f"[dev] {args.dataset}/{args.split} 앞 {len(dialogs)}대화 · {n_turn} turn · "
          f"gold {sum(sum(y) for _, y in dialogs)}", flush=True)

    cks = sorted(Path(args.ckpt_dir).glob("cpt_*.pth"),
                 key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    if args.steps:
        want = {int(x) for x in args.steps.split(",")}
        cks = [c for c in cks if int(re.search(r"(\d+)", c.name).group(1)) in want]

    # 산출 파일명에 **조건(백본)을 네임스페이스**로 박는다 — dse-bert-base 판(폐기)과 섞이면
    # step 키가 충돌해 조용히 잘못된 곡선이 만들어진다.
    rp = OUT / f"csm_ckpt_sweep_{args.backbone.split('/')[-1]}.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    for c in cks:
        step = int(re.search(r"(\d+)", c.name).group(1))
        if str(step) in res:
            continue
        m, wall = run_ckpt(c, dialogs, args.backbone, args.device)
        res[str(step)] = m
        print(f"  step {step:>6}  Score {m['score']:.4f}  (Pk {m['pk']:.3f} WD {m['wd']:.3f} "
              f"F1 {m['f1']:.3f}, 경계 {m['n_pred']}/{m['n_gold']})  {wall:.0f}s", flush=True)
        rp.write_text(json.dumps(res, indent=2, default=float))
    print(f"\n-> {rp}")


if __name__ == "__main__":
    main()
