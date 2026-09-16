#!/usr/bin/env python3
"""push 기반 스트리밍 분절기를 **서사 6편**에 돌려 예측을 덤프한다 (HP 오버라이드 가능).

`baseline_compare.preds_baseline` 이 하던 일을 떼어낸 것. 그쪽은 import 만으로 torch·sklearn·
학습 코드를 전부 끌어오고 HP 가 `FROZEN` 에 하드코딩돼 있어 **변형을 돌릴 수 없다.**
여기서는 같은 구동 규약(`push` 만 호출, `flush` 안 부름)을 유지하면서 HP 를 열어 둔다.

경계 인덱스 규약도 그대로다: 분절기는 "새 세그먼트를 여는 발화 번호"(1-based)를 주고,
전이 j = (문장 j → 문장 j+1) 이므로 발화 b 는 전이 b-1 에 대응한다.

    env -C /tmp CUDA_VISIBLE_DEVICES=0 /root/Hi-OnTop/.venv/bin/python \\
        EACL/methods/common/run_streaming_nes.py --method CSM --set delay=0 \\
        --label CSM-online-zero
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))

PRED = REPO / "outputs/experiments/2026-07-29_baseline_compare/preds"
OUT = REPO / "outputs/experiments/2026-08-04_online_baselines_close"


def build(method: str, device: str, over: dict):
    """동결 HP 를 baseline_compare 에서 가져와 `over` 로 덮어쓴다."""
    from baseline_compare import FROZEN, CSM_BACKBONE, csm_ckpt  # noqa: PLC0415

    hp = dict(FROZEN["narrative"][method.split("-")[0]])
    hp.update(over)
    if method.startswith("CSM"):
        from hi_ontop.baselines.csm_online import CSMOnlineDelay2 as C
        ck = csm_ckpt(None)
        return lambda: C(device=device, backbone=CSM_BACKBONE, ckpt_path=ck, **hp), hp
    if method == "TextTiling":
        from hi_ontop.baselines.texttiling_streaming import StreamingTextTiling as C
        return lambda: C(**hp), hp
    raise ValueError(f"지원하지 않는 method: {method}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--label", required=True, help="예측 덤프 이름 {label}_{ds}.json")
    ap.add_argument("--set", action="append", default=[],
                    help="HP 오버라이드 k=v (반복 가능). 값은 int/float/str 순으로 파싱")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    over = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        for cast in (int, float):
            try:
                over[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            over[k] = v

    from baseline_compare import load_dataset                # noqa: PLC0415
    from hi_ontop.dts_scoring import score_dialogues         # noqa: PLC0415

    mk, hp = build(args.method, args.device, over)
    print(f"{args.label} · {args.method} · HP={hp}", flush=True)

    rows = {}
    PRED.mkdir(parents=True, exist_ok=True)
    for ds in ("dev3", "test3"):
        data = load_dataset(ds)
        preds, delays = {}, []
        for did, sents, gold in data:
            seg = mk()
            v = np.zeros(len(gold), np.int8)
            for t, u in enumerate(sents, start=1):
                for b in seg.push(u):          # flush 는 부르지 않는다 (채점 경로와 동일)
                    delays.append(t - int(b))
                    if 1 <= int(b) <= len(v):
                        v[int(b) - 1] = 1
            preds[did] = v.tolist()
            print(f"  {did:<16} 문장 {len(sents):>4} · 예측 {int(v.sum()):>4}", flush=True)
        G = [g for _d, _s, g in data]
        ys = [np.concatenate([np.asarray(g, np.int8), [0]]).tolist() for g in G]
        ps = [np.concatenate([np.asarray(preds[d], np.int8), [0]]).tolist()
              for d, _s, _g in data]
        sc = score_dialogues(ys, ps)
        mx = max(delays) if delays else None
        rows[ds] = {k: sc[k] for k in ("pk", "wd", "f1", "score")} | {
            "n_pred": int(sum(sum(v) for v in preds.values())),
            "n_gold": int(sum(int(g.sum()) for g in G)),
            "max_delay": mx}
        (PRED / f"{args.label}_{ds}.json").write_text(json.dumps(preds))
        print(f"  {args.label} {ds:<6} Pk {sc['pk']:.4f} WD {sc['wd']:.4f} F1 {sc['f1']:.4f} "
              f"Score {sc['score']:.4f} · 실측 max delay {mx}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"streaming_{args.label}.json").write_text(json.dumps(
        {"method": args.method, "hp": hp, "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
