#!/usr/bin/env python3
"""CSM 논문 수치 재현 — 그리고 **dialseg711 미재현 건의 원인 진단**.

## 무엇을 진단하는가
`outputs/experiments/2026-08-02_csm_offline_verify/REPORT.md` 가 남긴 미해명 건:

> ✅ tiage 84~105% · superseg 100~112% ⇒ 재현된다.
> 🔴 dialseg711 은 최대 **76.6%** 로 재현되지 않는다. … 체크포인트가 아니라
>    **데이터·전처리 계보 차이**로 보인다. **미해명.**

그 측정은 **하네스 B**(Def-DTS 번들 + 우리 segeval)에서 이뤄졌다. 그런데 TextTiling 을
하네스 C 로 다시 재보니 **같은 방법인데 하네스만 바꿔도 tiage Score 가 0.3081 → 0.3628 로
움직였다**(`../texttiling/REPRO.md` §2). 즉 B 와 C 는 무시할 수 없게 다르다.

⇒ **가설: dialseg711 격차의 상당 부분이 하네스다.** 이 스크립트는 **같은 체크포인트**를
하네스 C 에 태워 그 가설을 검정한다. 체크포인트를 고정했으므로 차이가 나면 그건 하네스다.

## 🔴 한계 — 저자 체크포인트가 없다
저자 스크립트(`scripts/csm.sh`)는 `.cache/model_zoo/csm/csm-dailydial.pkl` 을 쓰는데
**그 파일은 배포처가 없다**(README 는 데이터 Drive 만 링크한다). 5월에 있던
`methods/CSM/cpt_277000.pth` 도 소실됐다(ledger D-B32 계열).
우리가 가진 것은 **DailyDialog 로 자체 재학습한 240개 체크포인트**뿐이다.

따라서 이 실행은 **"저자 체크포인트의 재현"이 아니라 "우리 체크포인트를 논문 조건에서 채점"**
이다. 절대값이 논문과 벌어져도 그것이 곧 미재현은 아니다. **판정의 초점은
`Δ(하네스 C − 하네스 B)`** 이고, 그것만이 체크포인트와 독립이다.

    PYTHONPATH=benchmarks/superdialseg/src .venv-superdialseg/bin/python \\
      EACL/methods/CSM/run_paper_repro.py --ckpt external/Dialogue-Topic-Segmenter/checkpoints/cpt_175335.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

METHODS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHODS))

from common.superdialseg_harness import (DATASETS, METRICS, PAPER_TABLE3,  # noqa: E402
                                         evaluate, print_comparison)

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/experiments/2026-08-03_csm_harness_diagnosis"

# 하네스 B 실측치 (2026-08-02_csm_offline_verify/REPORT.md, ckpt 175335 행)
HARNESS_B = {"tiage": 0.5353, "dialseg711": 0.5057, "superseg": 0.4647}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="CoherenceScoringModel 호환 state_dict (.pth)")
    ap.add_argument("--backbone", default="bert-base-uncased", help="저자 기본값")
    ap.add_argument("--max-utterance-len", type=int, default=50, help="저자 기본값")
    ap.add_argument("--cut-rate", type=float, default=1.0,
                    help="α. 2026-08-02 실행과 동일하게 1.0 (CSMSegmenter 기본값)")
    ap.add_argument("--batch-size", type=int, default=1, help="저자 csm.sh 기본값")
    ap.add_argument("--num-workers", type=int, default=4, help="저자 csm.sh 기본값")
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()

    import importlib.util

    import torch

    from super_dialseg.models.csm.modeling_csm import CSMSegmenter

    # 🔴 우리 체크포인트는 lxing532 `CoherenceNet` 산(産)이라 SuperDialseg
    # `CoherenceScoringModel` 과 **키 이름이 다르다**(가중치 shape 는 같다):
    #     bert.<rest>                 → bert.bert.<rest>
    #     coherence_decoder.<n>       → coherence_prediction_decoder.<n>
    # 그대로 로드하면 RuntimeError. 기존 러너의 검증된 remap 을 재사용한다
    # (중복 구현하면 드리프트 위험). NSP head(`bert.cls.*`)·pooler 는 우리 ckpt 에 없지만
    # `CoherenceScoringModel.forward` 가 **hidden_states[-1][:,0,:] 만** 쓰므로 추론에 무관하다.
    spec = importlib.util.spec_from_file_location(
        "_csm_offline_runner", REPO / "methods/CSM/offline/whole_dialogue.py")
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    load_lxing532_ckpt = _mod._load_lxing532_ckpt

    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    if not ckpt.exists():
        print(f"🔴 체크포인트 없음: {ckpt}")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"체크포인트 = {ckpt.name} · backbone={args.backbone} · "
          f"max_utterance_len={args.max_utterance_len} · cut_rate={args.cut_rate} · device={device}")

    rows = {}
    for ds in args.datasets.split(","):
        # 데이터셋마다 새 segmenter — evaluation 누적기가 초기화돼야 한다
        seg = CSMSegmenter(backbone=args.backbone, max_utterance_len=args.max_utterance_len,
                           cut_rate=args.cut_rate)
        load_lxing532_ckpt(seg, str(ckpt))
        seg.model.eval()
        seg.to(device)
        rows[ds] = evaluate(seg, ds, batch_size=args.batch_size, num_workers=args.num_workers)

    verdicts = print_comparison("csm", rows, args.tolerance)

    # ---- 핵심 진단: 하네스 B vs C (체크포인트 고정) ----
    print("\n" + "=" * 78)
    print("🔑 하네스 진단 — 같은 체크포인트, B(Def-DTS+segeval) vs C(공식 데이터+저자 채점기)")
    print("=" * 78)
    print(f"{'데이터셋':<12}{'하네스 B':>10}{'하네스 C':>10}{'Δ(C−B)':>10}{'논문':>9}{'C/논문':>9}")
    diag = {}
    for ds in rows:
        b = HARNESS_B.get(ds)
        c = rows[ds]["total_score"]
        p = PAPER_TABLE3["csm"][ds]["total_score"]
        diag[ds] = {"harness_B": b, "harness_C": c, "delta": (c - b) if b else None,
                    "paper": p, "pct_of_paper": 100 * c / p}
        db = f"{c - b:+.4f}" if b else "  —"
        print(f"{ds:<12}{(b if b else 0):>10.4f}{c:>10.4f}{db:>10}{p:>9.3f}{100*c/p:>8.1f}%")

    print("\n해석 지침: Δ 가 크면 이전 '미재현' 판정의 원인은 방법이 아니라 하네스다.")
    print("           Δ 가 작은데 논문과 벌어지면 남는 후보는 체크포인트(저자 것 부재)다.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(
        {"ckpt": str(ckpt), "config": vars(args), "ours_harnessC": rows,
         "paper": PAPER_TABLE3["csm"], "harness_B_reference": HARNESS_B,
         "diagnosis": diag, "verdicts": verdicts}, ensure_ascii=False, indent=2))
    print(f"\n저장: {OUT / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
