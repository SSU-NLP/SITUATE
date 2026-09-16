#!/usr/bin/env python3
"""M5 계열 체크포인트 **전수**를 한 무대에서 채점 (기본 = 봉인 시험지 `test3`).

🔒 **사전 선언 (실행 전 기록, D-B62).**
이 표는 **탐색적 부록**이다. 어떤 변형이 높게 나오든 `EACL/main_result.md` 의 우리 행은
**동결 기본 진입점 `configs/m5_default.json` = `models/m5_default/M5.pt` + 적응 임계 c=1.0**
으로 고정한다. 시험지 점수를 보고 변형을 고르면 D-M5 의 selection-on-test 오염을 그대로
되풀이하는 것이다. **여기서 고르지 않는다.**

두 작동점을 모두 낸다 (둘 다 gold 경계 개수를 쓰지 않는다 = deployable):
  - `quantile` : 🔒 앵커 동결 고정분위수 `ANCHOR_R`(D-B30). ⛔ **폐기된 작동점**(D-B61 §1) — 대조용
  - `adaptive`: 적응 임계 `s_t >= mean(s_<t) + 1.0*std(s_<t)` (online_nf) = 현행 규칙

    .venv/bin/python scripts/eacl_phase0/m5_family_eval.py --dataset test3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from baseline_compare import ANCHOR_R, OUT, load_dataset, score_A          # noqa: E402
from full_comparison_dev3 import lift_l1, purity_coverage                  # noqa: E402
from hi_ontop.factored_slot import FactoredSlot                            # noqa: E402
from train_anchor_m4m5 import EXP, Corpus, axis_probs                      # noqa: E402


def build(ck: dict, dev: str) -> FactoredSlot:
    """저장된 `args` 그대로 복원한다 — 변형마다 구조가 다르므로 부분 인자 복원은 금지."""
    a = ck["args"]
    none = lambda v: None if v in (None, "none", 0) else v                 # noqa: E731
    dflt = lambda k, d: d if a.get(k) is None else a[k]                    # noqa: E731
    # 🔴 옛 체크포인트는 args 에 키 자체가 없거나 None 이다(그 시절엔 플래그가 없었다).
    #    그때는 **생성자 기본값**이 쓰였으므로 None 을 그대로 넘기면 구조가 달라져 로드가 깨진다.
    m = FactoredSlot(
        slot_dim=a["slot_dim"], mono=ck["mono"], pos_weight=ck["pos_weight"],
        alpha=a.get("_alpha", (1.0, 0.5, 1.0, 0.4, 0.25)),
        d1=(True if a.get("input_path") is None else a["input_path"] == "axis_extractor"),
        shared_dim=none(a.get("shared_proj_dim")),
        tagger=none(a.get("tagger")), tagger_axes=dflt("tagger_axes", 5),
        block_readout=bool(a.get("block_readout")),
        dynamics=ck.get("dynamics"), pred_weight=dflt("pred_weight", 1.0),
        pred_space=dflt("pred_space", "obs"), head_feat=dflt("head_feat", "delta"),
        slot_reset=none(a.get("slot_reset")),
        slot_reset_dropout=dflt("slot_reset_dropout", 0.3),
        psi=none(a.get("psi")), psi_beta=dflt("psi_beta", 0.9),
        psi_dropout=dflt("psi_dropout", 0.3),
        bottleneck=dflt("bottleneck", 8), dropout=dflt("dropout", 0.4)).to(dev)
    m.load_state_dict(ck["state"])
    return m.eval()


def cuts(signals: list[np.ndarray], mode: str) -> list[np.ndarray]:
    if mode == "quantile":
        return [(s >= np.quantile(s, 1 - ANCHOR_R)).astype(np.int8) for s in signals]
    from hi_ontop.hi_ontop_v2 import adaptive_boundaries                   # noqa: PLC0415
    out = []
    for s in signals:
        b = adaptive_boundaries(np.concatenate([[0.0], s]).tolist(), c=1.0, mode="online_nf")
        out.append(np.asarray(b, np.int8)[: len(s)])
    return out


def metrics(data, P: list[np.ndarray]) -> dict:
    pr = {data[i][0]: P[i] for i in range(len(data))}
    s = score_A(data, pr)
    G = [g for _, _, g in data]
    pur, cov = purity_coverage(P, G)
    n_gold = sum(int(g.sum()) for g in G)
    n_pred = int(sum(int(p.sum()) for p in P))
    return {"score": s["score"], "pk": s["pk"], "wd": s["wd"], "f1": s["f1"],
            "n_pred": n_pred, "bor": n_pred / max(n_gold, 1), "purity": pur, "coverage": cov,
            "f_pc": 2 * pur * cov / (pur + cov) if (pur + cov) else 0.0, "L1": lift_l1(P, G)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="test3")
    ap.add_argument("--glob", default="M5*.pt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_dataset(args.dataset)
    corp = Corpus(args.dataset)
    assert corp.n_docs == len(data), f"문서 수 불일치 {corp.n_docs} vs {len(data)}"

    cks = sorted(EXP.glob(args.glob))
    cks.append(REPO / "models/m5_default/M5.pt")             # 동결 기본 진입점(같은 조건에서 재확인)
    print(f"[{args.dataset}] 문서 {len(data)} · gold "
          f"{sum(int(g.sum()) for _, _, g in data)} · 체크포인트 {len(cks)}개", flush=True)

    rows, failed = [], []
    for p in cks:
        name = p.stem if p.parent != REPO / "models/m5_default" else "m5_default(M5.pt)"
        try:
            ck = torch.load(p, map_location=dev, weights_only=False)
            if ck["args"].get("tagger", "none") not in (None, "none") and corp.feat is None:
                raise RuntimeError(f"tagger 변형인데 {args.dataset}_axisfeat.npy 가 없다 "
                                   "(extract_axis_features.py 먼저)")
            m = build(ck, dev)
            sig = [P.sum(1) for P in axis_probs(m, corp, list(range(corp.n_docs)), dev,
                                                bs=4, with_anchor=False)]
            r = {"ckpt": name, "split": ck["args"].get("split", "?")}
            for mode in ("quantile", "adaptive"):
                r[mode] = metrics(data, cuts(sig, mode))
            rows.append(r)
            print(f"  {name:64s} q {r['quantile']['score']:.4f} · a {r['adaptive']['score']:.4f}",
                  flush=True)
        except Exception as e:                               # 실패는 표에 그대로 남긴다
            failed.append({"ckpt": name, "error": f"{type(e).__name__}: {e}"})
            print(f"  🔴 {name:62s} {type(e).__name__}: {e}", flush=True)

    rp = OUT / f"m5_family_{args.dataset}.json"
    rp.write_text(json.dumps({"dataset": args.dataset, "rows": rows, "failed": failed},
                             ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n성공 {len(rows)} · 실패 {len(failed)}\n-> {rp}")


if __name__ == "__main__":
    main()
