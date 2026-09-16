"""epoch 별 성능 곡선 — 10 epoch 스냅숏 + 최종본을 6편에서 채점한다.

val 손실은 로그에 **매 epoch** 남지만, 분절 성능(Score/BOR/Lift/AUC)은 체크포인트가 있어야
계산된다. 지금 스냅숏 간격은 10 이므로 곡선의 x 축도 10 단위다.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
import train_anchor_m4m5 as T  # noqa: E402
from eval_all_models_alldocs import m5_scores  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=str(REPO / "outputs/experiments/m4m5_phase1_anchor"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate", type=float, default=0.0660)
    a = ap.parse_args()

    corpora = {"dev-3": T.Corpus("dev3"), "test3": T.Corpus("test3")}
    curve: dict[str, dict[str, list]] = {}
    for model in ("M5", "M4"):
        pat = f"{a.exp}/{model}_phase1__personkey_i2_per_slot_gsauthor_ep100_i2e100*seed?.pt"
        byep: dict[int, list[dict]] = {}
        for p in sorted(glob.glob(pat)):
            m = re.search(r"_ep(\d{3})_seed", p)
            blob = torch.load(p, map_location="cpu", weights_only=False)
            ep = int(m.group(1)) if m else int(blob.get("val_loss_best_epoch") or
                                               blob.get("epoch") or 999)
            key = ep if m else 999          # 999 = 최종본(조기종료 최적 가중치)
            sc = m5_scores(p, corpora, a.device, a.rate)
            byep.setdefault(key, []).append(sc)
        curve[model] = {}
        for ep, runs in sorted(byep.items()):
            docs = list(runs[0])
            agg = {}
            for k in ("score", "bor", "lift", "auc"):
                per_doc = [st.mean([r[d][k] for r in runs if r[d][k] == r[d][k]]) for d in docs]
                agg[k] = st.mean(per_doc)
                agg[k + "_sd"] = (st.stdev([st.mean([runs[i][d][k] for d in docs])
                                            for i in range(len(runs))]) if len(runs) > 1 else 0.0)
            agg["n_seed"] = len(runs)
            curve[model][str(ep)] = agg
            tag = "final" if ep == 999 else f"ep{ep}"
            print(f"[{model}] {tag:>6s} (seed {len(runs)}) Score {agg['score']:.4f}±{agg['score_sd']:.4f} "
                  f"BOR {agg['bor']:.3f} Lift {agg['lift']:.3f} AUC {agg['auc']:.4f}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "epoch_curve.json").write_text(json.dumps(curve, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print(f"-> {a.out/'epoch_curve.json'}")


if __name__ == "__main__":
    main()
