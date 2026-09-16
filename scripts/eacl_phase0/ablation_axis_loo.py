"""Ablation — 축 leave-one-out. 어느 차원이 실제로 경계를 만드는가.

해석가능성 주장의 증거. 학습된 M5 체크포인트를 그대로 쓰고 **읽기 단계에서만** 축을 빼서
경계 점수를 다시 만든다. 재학습이 아니라 readout LOO 다 — 5축이 함께 학습된 표현은 유지한 채,
"이 축의 신호가 경계 결정에 얼마나 기여하는가"를 잰다.

  full   : s = Σ_{k∈5} P(reset_k)
  −axis  : s = Σ_{k≠axis} P(reset_k)      (해당 축만 뺀다)
  only   : s = P(reset_axis)              (그 축 하나로만)

⚠️ 진단 전용이다. 축을 신호에서 **영구 제거하자는 제안이 아니다** — 약축이어도 5축은 유지한다.
⚠️ 경계 규칙은 적응임계(문서 내 상위 rate 분위수) 하나만 쓴다. rate 는 앵커에서 동결된 값이며
   dev-3 gold 를 쓰지 않는다. 지표는 Score / BOR / Lift + 퇴화 하한.

    python scripts/eacl_phase0/ablation_axis_loo.py --ckpt-glob '...M5_*_e40_seed*.pt' --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
import train_anchor_m4m5 as T  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries as AB  # noqa: E402

AXES = list(T.AXES)


def metrics(ys: list[list[int]], preds: list[list[int]]) -> dict:
    n_gold = sum(sum(y[:-1]) for y in ys)
    n_tr = sum(len(y) - 1 for y in ys)
    n_pred = sum(sum(p[:-1]) for p in preds)
    hit = sum(int(a) * int(b) for y, p in zip(ys, preds) for a, b in zip(y[:-1], p[:-1]))
    dens = n_gold / max(n_tr, 1)
    prec = hit / n_pred if n_pred else float("nan")
    return {"score": float(np.mean([score_dialogues([y], [p])["score"] for y, p in zip(ys, preds)])),
            "pooled_score": score_dialogues(ys, preds)["score"],
            "bor": (n_pred / n_gold) if n_gold else float("nan"),
            "lift": (prec / dens) if (n_pred and dens > 0) else float("nan"),
            "n_pred": int(n_pred)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-glob", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate", type=float, default=0.0657,
                    help="앵커 경계율(적응임계의 문서 내 분위수). 결과 JSON 에서 못 읽으면 이 값.")
    a = ap.parse_args()

    corpora = {"dev-3": T.Corpus("dev3"), "test3": T.Corpus("test3")}
    ck = sorted(glob.glob(a.ckpt_glob))
    if not ck:
        raise SystemExit(f"체크포인트 없음: {a.ckpt_glob}")
    print(f"체크포인트 {len(ck)}개")

    rows: list[dict] = []
    for path in ck:
        blob = torch.load(path, map_location=a.device, weights_only=False)
        ta = argparse.Namespace(**blob["args"])          # 학습 당시 인자 그대로 복원
        model = T.FactoredSlot(
            slot_dim=ta.slot_dim, mono=blob["mono"], pos_weight=blob["pos_weight"],
            d1=(ta.input_path == "axis_extractor"),
            shared_dim=(ta.shared_proj_dim or None),
            tagger=(None if ta.tagger == "none" else ta.tagger), tagger_axes=ta.tagger_axes,
            block_readout=ta.block_readout, dynamics=blob["dynamics"],
            pred_weight=ta.pred_weight, pred_space=ta.pred_space, head_feat=ta.head_feat,
            slot_reset=(None if ta.slot_reset == "none" else ta.slot_reset),
            slot_reset_dropout=ta.slot_reset_dropout,
            psi=(None if ta.psi == "none" else ta.psi), psi_beta=ta.psi_beta,
            psi_dropout=ta.psi_dropout, bottleneck=ta.bottleneck, dropout=ta.dropout).to(a.device)
        model.load_state_dict(blob["state"])
        # 앵커 경계율 = 결과 JSON 에서 읽는다(체크포인트엔 없다). 없으면 --rate 기본값.
        rate = None
        rj = Path(path).with_suffix("").name
        for cand in T.EXP.glob("results_*.json"):
            try:
                d = json.loads(cand.read_text())
            except Exception:
                continue
            for mk in ("M5", "M4"):
                if mk in d and rj.startswith(mk) and rj.endswith(cand.stem.split("results_")[-1]):
                    rate = d[mk].get("anchor_boundary_rate")
        r = float(rate) if rate else a.rate
        views = {"full": list(range(5))}
        for k, ax in enumerate(AXES):
            views[f"-{ax}"] = [j for j in range(5) if j != k]
            views[f"only:{ax}"] = [k]
        ys, per_view = [], {n: [] for n in views}
        for cname, c in corpora.items():
            probs = T.axis_probs(model, c, list(range(c.n_docs)), a.device, 4, with_anchor=False)
            for i, pr in enumerate(probs):
                g = c.gold[int(c.to[i]):int(c.to[i + 1])].astype(np.int8)
                ys.append(np.concatenate([g, [0]]).tolist())
                for name, cols in views.items():
                    sc = pr[:, cols].sum(1)
                    # 🔒 채택 적응임계 online_nf c=1.0 (미래 미사용). 분위수 방식은 오프라인이다.
                    b = np.asarray(AB(np.concatenate([[0.0], sc]).tolist(),
                                      c=1.0, mode="online_nf"), np.int8)[:len(sc)]
                    per_view[name].append(np.concatenate([b, [0]]).tolist())
        rows.append({"ckpt": Path(path).name, "rate": r,
                     **{name: metrics(ys, per_view[name]) for name in views}})
        print(f"  {Path(path).name}: full Score {rows[-1]['full']['score']:.4f}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "axis_loo.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    # 요약
    def agg(name: str, key: str) -> tuple[float, float]:
        v = [r[name][key] for r in rows if r[name][key] == r[name][key]]
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)

    print(f"\n{'view':22s} {'Score':>16s} {'ΔScore':>9s} {'BOR':>8s} {'Lift':>8s}")
    fm, _ = agg("full", "score")
    for name in ["full"] + [f"-{x}" for x in AXES] + [f"only:{x}" for x in AXES]:
        m, sd = agg(name, "score")
        b, _ = agg(name, "bor")
        l, _ = agg(name, "lift")
        print(f"{name:22s} {m:8.4f}±{sd:.4f} {m-fm:+9.4f} {b:8.3f} {l:8.3f}")
    print(f"\n-> {a.out/'axis_loo.json'}")


if __name__ == "__main__":
    main()
