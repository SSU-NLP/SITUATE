#!/usr/bin/env python3
r"""D-B58 — 축 사다리 w3 → i2 → chain (gpt-oss 앵커). dev-3 deployable Score.

## 무엇을
**앵커 축 정의만** 바꾼 세 조건을 나란히 놓는다. 라벨러(gpt-oss-120b)·학습 레시피·임계 규칙은 동일.

| 이름 | split | 인과 축 | 의도 축 |
|---|---|---|---|
| w3 | `phase1_gptoss` | 동결 | 동결 |
| i2 | `phase1_gptoss_i2` | 동결 | 재조작화(carry 없음·채택만) |
| chain | `phase1_gptoss_chainstrict_i2` | 사슬 strict(F12) | 재조작화 |

## 🔒 판정 규칙 (D-B58 §2, 사용자 지정 — 결과 보고 수정 금지)
- **w3 > i2 → 의도 축을 바꾸지 않는다.**
- **i2 > chain → 인과 축도 바꾸지 않는다.**

⚠️ 이것은 단순 대소 비교이지 비열등 문턱이 아니다. **|Δ| < SD 면 "구별 안 됨"으로 병기**한다
(규칙을 바꾸는 것이 아니라 해상도를 함께 보고하는 것).

## 하지 않는 것
- c·임계 재탐색(`online_nf`, c=1.0 고정, D-B39) · 축 제거 · 시험지 개봉
- 🔴 dev-3 절대 Score 를 성능으로 인용하지 않는다 — Δ·방향만.
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
from db29_final_eval import load  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402

OUT = REPO / "outputs/experiments/2026-08-01_gptoss_axis_ladder"
SEEDS = (0, 1, 2)
C_MAIN = 1.0                                  # 🔒 D-B39 채택값
CONDS = [("w3 (인과 동결 · 의도 동결)", "_phase1_gptoss_per_slot_slot_pwone"),
         ("i2 (인과 동결 · 의도 재조작화)", "_phase1_gptoss_i2_per_slot_slot_pwone"),
         ("chain (인과 사슬 · 의도 재조작화)", "_phase1_gptoss_chainstrict_i2_per_slot_slot_pwone")]


def cuts(s: np.ndarray) -> np.ndarray:
    seq = np.concatenate([[0.0], s]).tolist()
    return np.asarray(adaptive_boundaries(seq, c=C_MAIN, mode="online_nf", warmup=5),
                      np.int8)[:len(s)]


def main() -> None:
    argparse.ArgumentParser(description="D-B58 축 사다리 dev-3 평가").parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    dev3 = Corpus("dev3")
    golds = [dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])].astype(np.int8)
             for i in range(dev3.n_docs)]
    names = [str(d) for d in dev3.doc_ids]
    n_gold = int(sum(int(g.sum()) for g in golds))
    print(f"dev-3 {names} · gold {n_gold}", flush=True)

    agg = {}
    for label, ns in CONDS:
        rows = []
        for sd in SEEDS:
            ck = REPO / f"outputs/experiments/m4m5_phase1_anchor/M5{ns}_seed{sd}.pt"
            if not ck.exists():
                print(f"  ⚠️ 없음 {ck.name}", flush=True)
                continue
            m = load("M5", sd, dev, ns=ns)[0]
            with torch.no_grad():
                P = axis_probs(m, dev3, list(range(dev3.n_docs)), dev, bs=64, with_anchor=False)
            per, ys, ps, npred = [], [], [], 0
            for Pi, g in zip(P, golds):
                b = cuts(Pi.sum(1))[:len(g)]
                y = np.concatenate([g, [0]]).tolist(); p = np.concatenate([b, [0]]).tolist()
                per.append(score_dialogues([y], [p])["score"]); ys.append(y); ps.append(p)
                npred += int(b.sum())
            rows.append({"seed": sd, "score": float(score_dialogues(ys, ps)["score"]),
                         "per_doc": {names[i]: float(v) for i, v in enumerate(per)},
                         "n_pred": npred})
            print(f"  [{label}] seed{sd} {rows[-1]['score']:.4f} 예측 {npred}", flush=True)
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
        if rows:
            ss = [r["score"] for r in rows]
            agg[label] = {"ns": ns, "runs": rows, "mu": float(np.mean(ss)),
                          "sd": float(np.std(ss, ddof=1)) if len(ss) > 1 else 0.0,
                          "n_pred": float(np.mean([r["n_pred"] for r in rows])),
                          "per_doc": {d: float(np.mean([r["per_doc"][d] for r in rows]))
                                      for d in names}}
    if len(agg) < 3:
        print("🔴 세 조건이 다 없다 — 판정 보류", flush=True)

    L = ["# D-B58 — 앵커 축 사다리 (gpt-oss-120b 라벨러) · dev-3", "",
         "## 실험 setup", "",
         "- **바꾼 것은 앵커 축 정의뿐.** 라벨러(gpt-oss-120b) · 학습 레시피"
         "(`--pos-weight-mode one --pred-space slot`, batch 48 · per_slot · epochs 8) ·",
         f"  임계(`online_nf`, c={C_MAIN} 고정, D-B39) 전부 동일. seed 0/1/2.",
         "- **지표** = deployable Score(경계 개수를 방법에 주지 않는다). 동결 5축 전부.",
         f"- **데이터** = dev-3 {dev3.n_docs}편 · gold **{n_gold}**. 봉인 시험지 미개봉.", "",
         "## 결과 (seed 0/1/2 평균 ±SD)", "",
         "| 조건 | " + " | ".join(names) + " | Score | ±SD | 예측 | 과분절 |",
         "|---|" + "---|" * (len(names) + 4)]
    for label, _ in CONDS:
        if label not in agg:
            continue
        a = agg[label]
        L.append(f"| {label} | " + " | ".join(f"{a['per_doc'][x]:.3f}" for x in names)
                 + f" | **{a['mu']:.4f}** | ±{a['sd']:.4f} | {a['n_pred']:.0f} | "
                   f"{a['n_pred']/max(n_gold,1):.2f}× |")

    L += ["", "## 🔒 판정 (D-B58 §2 — 사용자 지정 규칙)", ""]
    verdict = {}
    for a_lbl, b_lbl, axis, keep_msg, change_msg in [
            (CONDS[0][0], CONDS[1][0], "의도 축",
             "**의도 축을 바꾸지 않는다.**", "의도 축 재조작화가 앞선다."),
            (CONDS[1][0], CONDS[2][0], "인과 축",
             "**인과 축도 바꾸지 않는다.**", "인과 축 사슬 교체가 앞선다.")]:
        if a_lbl not in agg or b_lbl not in agg:
            L.append(f"- {axis}: 🔴 조건 누락 — 판정 불가"); continue
        A, B = agg[a_lbl], agg[b_lbl]
        d = A["mu"] - B["mu"]
        pooled = max(A["sd"], B["sd"])
        tie = abs(d) < pooled
        verdict[axis] = {"delta": d, "sd": pooled, "tie": bool(tie), "keep": bool(d > 0)}
        L += [f"### {axis} — Δ({a_lbl.split(' ')[0]} − {b_lbl.split(' ')[0]}) = **{d:+.4f}** "
              f"(SD {pooled:.4f})",
              f"- ⇒ {keep_msg if d > 0 else change_msg}"
              + ("  🔴 **단 |Δ| < SD 라 두 조건은 구별되지 않는다** — 이 판정은 우연일 수 있다."
                 if tie else ""), ""]

    L += ["## 읽는 법 — 🔴 절대값 인용 금지", "",
          "`configs/m5_default.json` caveat 승계: **dev-3 절대 Score 를 성능으로 인용하지 않는다.",
          "Δ·방향만.** dev-3 는 연습장 3편이고 pieman 에 선택압이 누적돼 있다.", "",
          "**과분절**을 함께 본다. 앵커 경계율이 조건마다 크게 다르다"
          "(w3 0.0861 · i2 0.0782 · chain **0.0211**) — chain 은 경계를 훨씬 적게 찍도록 학습된다.",
          "Score 변화가 *위치* 때문인지 *개수* 때문인지 이 표만으로는 갈리지 않는다.", "",
          "## 한계 (제거 금지)", "",
          "- **chain 조건은 순수하지 않다** — 사슬 라벨이 없는 **154편(1.5%)** 은 인과 열이 동결 규칙 그대로다.",
          "- **임베딩 재인코딩 없음** — `phase1_emb.npy` 슬라이스 재사용(`build_gptoss_split.py`).",
          "- **seed 3개.** 앵커는 k=1 단일 draw 라 라벨 노이즈는 못 잡는다.",
          "- 이 사다리는 **gpt-oss 앵커 위에서만** 성립한다. Qwen 앵커에서 같은 순위가 나온다는 보장 없음."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "ladder.json").write_text(json.dumps({"agg": agg, "verdict": verdict, "n_gold": n_gold},
                                                indent=2, ensure_ascii=False, default=float),
                                     encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
