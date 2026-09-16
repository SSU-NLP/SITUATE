#!/usr/bin/env python3
"""D-B29 F단계 — 확정 조합 + 기준선을 **dev-3 개봉**해 평가한다.

## 왜 지금 열어도 되나
D-B29 프로토콜: 구조·HP·임계를 **전부 앵커 val 5-fold OOF** 로 골랐고, 탐색 29+4 run 은
`--dev3` 없이 돌아 dev-3 를 한 번도 보지 않았다. 따라서 dev-3 는 **선택에 쓰이지 않았고**
여기서의 개봉은 *외적 방향 확인*이다(D-B29 7단계).

## 확정 조합 (D-B29 규칙 적용 결과)
| 축 | 후보 | 판정 |
|---|---|---|
| 용량 (H) | slot {128,256,384} × bn {16,32} | 최고 Δ +0.0016 < 동률기준 0.005 → **기준선(단순한 쪽)** |
| 학습 (L) | lr {1e-4,3e-4} × ep {8,16} | lr 1e-4 는 −0.004 로 나쁨. ep16 은 seed 불일치 → **기준선** |
| 잔차특징 (P) | resid_signed / resid_sq / pred_centric | 최고 Δ +0.0006, 기준 +0.01 미달 · seed 순위 반전 → **delta** |
| ψₑ (Ψ) | cummean / ema | 최고 Δ −0.0006, 기준 +0.005 미달 → **없음** |
| **임계 규칙** | 고정분위수 / otsu / offline·online μ+cσ | **고정분위수가 OOF 200 fold 중 192 승** → **고정분위수** |

⇒ **모델은 기준선 그대로. 바뀌는 것은 임계 규칙 하나뿐이며, 그것도 앵커 OOF 가 골랐다.**

🔴 이 스크립트는 **재학습하지 않는다** — wave1 B묶음이 저장한 체크포인트를 그대로 평가한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.factored_slot import FactoredSlot  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from train_anchor_m4m5 import EXP, T_STAR, Corpus, axis_probs  # noqa: E402

SEEDS = (0, 1, 2)
NS = "_per_slot_slot"                       # 기준선 네임스페이스 (dynamics per_slot · pred_space slot)


def load(tag: str, seed: int, dev: str, ns: str = NS):
    """`ns` 는 산출 네임스페이스(기본 = D-B29 기준선). D-B33 애블레이션이 다른 ns 로 재사용한다."""
    ck = torch.load(EXP / f"{tag}{ns}_seed{seed}.pt", map_location=dev, weights_only=False)
    m = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"], pos_weight=ck["pos_weight"],
                     dynamics=ck.get("dynamics"), pred_space=ck["args"].get("pred_space", "obs"),
                     bottleneck=ck["args"].get("bottleneck", 8),
                     d1=ck["args"].get("input_path", "axis_extractor") == "axis_extractor",
                     shared_dim=ck["args"].get("shared_proj_dim") or None,
                     tagger=(ck["args"].get("tagger") if ck["args"].get("tagger") != "none"
                             else None),
                     tagger_axes=ck["args"].get("tagger_axes", 5),
                     block_readout=ck["args"].get("block_readout", False),
                     dropout=ck["args"].get("dropout", 0.4)).to(dev)
    m.load_state_dict(ck["state"])
    return m.eval(), ck


def anchor_rate(c: Corpus, va: list[int]) -> float:
    """🔒 앵커 val 의사-gold 경계율 — 임계 동결값. dev-3 미사용."""
    g = np.concatenate([(c.resets[int(c.to[i]):int(c.to[i + 1])].sum(1) >= T_STAR) for i in va])
    return float(g.mean())


def eval_dev3(m, dev3: Corpus, dev: str, rate: float) -> dict:
    probs = axis_probs(m, dev3, list(range(dev3.n_docs)), dev, bs=4, with_anchor=False)
    ys, out = [], {}
    acc: dict[str, list] = {}
    npred: dict[str, int] = {}
    per_doc: dict[str, dict] = {}
    for i, pr in enumerate(probs):
        g = dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])].astype(np.int8)
        s = pr.sum(1)
        seq = np.concatenate([[0.0], s]).tolist()
        rules = {"🔒고정분위수(앵커동결)": (s >= np.quantile(s, 1 - rate)).astype(np.int8),
                 "[참고] online μ+1.0σ": np.asarray(
                     adaptive_boundaries(seq, c=1.0, mode="online_nf"), np.int8)[:len(s)],
                 "[참고] otsu": np.asarray(adaptive_boundaries(seq, mode="otsu"), np.int8)[:len(s)]}
        y = np.concatenate([g, [0]]).tolist()
        ys.append(y)
        for k, b in rules.items():
            p = np.concatenate([b, [0]]).tolist()
            acc.setdefault(k, []).append(p)
            npred[k] = npred.get(k, 0) + int(b.sum())
        per_doc[dev3.doc_ids[i]] = {
            "n_gold": int(g.sum()),
            "score": score_dialogues([y], [acc["🔒고정분위수(앵커동결)"][-1]])["score"],
            "n_pred": int(rules["🔒고정분위수(앵커동결)"].sum())}
    for k, v in acc.items():
        out[k] = {"score": score_dialogues(ys, v)["score"], "n_pred": npred[k]}
    out["per_doc"] = per_doc
    return out


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c, dev3 = Corpus("phase1"), Corpus("dev3")
    rng = np.random.default_rng(20260728)                 # 학습과 동일 분할
    va = rng.permutation(c.n_docs)[:max(1, int(c.n_docs * 0.05))].tolist()
    rate = anchor_rate(c, va)
    print(f"[동결] 앵커 val 의사-gold 경계율 r = {rate:.4f}  (dev-3 미사용)", flush=True)

    res: dict = {"anchor_rate": rate, "M5": {}, "M4": {}}
    for tag in ("M5", "M4"):
        for s in SEEDS:
            res[tag][s] = eval_dev3(load(tag, s, dev)[0], dev3, dev, rate)
            print(f"  {tag} seed{s}  Score {res[tag][s]['🔒고정분위수(앵커동결)']['score']:.4f}",
                  flush=True)

    key = "🔒고정분위수(앵커동결)"
    m5 = [res["M5"][s][key]["score"] for s in SEEDS]
    m4 = [res["M4"][s][key]["score"] for s in SEEDS]
    d = [a - b for a, b in zip(m5, m4)]
    ys = [np.concatenate([dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])], [0]]).tolist()
          for i in range(dev3.n_docs)]
    floor = score_dialogues(ys, [[0] * len(y) for y in ys])["score"]

    L = ["# D-B29 F단계 — 확정 조합 dev-3 개봉 (3 seed)\n",
         "> **프로토콜 준수:** 탐색 33 run 은 전부 `--dev3` 없이 돌아 dev-3 를 한 번도 안 봤다."
         " 구조·HP·임계는 **앵커 val 5-fold OOF** 가 골랐다. 여기 개봉은 *외적 방향 확인*이다.",
         "> **확정 = 기준선 모델 그대로.** 바뀐 것은 **임계 규칙 하나**(적응 μ+cσ → 고정분위수)이며,"
         f" 그 값 r={rate:.4f} 도 **앵커에서** 왔다.",
         "> 🔴 dev-3 는 연습장이다. 이 표로 후보를 다시 고르지 않는다(D-B29 7단계).\n",
         "\n## 결과 — dev-3 pooled Score (3 seed)\n",
         "| 모델 | seed 0 | seed 1 | seed 2 | 평균 ± std |", "|---|---|---|---|---|",
         f"| **M5-FactoredSlot** | {m5[0]:.4f} | {m5[1]:.4f} | {m5[2]:.4f} | "
         f"**{np.mean(m5):.4f} ± {np.std(m5, ddof=1):.4f}** |",
         f"| M4-MonoAnchor | {m4[0]:.4f} | {m4[1]:.4f} | {m4[2]:.4f} | "
         f"{np.mean(m4):.4f} ± {np.std(m4, ddof=1):.4f} |",
         f"| **Δ (M5−M4)** | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | "
         f"**{np.mean(d):+.4f} ± {np.std(d, ddof=1):.4f}** |",
         f"| *경계 없음 (하한)* | | | | *{floor:.4f}* |",
         "\n## 임계 규칙 대조 (M5, seed 평균)\n",
         "| 규칙 | dev-3 Score | 예측 경계 |", "|---|---|---|"]
    for k in ("🔒고정분위수(앵커동결)", "[참고] online μ+1.0σ", "[참고] otsu"):
        sc = np.mean([res["M5"][s][k]["score"] for s in SEEDS])
        npd = np.mean([res["M5"][s][k]["n_pred"] for s in SEEDS])
        L.append(f"| {k} | {sc:.4f} | {npd:.0f} |")
    L += ["\n> 🔒 표시가 **앵커 OOF 가 고른 규칙**이다(200 fold 중 192 승). 나머지는 참고용이며"
          " **이 표를 보고 규칙을 바꾸면 selection-on-dev 오염**이다.",
          "\n## 문서별 (앵커동결 규칙, seed 0)\n", "| doc | gold | 예측 | Score |", "|---|---|---|---|"]
    for dname, v in res["M5"][0]["per_doc"].items():
        L.append(f"| {dname} | {v['n_gold']} | {v['n_pred']} | {v['score']:.4f} |")
    L += ["\n⚠️ dev-3 = 연습장(3편). 시험지 미개봉. 앵커 k=1 (S_self 0.659 = 상한).",
          "⚠️ 이번 배치에서 **용량·학습HP·잔차특징·ψₑ 전부 무효**였다 — 병목은 그 중에 없다."]
    p = EXP / "DB29_FINAL_DEV3.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    (EXP / "db29_final_dev3.json").write_text(json.dumps(res, indent=2, ensure_ascii=False,
                                                         default=float))
    print("\n".join(L))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
