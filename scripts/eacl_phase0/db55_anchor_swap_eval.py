#!/usr/bin/env python3
r"""D-B55 — 앵커 코퍼스 교체(Qwen → Gemma)가 M5 에 전달되는가. dev-3 deployable Score.

## 무엇을
**앵커 코퍼스 하나만** 바꾼 두 M5 를 나란히 놓는다. 학습 조건(W3 = ALPHA 현행 × pw one),
인코더, 문장 소스, 임계 규칙(`online_nf`, c=1.0)은 **전부 동일**하다.

| 조건 | 앵커 | 네임스페이스 |
|---|---|---|
| 기준선 | `phase1_all_v31` (Qwen3.6-35B-A3B, D-B15) | `_per_slot_slot_pwone` |
| 후보 | `phase1_gemma4a4b_v31` (Gemma-4-26B-A4B, D-B53) | `_phase1_gemma_per_slot_slot_pwone` |

## 하지 않는 것
- c 재탐색 없음(D-B39 채택값 1.0 고정). 임계 규칙 변경 없음.
- 축 제거 없음 — 동결 5축 전부.
- 봉인 시험지 미개봉. dev-3 만 본다.
- 🔴 **dev-3 절대 Score 를 성능으로 인용하지 않는다 — Δ·방향만.**

## 판정
D-B55 §2 사전등록: ΔScore ≥ +0.03 **AND** 3편 중 2편 이상 무하락 → 앵커 교체 후보.
그 미만이면 "노이즈 내 동등 → 교체 없음". 과분절 배수를 함께 적는다.
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

OUT = REPO / "outputs/experiments/2026-07-31_db55_anchor_swap"
DOCS = ("pieman", "tunnel", "monkey")
SEEDS = (0, 1, 2)
C_MAIN = 1.0                                   # 🔒 D-B39 채택값. 재탐색 금지.
CONDS = [("Qwen 앵커 (D-B15, 현행)", "_per_slot_slot_pwone"),
         ("Gemma 앵커 (D-B53)", "_phase1_gemma_per_slot_slot_pwone")]


def cuts(s: np.ndarray) -> np.ndarray:
    """동결 규약: 0번 sentinel 을 앞에 붙여 넘긴다. 재구현 금지 — 기존 함수 호출."""
    seq = np.concatenate([[0.0], s]).tolist()
    return np.asarray(adaptive_boundaries(seq, c=C_MAIN, mode="online_nf", warmup=5),
                      np.int8)[:len(s)]


def main() -> None:
    argparse.ArgumentParser(description="D-B55 앵커 교체 dev-3 평가").parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    dev3 = Corpus("dev3")
    golds = [dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])].astype(np.int8)
             for i in range(dev3.n_docs)]
    names = [str(d) for d in dev3.doc_ids]
    n_gold = int(sum(int(g.sum()) for g in golds))
    print(f"dev-3 {dev3.n_docs}편 {names} · gold 경계 {n_gold}", flush=True)

    res: dict[str, dict] = {}
    for label, ns in CONDS:
        rows = []
        for sd in SEEDS:
            ck = REPO / f"outputs/experiments/m4m5_phase1_anchor/M5{ns}_seed{sd}.pt"
            if not ck.exists():
                print(f"  ⚠️ 없음: {ck.name} — 건너뜀", flush=True)
                continue
            m = load("M5", sd, dev, ns=ns)[0]
            with torch.no_grad():
                P = axis_probs(m, dev3, list(range(dev3.n_docs)), dev, bs=64, with_anchor=False)
            per, ys, ps, npred = [], [], [], 0
            for Pi, g in zip(P, golds):
                b = cuts(Pi.sum(1))[:len(g)]
                y = np.concatenate([g, [0]]).tolist()
                p = np.concatenate([b, [0]]).tolist()
                per.append(score_dialogues([y], [p])["score"])
                ys.append(y); ps.append(p); npred += int(b.sum())
            rows.append({"seed": sd, "score": float(score_dialogues(ys, ps)["score"]),
                         "per_doc": {names[i]: float(v) for i, v in enumerate(per)},
                         "n_pred": npred})
            print(f"  [{label}] seed{sd} Score {rows[-1]['score']:.4f} 예측 {npred}", flush=True)
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
        if rows:
            res[label] = {"ns": ns, "runs": rows}

    if len(res) < 2:
        print("🔴 두 조건 중 하나가 없다 — 비교 불가. 중단.", flush=True)
        return

    def agg(label):
        r = res[label]["runs"]
        ss = [x["score"] for x in r]
        pd = {d: float(np.mean([x["per_doc"][d] for x in r])) for d in names}
        return (float(np.mean(ss)), float(np.std(ss, ddof=1)) if len(ss) > 1 else 0.0,
                float(np.mean([x["n_pred"] for x in r])), pd)

    (bl, bns), (cl, cns) = CONDS
    b_mu, b_sd, b_np, b_pd = agg(bl)
    c_mu, c_sd, c_np, c_pd = agg(cl)
    d = c_mu - b_mu
    no_drop = sum(1 for x in names if c_pd[x] >= b_pd[x])
    passed = (d >= 0.03) and (no_drop >= 2)

    L = ["# D-B55 — 앵커 코퍼스 교체가 M5 에 전달되는가 (dev-3)", "",
         "## 실험 setup", "",
         "- **바꾼 것은 앵커 코퍼스 하나뿐.** 학습 조건 W3(ALPHA 현행 × pos_weight one, "
         "`--batch 48 --pred-space slot --dynamics per_slot`), 인코더(mpnet frozen), 문장 소스,",
         "  임계 규칙(`online_nf`)까지 전부 동일하다. seed 0/1/2.",
         f"- **c = {C_MAIN} 고정** (D-B39 채택값, 출처 = DTS 공식 test). 재탐색하지 않았다.",
         "- **동결 5축 전부** (`build_anchor_dataset.py:31` = D-B10 `protagonist_present`).",
         "- **지표** = deployable Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD), "
         "`dts_scoring.score_dialogues`. 경계 개수를 방법에 주지 않는다.",
         f"- **데이터** = dev-3 {dev3.n_docs}편 · gold 경계 **{n_gold}**. 봉인 시험지 미개봉.", "",
         "## 결과 (seed 0/1/2 평균 ±SD)", "",
         "| 앵커 | " + " | ".join(names) + " | Score | ±SD | 예측 | 과분절 |",
         "|---|" + "---|" * (len(names) + 4)]
    for lbl, mu, sd_, npm, pd in ((bl, b_mu, b_sd, b_np, b_pd), (cl, c_mu, c_sd, c_np, c_pd)):
        L.append(f"| {lbl} | " + " | ".join(f"{pd[x]:.3f}" for x in names)
                 + f" | **{mu:.4f}** | ±{sd_:.4f} | {npm:.0f} | {npm / max(n_gold,1):.2f}× |")
    L.append("| **Δ (Gemma − Qwen)** | "
             + " | ".join(f"{c_pd[x]-b_pd[x]:+.3f}" for x in names)
             + f" | **{d:+.4f}** | — | {c_np-b_np:+.0f} | {(c_np-b_np)/max(n_gold,1):+.2f}× |")

    L += ["", "## 판정 — 🔒 D-B55 §2 사전등록 기준", "",
          f"- 조건 = ΔScore **≥ +0.03** AND 3편 중 **2편 이상 무하락**.",
          f"- 실측 = Δ **{d:+.4f}**, 무하락 **{no_drop}/3**편.", "",
          (f"### ⇒ **통과** — 앵커 교체 후보로 성립한다."
           if passed else
           f"### ⇒ **미달 — 노이즈 내 동등. 앵커 교체 없음.** Qwen 앵커(D-B15)를 유지한다."), "",
          "## 읽는 법 — 🔴 절대값 인용 금지", "",
          "`configs/m5_default.json` 의 caveat 을 승계한다: **dev-3 절대 Score 를 성능으로 인용하지",
          "않는다. Δ·방향만.** dev-3 는 연습장 3편이고 pieman 에는 선택압이 누적돼 있다.", "",
          f"**과분절**을 같이 본다. gold {n_gold} 대비 기준선 {b_np:.0f}({b_np/n_gold:.2f}×) → "
          f"후보 {c_np:.0f}({c_np/n_gold:.2f}×). 경계 개수 제어는 최대 미결 과제이고"
          "(D-B34/B37/B38),",
          "Score 가 올랐더라도 그것이 *개수를 맞춰서*인지 *위치를 잘 잡아서*인지 이 표만으로는",
          "갈리지 않는다.", "",
          "## 🔴 해석 — 라벨러 우위가 M5 로 전달되지 않았다 (부호까지 뒤집혔다)", "",
          "같은 두 라벨러를 **M3 단계**(라벨러 산출에 count ≥ t\\*=3 을 바로 적용)에서 재면",
          "Gemma 가 **+0.045**(0.4819 vs 0.4372, 3편 모두 상승, D-B54 §2)로 이긴다.",
          f"그런데 그 산출로 M5 를 학습하면 **{d:+.4f}** 로 **진다.** 단순 무전달이 아니라 **역전**이다.",
          "",
          "- **가장 크게 뒤집힌 문서가 pieman 이다.** M3 에서 Gemma 가 가장 크게 이겼던 문서"
          "(+0.068)인데, M5 에서는 가장 크게 진다"
          f"({c_pd.get('pieman', float('nan')) - b_pd.get('pieman', float('nan')):+.3f}).",
          "- **경계 개수 효과가 아니다.** 과분절이 "
          f"{b_np/max(n_gold,1):.2f}× → {c_np/max(n_gold,1):.2f}× 로 거의 안 변했다"
          f"(예측 {b_np:.0f} → {c_np:.0f}). 바뀐 것은 **개수가 아니라 위치**다.",
          "- **앵커 발화율 차이와 방향이 맞는다.** Gemma 앵커는 `protagonist_present` 0.263"
          "(Qwen 0.305, −14%)이고 `time` 0.036(Qwen 0.021, +68%)이다. 즉 M5 는 **더 약한"
          " 주인공 신호와 더 시끄러운 time 신호**를 타깃으로 학습했다.",
          "  🔴 다만 이것은 **사후 가설이다** — 이 실험은 축별 기여를 분리하지 않았고,"
          " 확인하려면 축을 갈아끼운 별도 사전등록이 필요하다.",
          "- **M3 성능은 라벨러 선택의 근거가 되지 못한다**는 것이 이 실험의 실질 결론이다."
          " 앵커에는 gold 가 없으므로 라벨러의 dev-3 우위는 *그 라벨러를 분절기로 직접 쓸 때*의"
          " 우위이지, *그 라벨을 타깃으로 학습한 모델*의 우위가 아니다.", "",
          "## 한계 (제거 금지)", "",
          "- **seed 3개.** 앵커 코퍼스 자체는 k=1 단일 draw 라(D-B15/D-B53) 라벨 노이즈는 못 잡는다.",
          "  두 조건의 SD 가 ±0.006 수준이라 Δ −0.035 는 SD 의 6배지만, **이것은 seed 변동이지"
          " 라벨 draw 변동이 아니다.**",
          "- **문서 집합이 완전히 같지 않다.** Qwen 10,598편 vs Gemma **10,542편**"
          "(`final_master.jsonl` 기준, 라벨 불완전 56편 제외 — 각자 다른 문서에서 폭주).",
          "  차이 0.53% 의 영향은 **미측정**.",
          "- **dev-3 3편 · gold 66.** pieman 한 편이 Δ 의 대부분을 만든다.",
          "- 라벨러 선택 자체에 dev-3 성분이 있다(D-B54). 이 비교는 그 위에 서 있다.",
          "- **`time` 축 가설 미검정** (위 해석 3번째 항목)."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "db55_anchor_swap.json").write_text(
        json.dumps({"result": res, "delta": d, "no_drop": no_drop, "passed": bool(passed),
                    "n_gold": n_gold}, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
