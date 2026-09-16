#!/usr/bin/env python3
"""착수 전 확인 — 채택 체크포인트(pos_weight 제거) + **적응 임계**로 대화 세트 하한 통과 여부.

## 왜
`m4m5_dts_transfer.py` 표에서 superseg 0.3157 / ami 0.2737 이 무경계 하한을 못 넘었다.
그러나 그 표는 (a) **폐기된 체크포인트**(`_per_slot_slot` = pos_weight 살아있는 W0)와
(b) **문서 내 상위 6.49% 분위수** 규칙을 쓴다.

음성 트랙 조건 ② 가 실제로 쓰는 것은 (a') 채택본 `models/m5_default/M5.pt`
(= `M5_per_slot_slot_pwone_seed0.pt`, 양성 가중치 전축 1) 와 (b') **적응 임계** μ + 1.0σ 다.
계수 1.0 의 출처가 SuperDialseg 공식 test 이므로 superseg 는 특히 확인 대상이다.

**판정:** 하한을 넘으면 기존 표는 다른 경계 규칙 얘기이므로 음성 트랙 진행.
못 넘으면 조건 ② 의 임계 규칙을 사전 등록에서 다시 쓴다.

🔴 점수 확인 전에 기준을 적어 둔다 — 사후 기준 변경 금지.
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
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402  🔒 동결 구현
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402

CONFIG = REPO / "configs/m5_default.json"
SETS = ("dev3", "dts_tiage", "dts_dialseg711", "dts_superseg", "dts_ami")
OUT = REPO / "outputs/experiments/2026-07-31_voice_duplex"


def frozen_adaptive(s: np.ndarray, c: float, mode: str) -> np.ndarray:
    """`m5_default.segment()` 와 **동일한 호출 규약** — 0 sentinel 을 앞에 붙이고 [:n] 로 자른다.

    재구현 금지: 동결 `hi_ontop_v2.adaptive_boundaries` 를 그대로 호출한다.
    (online_nf 는 현재 표본 s_t 를 통계에 포함한 뒤 임계를 계산한다 — 미래는 쓰지 않는다.)
    """
    n = len(s)
    return np.asarray(adaptive_boundaries(np.concatenate([[0.0], s]).tolist(), c=c, mode=mode),
                      np.int8)[:n]


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    c_thr = cfg["threshold"]["c"]
    mode = cfg["threshold"]["mode"]
    assert mode == "online_nf", mode
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(REPO / cfg["checkpoint"], map_location=dev, weights_only=False)
    m = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"],
                     pos_weight=ck["pos_weight"], dynamics=ck.get("dynamics"),
                     d1=ck["args"].get("input_path", "axis_extractor") == "axis_extractor",
                     shared_dim=ck["args"].get("shared_proj_dim") or None,
                     pred_space=ck["args"].get("pred_space", "obs")).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()

    rows = []
    for name in SETS:
        corp = Corpus(name)
        idx = list(range(corp.n_docs))
        probs = axis_probs(m, corp, idx, dev, bs=8, with_anchor=False)
        ys, p_ad, p_pct = [], [], []
        for i, pr in zip(idx, probs):
            g = corp.gold[int(corp.to[i]):int(corp.to[i + 1])].astype(np.int8)
            s = pr.sum(1)
            ys.append(np.concatenate([g, [0]]).tolist())
            p_ad.append(np.concatenate([frozen_adaptive(s, c_thr, mode), [0]]).tolist())
            p_pct.append(np.concatenate(
                [(s >= np.quantile(s, 1.0 - 0.0649)).astype(np.int8), [0]]).tolist())
        floor = score_dialogues(ys, [[0] * len(y) for y in ys])["score"]
        ad = score_dialogues(ys, p_ad)["score"]
        pct = score_dialogues(ys, p_pct)["score"]
        n_pred = int(sum(sum(x) for x in p_ad))
        n_gold = int(sum(sum(x) for x in ys))
        n_trans = int(sum(len(x) for x in ys))
        rows.append((name, ad, pct, floor, n_pred, n_gold, corp.n_docs, n_trans))
        print(f"{name:16s} 적응 {ad:.4f} · 분위수 {pct:.4f} · 하한 {floor:.4f} "
              f"· 예측 {n_pred} / 정답 {n_gold} / 전이 {n_trans} ({corp.n_docs}편)", flush=True)

    # ---- 프록시 회귀 (필수 사전 점검) -------------------------------------
    # 하한 대비 차이가 **예측 밀도 하나로** 설명되는가. R^2 >= 0.9 면 그 설명이 지배적이다.
    # n=5 라 결론용이 아니라 "배율이 원인" 같은 진술을 세울 수 있는지 확인용이다.
    delta = np.array([r[1] - r[3] for r in rows])
    proxies = {
        "배율 (예측/정답)": np.array([r[4] / max(r[5], 1) for r in rows]),
        "예측 밀도 (예측/전이)": np.array([r[4] / max(r[7], 1) for r in rows]),
        "정답 밀도 (정답/전이)": np.array([r[5] / max(r[7], 1) for r in rows]),
    }
    reg = {}
    for k, x in proxies.items():
        sl, ic = np.polyfit(x, delta, 1)
        r2 = 1.0 - ((delta - (sl * x + ic)) ** 2).sum() / ((delta - delta.mean()) ** 2).sum()
        reg[k] = (float(r2), float(np.corrcoef(x, delta)[0, 1]))
        print(f"  프록시 {k:22s} R^2 {r2:+.3f} · r {reg[k][1]:+.3f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    L = ["# 착수 전 확인 — 채택 체크포인트 + 적응 임계의 대화 세트 하한 통과 여부\n",
         f"- 체크포인트: `{cfg['checkpoint_origin']}` (양성 가중치 전축 1)",
         f"- 임계: `{cfg['threshold']['mode']}` c={c_thr} — μ + cσ. 동결 `hi_ontop_v2.adaptive_boundaries` 호출",
         "  (현재 표본 s_t 를 통계에 포함한 뒤 임계 계산. 미래는 쓰지 않는다.)",
         "- 분위수 열은 대조용(문서 내 상위 6.49%). 기존 전이 표가 쓰던 규칙.",
         "- 하네스 검증: dev3 적응 = `m5_default.json` 의 `dev3_seed0_this_checkpoint` 와 일치해야 한다.\n",
         "| 세트 | 적응 임계 | 분위수 | 무경계 하한 | 하한 대비 | 예측/정답/전이 |",
         "|---|---|---|---|---|---|"]
    for name, ad, pct, floor, npred, ngold, _, ntr in rows:
        d = ad - floor
        mark = "상회" if d >= 0.06 else ("**미달**" if d <= -0.06 else "*해석 불가*")
        L.append(f"| {name} | **{ad:.4f}** | {pct:.4f} | {floor:.4f} | "
                 f"{d:+.4f} {mark} | {npred}/{ngold}/{ntr} |")
    L += ["\n> 판정 표기는 **0.06 규칙**을 적용한다 — |차이| < 0.06 은 미달이 아니라 **해석 불가**다",
          "> (앵커 재현성 0.87 에서 유도된 기존 규칙).",
          "\n## 프록시 회귀 — 하한 대비 차이가 단일 변수로 설명되는가\n",
          "| 프록시 | R² | 상관 r |", "|---|---|---|"]
    for k, (r2, r) in reg.items():
        L.append(f"| {k} | {r2:+.3f} | {r:+.3f} |")
    L += ["\n> n=5 라 결론을 내리는 용도가 아니다. **R² 가 낮으면 '과예측 배율이 원인' 같은 단일 원인",
          "> 진술을 세울 수 없다**는 것까지가 이 회귀가 말하는 전부다.",
          "\n## 이 표의 한계 (판독 시 반드시 같이 읽을 것)\n",
          "- **표본이 작다.** superseg 정답 24개, ami 정답 11개. 정답 11개에서 나온 차이에",
          "  신뢰구간을 붙이면 대부분을 덮는다.",
          "- **FDB 예측력이 약하다.** 이 네 세트의 gold 는 **화제 경계**(세트당 11~49개)이고",
          "  Full-Duplex-Bench 의 gold 는 **턴 전환**이다. 밀도가 자릿수로 다르므로",
          "  화제 기준 과예측이 턴 기준으로는 과소일 수도 있다. **양방향으로 약해지는 관계**이며,",
          "  이 표를 음성 성능의 예측치로 쓰지 않는다.",
          "- 🔴 절대값을 성능으로 인용하지 않는다 — Δ·방향만. 대화는 학습 도메인 밖(D-M8)."]
    (OUT / "PRECHECK_adaptive_floor.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump([{"set": r[0], "adaptive": r[1], "percentile": r[2], "floor": r[3],
                "n_pred": r[4], "n_gold": r[5], "n_docs": r[6], "n_trans": r[7]} for r in rows],
              (OUT / "precheck_adaptive_floor.json").open("w"), ensure_ascii=False, indent=2)
    print("->", OUT / "PRECHECK_adaptive_floor.md")


if __name__ == "__main__":
    main()
