#!/usr/bin/env python3
"""M4/M5 적응 임계 평가 — 고정 밀도 사전(6.49%)을 문서 적응 임계로 대체.

## 왜
지금까지 M4/M5 경계 규칙은 **문서 내 상위 6.49% 고정**이었다. 그 6.49% 는 앵커 코퍼스에서
`count ≥ t*=3` 으로 만든 경계율이므로 **정수 임계 t\\* 가 밀도 사전에 그대로 녹아 있다**
(D-B16 이 지목한 "정수 임계 병목"이 이 경로로 살아 있었다).

그런데 gold 밀도는 문서마다 **3.6% ~ 19%** 로 5배 넘게 다르다(pieman 0.19 · tunnel 0.036 ·
monkey 0.074). 고정 밀도로는 원리적으로 못 맞춘다.
그리고 프로젝트 자신의 기존 DTS 실험(`2026-06-13_dts_official_rescore`)도 **데이터셋마다 다른
percentile**(p60/p70/p80)을 썼다 — 고정이 아니었다.

## 무엇을
**이미 구현된** `hi_ontop.hi_ontop_v2.adaptive_boundaries` 를 재사용한다(새로 만들지 않음):
| 모드 | 임계 | 성질 |
|---|---|---|
| `otsu` | 히스토그램 이분산 최대화 | **파라미터 없음**, 오프라인 |
| `offline` | μ + c·σ (문서 전체) | c 하나, 오프라인 |
| `online` | μ + c·σ (누적, 인과적) | **미래 미사용 = 실시간 배포 가능** |
| 고정 percentile | 상위 r 분위수 | 현행(비교 기준) |

🔴 deployable 규약: 넷 다 **gold 개수를 쓰지 않는다.** `online` 은 미래 발화도 안 본다.
🔴 지위: **탐색적**(dev-3 = 연습장). c 는 dev-3 에서 고르므로 **확증 근거가 아니다** —
확정 패스에서는 c 를 앵커 코퍼스에서 정해 동결해야 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.factored_slot import FactoredSlot  # noqa: E402
from hi_ontop.hi_ontop_v2 import _otsu_threshold, adaptive_boundaries  # noqa: E402
from train_anchor_m4m5 import EXP, T_STAR, Corpus, axis_probs  # noqa: E402

ANCHOR_RATE = 0.0649
SETS = ("dev3", "dts_full_tiage", "dts_full_dialseg711", "dts_full_superseg")
MODELS = [("M5", "_per_slot_slot", 0), ("M4", "_per_slot_slot", 0)]
C_GRID = (0.5, 1.0, 1.5, 2.0)


def rules(s: np.ndarray, c: float) -> dict[str, np.ndarray]:
    """전이별 연속 점수 s -> 규칙별 경계. 전부 gold 미사용."""
    seq = np.concatenate([[0.0], s])                      # adaptive_boundaries 규약(0번 sentinel)
    out = {"고정 6.49%": (s >= np.quantile(s, 1 - ANCHOR_RATE)).astype(np.int8),
           "otsu": np.asarray(adaptive_boundaries(seq.tolist(), mode="otsu"), np.int8),
           f"offline μ+{c}σ": np.asarray(adaptive_boundaries(seq.tolist(), c=c, mode="offline"), np.int8),
           f"online μ+{c}σ": np.asarray(adaptive_boundaries(seq.tolist(), c=c, mode="online_nf"), np.int8)}
    return {k: v[:len(s)] for k, v in out.items()}


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corp = {s: Corpus(s) for s in SETS}
    res: dict = {}
    for tag, ns, seed in MODELS:
        ck = torch.load(EXP / f"{tag}{ns}_seed{seed}.pt", map_location=dev, weights_only=False)
        m = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"],
                         pos_weight=ck["pos_weight"], dynamics=ck.get("dynamics"),
                         pred_space=ck["args"].get("pred_space", "obs")).to(dev)
        m.load_state_dict(ck["state"])
        for sname in SETS:
            c_ = corp[sname]
            probs = axis_probs(m, c_, list(range(c_.n_docs)), dev, bs=8, with_anchor=False)
            for cval in C_GRID:
                acc: dict[str, list] = {}
                ys = []
                npred: dict[str, int] = {}
                for i, pr in enumerate(probs):
                    g = c_.gold[int(c_.to[i]):int(c_.to[i + 1])].astype(np.int8)
                    ys.append(np.concatenate([g, [0]]).tolist())
                    for k, b in rules(pr.sum(1), cval).items():
                        acc.setdefault(k, []).append(np.concatenate([b, [0]]).tolist())
                        npred[k] = npred.get(k, 0) + int(b.sum())
                for k, ps in acc.items():
                    if "μ" not in k and cval != C_GRID[0]:
                        continue                          # c 무관 규칙은 한 번만
                    res[(tag, sname, k)] = (score_dialogues(ys, ps)["score"], npred[k])
            print(f"  [{tag}] {sname} 완료", flush=True)

    hdr = ["dev-3(서사)", "tiage(100)", "dialseg711(711)", "superseg(1322)"]
    L = ["# M4/M5 적응 임계 평가 — 고정 밀도 사전 vs 문서 적응\n",
         "> 🔴 **탐색적**(dev-3 = 연습장). c 를 dev-3 에서 고르면 확증 근거가 아니다 —"
         " 확정 패스에서는 **앵커 코퍼스에서 c 를 정해 동결**해야 한다.",
         "> ✅ DTS = **SuperDialseg 공식 test 전체**(tiage 100 / dialseg711 711 / superseg 1322). D-B24 의 부분집합 문제 해소.",
         "> ⚠️ 단 인코더가 다르다 — 공식 수치는 `multi-qa-mpnet`, 본 파이프라인은"
         " `paraphrase-multilingual-mpnet`. 절대값 직접 비교 시 이 차이를 명시할 것.",
         "> 전 규칙이 **gold 개수 미사용**. `online` 은 **미래도 안 본다**(실시간 배포 가능).\n",
         "\n## 고정 6.49% 가 t\\* 를 물려받는 경로\n",
         "앵커 코퍼스에서 `count ≥ t*=3` → 경계율 **0.0649** → 그 값을 밀도 사전으로 사용."
         " 즉 정수 임계를 없앤 게 아니라 **그 임계가 만든 밀도를 물려받았다**.",
         "gold 실제 밀도는 문서마다 **0.036~0.19**(5배 차이) — 고정으로는 원리적으로 못 맞춘다.\n"]
    for tag, _, _ in MODELS:
        L += [f"\n## {tag} (slot-native, seed 0)\n",
              "| 규칙 | " + " | ".join(hdr) + " |", "|---|" + "---|" * len(hdr)]
        keys = [k for (t, s, k) in res if t == tag and s == "dev3"]
        seen = set()
        for k in keys:
            if k in seen:
                continue
            seen.add(k)
            row = []
            for s in SETS:
                v = res.get((tag, s, k))
                row.append(f"{v[0]:.4f} ({v[1]})" if v else "—")
            L.append(f"| {k} | " + " | ".join(row) + " |")
        none_row = []
        for s in SETS:
            c_ = corp[s]
            ys = [np.concatenate([c_.gold[int(c_.to[i]):int(c_.to[i+1])], [0]]).tolist()
                  for i in range(c_.n_docs)]
            none_row.append(score_dialogues(ys, [[0] * len(y) for y in ys])["score"])
        L.append("| **경계 없음 (하한)** | " + " | ".join(f"**{v:.4f}**" for v in none_row) + " |")
        L.append("| *gold 경계 수* | " + " | ".join(
            f"*{int(corp[s].gold.sum())}*" for s in SETS) + " |")
    L += ["\n> 괄호 안 = 예측한 경계 개수. gold 개수와 대조하면 과/과소 분절이 보인다.",
          "\n⚠️ seed 1개(slot-native). CLAUDE.md 3-run 요건 미충족."]
    p = REPO / "outputs/experiments/m4m5_phase1_anchor/ADAPTIVE_THRESHOLD.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
