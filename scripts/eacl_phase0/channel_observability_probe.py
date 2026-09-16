#!/usr/bin/env python3
"""채널 관측가능성 진단 — M1 통제 · tiage CI · 축별 AUC.

판독 규칙은 **숫자를 보기 전에** 고정했다:
`outputs/experiments/2026-07-31_channel_observability/READOUT_RULES.md`

## 세 검정 (전부 기존 데이터·기존 체크포인트, 학습 없음)
1. **M1-CosDrift 통제** — 인접 문장 임베딩 코사인 거리. 경쟁자가 아니라 **과제 난이도계**다.
   M5 의 dialseg711 0.728 이 능력인지 과제가 자명한 건지 가른다.
2. **tiage 부트스트랩 CI** — 0.418 이 역방향인지 노이즈인지.
3. **축별 AUC** — D-M7 과 같은 형식. "대화에서 어느 축이 죽는가".

🔴 n=5. 가설 분해용이며 결론용이 아니다. 절대값을 성능으로 인용하지 않는다.
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
from hi_ontop.factored_slot import AXES, FactoredSlot  # noqa: E402
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402

CONFIG = REPO / "configs/m5_default.json"
SETS = ("dev3", "dts_tiage", "dts_dialseg711", "dts_superseg", "dts_ami")
OUT = REPO / "outputs/experiments/2026-07-31_channel_observability"
N_BOOT = 2000
SEED = 20260731
AXIS_KO = {"protagonist": "주인공", "space": "공간", "time": "시간",
           "causality": "인과", "intentionality": "의도"}


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def boot_ci(vals: list[float], rng: np.random.Generator) -> tuple[float, float]:
    """문서 수준 부트스트랩 95% CI. `vals` = 문서별 값(NaN 제외 후)."""
    v = np.asarray([x for x in vals if not np.isnan(x)], float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    means = v[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(REPO / cfg["checkpoint"], map_location=dev, weights_only=False)
    m = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"],
                     pos_weight=ck["pos_weight"], dynamics=ck.get("dynamics"),
                     d1=ck["args"].get("input_path", "axis_extractor") == "axis_extractor",
                     shared_dim=ck["args"].get("shared_proj_dim") or None,
                     pred_space=ck["args"].get("pred_space", "obs")).to(dev)
    m.load_state_dict(ck["state"]); m.eval()
    rng = np.random.default_rng(SEED)

    res = {}
    for name in SETS:
        corp = Corpus(name)
        idx = list(range(corp.n_docs))
        probs = axis_probs(m, corp, idx, dev, bs=8, with_anchor=False)
        emb = np.asarray(corp.emb)                       # [N_sent, 768] L2 정규화됨
        a5, a1, adiff, ax = [], [], [], {k: [] for k in AXES}
        for i, pr in zip(idx, probs):
            g = corp.gold[int(corp.to[i]):int(corp.to[i + 1])].astype(np.int8)
            s0, s1 = int(corp.so[i]), int(corp.so[i + 1])
            e = emb[s0:s1]
            # M1-CosDrift: 인접 문장 코사인 거리(= 1 - 내적, L2 정규화 가정). 학습 없음.
            m1 = 1.0 - (e[:-1] * e[1:]).sum(1)
            s5 = pr.sum(1)
            assert len(m1) == len(g) == len(s5), (len(m1), len(g), len(s5))
            v5, v1 = auc(g, s5), auc(g, m1)
            a5.append(v5); a1.append(v1); adiff.append(v5 - v1)
            for k, axname in enumerate(AXES):
                ax[axname].append(auc(g, pr[:, k]))
        res[name] = {
            "n_docs": corp.n_docs,
            "m5": float(np.nanmean(a5)), "m5_ci": boot_ci(a5, rng),
            "m1": float(np.nanmean(a1)), "m1_ci": boot_ci(a1, rng),
            "diff": float(np.nanmean(adiff)), "diff_ci": boot_ci(adiff, rng),
            "axes": {k: {"auc": float(np.nanmean(v)), "ci": boot_ci(v, rng)}
                     for k, v in ax.items()},
        }
        r = res[name]
        print(f"{name:16s} M5 {r['m5']:.3f} [{r['m5_ci'][0]:.3f},{r['m5_ci'][1]:.3f}] · "
              f"M1 {r['m1']:.3f} [{r['m1_ci'][0]:.3f},{r['m1_ci'][1]:.3f}] · "
              f"차 {r['diff']:+.3f} [{r['diff_ci'][0]:+.3f},{r['diff_ci'][1]:+.3f}]", flush=True)

    _write(res)


def _covers(ci: tuple[float, float], x: float) -> bool:
    return ci[0] <= x <= ci[1]


def _write(res: dict) -> None:
    L = ["# 채널 관측가능성 진단 — M1 통제 · 부트스트랩 CI · 축별 AUC\n",
         "판독 규칙은 **숫자를 보기 전에** 고정했다 → `READOUT_RULES.md`",
         f"문서 수준 부트스트랩 {N_BOOT}회, 시드 {SEED}. 채택 체크포인트, 학습 없음.\n",
         "## 1. M1-CosDrift 통제 — 과제 난이도계\n",
         "| 세트 | 문서 | M5 신호 AUC | M1-CosDrift AUC | M5 − M1 [95% CI] |",
         "|---|---|---|---|---|"]
    for s in SETS:
        r = res[s]
        L.append(f"| {s} | {r['n_docs']} | {r['m5']:.3f} [{r['m5_ci'][0]:.3f}, {r['m5_ci'][1]:.3f}] | "
                 f"{r['m1']:.3f} [{r['m1_ci'][0]:.3f}, {r['m1_ci'][1]:.3f}] | "
                 f"**{r['diff']:+.3f}** [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}] |")

    # 사전 고정된 A·B·C 판독
    A = _covers(res["dts_dialseg711"]["diff_ci"], 0.0)
    B = not _covers(res["dev3"]["diff_ci"], 0.0) and res["dev3"]["diff"] > 0
    C = all(_covers(res[s]["m5_ci"], 0.5) and _covers(res[s]["m1_ci"], 0.5)
            for s in ("dts_superseg", "dts_tiage"))
    L += ["\n### 사전 고정 판독 (READOUT_RULES §1)\n",
          "| # | 조건 | 성립? | 판독 |", "|---|---|---|---|",
          f"| A | dialseg711 에서 M5 − M1 ≈ 0 | {'**성립**' if A else '불성립'} | "
          f"{'0.728 은 과제가 쉬운 것 — 인공 경계 증거 소멸' if A else 'M5 가 M1 을 넘음 — 과제 난이도만으로 설명 안 됨'} |",
          f"| B | dev-3 에서 M5 − M1 > 0 | {'**성립**' if B else '불성립'} | "
          f"{'서사에서 요인화가 실제로 기여' if B else '서사에서도 M1 대비 우위 미확인'} |",
          f"| C | superseg·tiage 에서 M5·M1 둘 다 0.5 근처 | {'**성립**' if C else '불성립'} | "
          f"{'대화 채널 실패 확정' if C else '둘 중 하나 이상이 우연과 구별됨'} |",
          f"\n**A·B·C 모두 성립 = {A and B and C}** → "
          + ("채널 가설이 닫힌다. 초안의 중심 주장이 된다."
             if (A and B and C) else
             "**단일 설명을 주장하지 않는다.** 초안 중심 주장 자리에 "
             "\"다섯 채널에서 측정했고 단일 설명은 찾지 못했다\"가 들어간다."),
          "\n## 2. tiage — 역방향인가 노이즈인가\n"]
    t = res["dts_tiage"]
    cov = _covers(t["m5_ci"], 0.5)
    L += [f"M5 신호 AUC **{t['m5']:.3f}** [{t['m5_ci'][0]:.3f}, {t['m5_ci'][1]:.3f}] (문서 {t['n_docs']}편)\n",
          f"- CI 가 0.5를 **{'덮는다' if cov else '덮지 않는다'}** ⇒ "
          + ("**노이즈다. 중심 주장에서 뺀다.**" if cov else
             "**역방향 자체가 발견이다** — 자연스러운 화제 이동 시점에 리셋 확률이 오히려 낮다. "
             "\"신호가 없다\"보다 강한 진술이며 조작화가 무엇을 잘못 잡는지에 대한 단서다."),
          "\n## 3. 축별 AUC — 어느 축이 어느 채널에서 죽는가\n",
          "D-M7(극 대사, SHARE 25편)과 같은 형식. 마지막 열이 그 측정이다.\n",
          "| 축 | " + " | ".join(SETS) + " | 극 대사 (D-M7) |",
          "|---|" + "---|" * (len(SETS) + 1)]
    dm7 = {"protagonist": 0.498, "space": 0.527, "time": 0.507,
           "causality": 0.541, "intentionality": 0.529}
    for a in AXES:
        row = " | ".join(f"{res[s]['axes'][a]['auc']:.3f}" for s in SETS)
        L.append(f"| {AXIS_KO[a]} | {row} | {dm7[a]:.3f} |")
    L.append("| **합 (M5 신호)** | " + " | ".join(f"**{res[s]['m5']:.3f}**" for s in SETS) + " | 0.568 |")

    L += ["\n> 축별 실패 유형(기존 분류): **포화**(인과, 발화율 0.695) · "
          "**관측 불가**(공간 실제 83% 변경에 recall 4%, 시간 recall 0%) · **구조적**(주인공, 화자 고정).",
          "\n---\n",
          "> 🔴 **n=5 다.** 회귀 R² 도 AUC 순위도 다섯 점 위에서 나온 것이다.",
          "> 이 표들은 **가설 분해용이며 결론용이 아니다.** 절대값을 성능으로 인용하지 않는다."]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PROBE_channel_observability.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(res, (OUT / "probe_channel_observability.json").open("w"),
              ensure_ascii=False, indent=2)
    print(f"\n사전 고정 판독 — A {A} · B {B} · C {C} · 전부성립 {A and B and C}")
    print("->", OUT / "PROBE_channel_observability.md")


if __name__ == "__main__":
    main()
