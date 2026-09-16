#!/usr/bin/env python3
"""채널 관측가능성 — **정정 통합본** (주지표 = deployable DTS Score, AUC 는 보조).

## 앞선 산출물의 오류 둘 (이 스크립트가 고치는 것)
1. **주지표를 AUC 로 냈다.** 프로젝트 규칙은 Score 가 주지표이고 AUC 는 보조다(CLAUDE.md).
   → Score 를 헤드라인으로, AUC 를 보조로 내린다.
2. **ami 를 DTS 채점기로 쟀다.** AMI 는 전용 확정 채점기가 있다(`hi_ontop.ami_scoring`,
   **±2 tolerant F1**). exact F1 은 M5·M1 만 일방적으로 깎는다(무경계 하한은 F1=0 이라 영향 없음).
   → 세트별로 올바른 채점기로 분기한다.

## 표본 경고
`dts_ami` 는 **회의 2편·gold 11개**다. 주 분석은 **ami 를 뺀 4세트**로 하고 ami 는 따로 보고한다.

🔴 절대값을 성능으로 인용하지 않는다 — Δ·방향만. 대화는 학습 도메인 밖(D-M8).
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
from hi_ontop.ami_scoring import score_meetings  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.factored_slot import AXES, FactoredSlot  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402  🔒 동결
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402

CONFIG = REPO / "configs/m5_default.json"
SETS = ("dev3", "dts_tiage", "dts_dialseg711", "dts_superseg", "dts_ami")
PRIMARY = ("dev3", "dts_tiage", "dts_dialseg711", "dts_superseg")   # ami 제외 = 주 분석
AMI_SETS = {"dts_ami"}
OUT = REPO / "outputs/experiments/2026-07-31_channel_observability"
N_BOOT, SEED = 2000, 20260731
AXIS_KO = {"protagonist": "주인공", "space": "공간", "time": "시간",
           "causality": "인과", "intentionality": "의도"}
DM7 = {"protagonist": 0.498, "space": 0.527, "time": 0.507,
       "causality": 0.541, "intentionality": 0.529}


def score_set(name: str, golds: list[np.ndarray], preds: list[np.ndarray]) -> float:
    """세트별 **올바른** 채점기로 분기. ami = ±2 tolerant F1 + 시작-turn, 나머지 = DTS."""
    if name in AMI_SETS:
        g = [np.concatenate([[0], x]).tolist() for x in golds]
        p = [np.concatenate([[0], x]).tolist() for x in preds]
        return score_meetings(g, p)["score"]
    g = [np.concatenate([x, [0]]).tolist() for x in golds]
    p = [np.concatenate([x, [0]]).tolist() for x in preds]
    return score_dialogues(g, p)["score"]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def boot_ci(vals: list[float], rng) -> tuple[float, float]:
    v = np.asarray([x for x in vals if not np.isnan(x)], float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    means = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    c_thr, mode = cfg["threshold"]["c"], cfg["threshold"]["mode"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(REPO / cfg["checkpoint"], map_location=dev, weights_only=False)
    m = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"],
                     pos_weight=ck["pos_weight"], dynamics=ck.get("dynamics"),
                     d1=ck["args"].get("input_path", "axis_extractor") == "axis_extractor",
                     shared_dim=ck["args"].get("shared_proj_dim") or None,
                     pred_space=ck["args"].get("pred_space", "obs")).to(dev)
    m.load_state_dict(ck["state"]); m.eval()
    rng = np.random.default_rng(SEED)

    def ad(s: np.ndarray) -> np.ndarray:
        n = len(s)
        return np.asarray(adaptive_boundaries(np.concatenate([[0.0], s]).tolist(),
                                              c=c_thr, mode=mode), np.int8)[:n]

    res = {}
    for name in SETS:
        c = Corpus(name)
        idx = list(range(c.n_docs))
        probs = axis_probs(m, c, idx, dev, bs=8, with_anchor=False)
        emb = np.asarray(c.emb)
        G, B5, B1, Z = [], [], [], []
        a5, a1, adf, ax = [], [], [], {k: [] for k in AXES}
        for i, pr in zip(idx, probs):
            g = c.gold[int(c.to[i]):int(c.to[i + 1])].astype(np.int8)
            e = emb[int(c.so[i]):int(c.so[i + 1])]
            s5 = pr.sum(1)
            m1 = 1.0 - (e[:-1] * e[1:]).sum(1)      # M1-CosDrift, 학습 없음
            G.append(g); B5.append(ad(s5)); B1.append(ad(m1)); Z.append(np.zeros(len(g), np.int8))
            v5, v1 = auc(g, s5), auc(g, m1)
            a5.append(v5); a1.append(v1); adf.append(v5 - v1)
            for k, a in enumerate(AXES):
                ax[a].append(auc(g, pr[:, k]))
        res[name] = {
            "n_docs": c.n_docs, "n_trans": int(sum(len(x) for x in G)),
            "n_gold": int(sum(x.sum() for x in G)),
            "scorer": "AMI 공식 (±2 tolerant F1)" if name in AMI_SETS else "DTS 공식 (exact F1)",
            "score_m5": score_set(name, G, B5),
            "score_m1": score_set(name, G, B1),
            "score_floor": score_set(name, G, Z),
            "n_pred_m5": int(sum(x.sum() for x in B5)),
            "n_pred_m1": int(sum(x.sum() for x in B1)),
            "auc_m5": float(np.nanmean(a5)), "auc_m5_ci": boot_ci(a5, rng),
            "auc_m1": float(np.nanmean(a1)), "auc_m1_ci": boot_ci(a1, rng),
            "auc_diff": float(np.nanmean(adf)), "auc_diff_ci": boot_ci(adf, rng),
            "axes": {k: float(np.nanmean(v)) for k, v in ax.items()},
        }
        r = res[name]
        r["d_floor"] = r["score_m5"] - r["score_floor"]
        r["d_m1"] = r["score_m5"] - r["score_m1"]
        print(f"{name:16s} Score M5 {r['score_m5']:.4f} · M1 {r['score_m1']:.4f} · "
              f"하한 {r['score_floor']:.4f} | M5−하한 {r['d_floor']:+.4f} · M5−M1 {r['d_m1']:+.4f}",
              flush=True)
    _write(res)


def _verdict(d: float) -> str:
    return "상회" if d >= 0.06 else ("**미달**" if d <= -0.06 else "*해석 불가*")


def _write(res: dict) -> None:
    L = ["# REPORT — 채널 관측가능성 (M5 대 M1-CosDrift), 정정 통합본\n",
         "**주지표 = deployable DTS Score. AUC 는 보조다.**",
         "판독 규칙은 숫자를 보기 전에 고정했다 → `READOUT_RULES.md`\n",
         "## 실험 setup\n",
         "| 항목 | 값 |", "|---|---|",
         "| 목적 | 대화 세트에서 M5 가 무경계 하한 아래인 원인. 신호인가 임계인가 채널인가 |",
         "| 모델 | 채택 체크포인트 `M5_per_slot_slot_pwone_seed0.pt` (양성 가중치 전축 1). **학습 없음** |",
         "| 임계 | 동결 `online_nf` c=1.0 — `hi_ontop_v2.adaptive_boundaries` 직접 호출(재구현 금지) |",
         "| 대조 | **M1-CosDrift** = 인접 문장 임베딩 코사인 거리. 경쟁자가 아니라 **과제 난이도계**. 같은 임계를 태운다 |",
         "| 하한 | 무경계(전 시점 0) |",
         "| 채점 | 세트별 분기 — ami = `hi_ontop.ami_scoring`(**±2 tolerant F1**, 시작-turn), 나머지 = `hi_ontop.dts_scoring`(exact F1) |",
         "| 해석 문턱 | \\|Δ\\| < 0.06 은 **해석 불가**(앵커 재현성 0.87 에서 유도) |",
         "",
         "## 🔴 앞선 산출물 정정 둘\n",
         "1. **주지표를 AUC 로 냈던 것을 Score 로 되돌린다.** 프로젝트 규칙상 AUC 는 headline 이 될 수 없다.",
         "2. **ami 를 DTS 채점기(exact F1)로 쟀던 것을 AMI 공식 채점기(±2 tolerant)로 고친다.**",
         "   exact F1 은 M5·M1 만 깎고 무경계 하한(F1=0)에는 영향이 없어 **일방적 과소평가**였다.",
         "   ⇒ `PRECHECK_adaptive_floor.md` · `DIAG_dialogue_floor_cause.md` ·",
         "   `PROBE_channel_observability.md` 의 **ami 행과 AUC 헤드라인은 이 문서로 대체된다.**",
         "",
         "## 결과 — 주지표 (deployable DTS Score)\n",
         "| 세트 | 문서 | gold | M5 | M1-CosDrift | 무경계 하한 | M5−하한 | M5−M1 | 예측 M5/M1 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        r = res[s]
        tag = " ⚠️" if s in AMI_SETS else ""
        L.append(f"| {s}{tag} | {r['n_docs']} | {r['n_gold']} | **{r['score_m5']:.4f}** | "
                 f"{r['score_m1']:.4f} | {r['score_floor']:.4f} | "
                 f"{r['d_floor']:+.4f} {_verdict(r['d_floor'])} | "
                 f"{r['d_m1']:+.4f} {_verdict(r['d_m1'])} | {r['n_pred_m5']}/{r['n_pred_m1']} |")
    L += ["\n⚠️ `dts_ami` = **회의 2편 · gold 11개**. 주 분석에서 제외하고 아래에 따로 적는다.\n",
          "### 주 분석 (ami 제외 4세트)\n"]
    prim = [res[s] for s in PRIMARY]
    n_over = sum(1 for r in prim if r["d_floor"] >= 0.06)
    n_und = sum(1 for r in prim if r["d_floor"] <= -0.06)
    n_amb = len(prim) - n_over - n_und
    L += [f"- **무경계 하한 대비: {n_over} 상회 / {n_amb} 해석 불가 / {n_und} 미달**",
          f"- **M1 대비 M5 가 유의하게 앞선 세트: "
          f"{sum(1 for r in prim if r['d_m1'] >= 0.06)}개** "
          f"(뒤진 세트 {sum(1 for r in prim if r['d_m1'] <= -0.06)}개)",
          "\n### ami (검정력 부족, 참고)\n",
          f"- 정정 후 M5 {res['dts_ami']['score_m5']:.4f} · 하한 {res['dts_ami']['score_floor']:.4f} "
          f"→ **{res['dts_ami']['d_floor']:+.4f} {_verdict(res['dts_ami']['d_floor'])}**",
          "- 정정 전(DTS exact F1) 값은 0.1959 대 0.3244 = −0.1285 '미달'이었다. **그 판정은 철회한다.**",
          "\n## 보조 — 신호 순위력 (AUC, 임계 무관)\n",
          "임계 규칙을 걷어낸 순위력. **하한 미달이 신호 탓인지 임계 탓인지** 가르는 용도로만 쓴다.\n",
          "| 세트 | M5 신호 | M1-CosDrift | M5 − M1 [95% CI] |", "|---|---|---|---|"]
    for s in SETS:
        r = res[s]
        L.append(f"| {s} | {r['auc_m5']:.3f} [{r['auc_m5_ci'][0]:.3f}, {r['auc_m5_ci'][1]:.3f}] | "
                 f"{r['auc_m1']:.3f} [{r['auc_m1_ci'][0]:.3f}, {r['auc_m1_ci'][1]:.3f}] | "
                 f"{r['auc_diff']:+.3f} [{r['auc_diff_ci'][0]:+.3f}, {r['auc_diff_ci'][1]:+.3f}] |")

    L += ["\n## 보조 — 축별 AUC (D-M7 과 같은 형식)\n",
          "| 축 | " + " | ".join(SETS) + " | 극 대사 (D-M7) |",
          "|---|" + "---|" * (len(SETS) + 1)]
    for a in AXES:
        L.append(f"| {AXIS_KO[a]} | " + " | ".join(f"{res[s]['axes'][a]:.3f}" for s in SETS)
                 + f" | {DM7[a]:.3f} |")
    L.append("| **합 (M5 신호)** | " + " | ".join(f"**{res[s]['auc_m5']:.3f}**" for s in SETS)
             + " | 0.568 |")

    L += ["\n> 축별 실패 유형(기존 분류): **포화**(인과, 발화율 0.695) · "
          "**관측 불가**(공간 실제 83% 변경에 recall 4%, 시간 recall 0%) · **구조적**(주인공, 화자 고정).",
          "\n## 한계 / 검증 미해결\n",
          "- **n=4(주 분석) · n=5(전체).** 회귀도 순위도 그 위에서 나온다. **가설 분해용이며 결론용이 아니다.**",
          "- **ami 회의 2편·gold 11개.** 어떤 판정도 세우지 않는다.",
          "- **ami 정렬 미해결.** gold 를 −1 밀면 M5 AUC 가 0.630 [0.518, 0.731] → 0.799 [0.662, 0.902] 로",
          "  오른다. 빌더(`build_ami_pack.py`)는 전이 s ↔ turn s−1|s 로 **정렬이 맞음을 확증**했고",
          "  DTS 빌더도 동일 인덱싱이다. 양성 11개로는 실재하는 지연인지 잡음인지 못 가른다.",
          "  실재한다면 \"회의에서 화제 전환을 한 턴 늦게 잡는다\"가 되나 **주장하지 않는다.**",
          "- 대화 세트는 전부 **학습 도메인 밖**이다(D-M8: 영어 서사 픽션 단일).",
          "- 절대값을 성능으로 인용하지 않는다 — Δ·방향만."]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(res, (OUT / "report_channel_observability.json").open("w"),
              ensure_ascii=False, indent=2, default=float)
    print("\n->", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
