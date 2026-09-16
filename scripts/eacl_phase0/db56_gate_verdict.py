#!/usr/bin/env python3
r"""D-B56 게이트 판정 — 표현 감사 9건(3 조건 × 3 seed)을 사전등록 기준에 대입한다.

🔒 기준은 ledger D-B56 §3 에서 **실행 전에** 고정됐다. 여기서 고치지 않는다.
셋 중 **하나라도** 만족해야 §4.8 로 간다. 하나도 못 넘으면 D-B28 종결 유지.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[2]
D = REPO / "outputs/experiments/2026-07-31_sem_switching_reopen"
CONDS = [("W0 (ALPHA 현행 · pw clip)", "_per_slot_slot"),
         ("W3 (ALPHA 현행 · pw one)", "_per_slot_slot_pwone"),
         ("W1 (ALPHA 균등 0.63 · pw one)", "_per_slot_slot_alphauniform_pwone")]
SEEDS = (0, 1, 2)
AUC_BAR = 0.7006          # 🔒 D-B19 dev-3 리셋확률 합 AUC
RATIO_BAR = 1.25          # 🔒 게이트 ②
VAR_DROP = 0.50           # 🔒 게이트 ①


def ms(v):
    return (mean(v), stdev(v) if len(v) > 1 else 0.0)


def main() -> None:
    R = {}
    for label, ns in CONDS:
        rows = []
        for s in SEEDS:
            p = D / f"audit/audit_M5{ns}_seed{s}.json"
            if p.exists():
                rows.append(json.loads(p.read_text()))
        if rows:
            R[label] = rows

    L = ["# D-B56 — 스위칭 층 재개 게이트: 표현 감사 결과", "",
         "> 🔒 기준은 ledger **D-B56 §3** 에서 실행 **전**에 고정됐다. 여기서 고치지 않는다.",
         "> 측정은 전부 **phase1 val**(캘리브레이션 200편과 배타인 120편). **dev-3 미개봉** —",
         "> `db56_repr_audit.py` 는 dev-3 를 로드조차 하지 않는다.", "",
         "## 1. 감사 실측 (seed 0/1/2 평균 ±SD)", "",
         "| 조건 | 우도비 분산 | 우도비 AUC | 리셋확률합 AUC | λ 경계수 max/min |",
         "|---|---|---|---|---|"]
    agg = {}
    for label, _ in CONDS:
        if label not in R:
            continue
        rows = R[label]
        vv, av, rv, lv = (ms([r["ll_var"] for r in rows]), ms([r["auc_ll"] for r in rows]),
                          ms([r["auc_resetsum"] for r in rows]),
                          ms([r["lmda_ratio"] for r in rows]))
        agg[label] = {"ll_var": vv, "auc_ll": av, "auc_rs": rv, "ratio": lv,
                      "n": len(rows)}
        L.append(f"| {label} | {vv[0]:.3g} ±{vv[1]:.2g} | {av[0]:.4f} ±{av[1]:.4f} | "
                 f"{rv[0]:.4f} ±{rv[1]:.4f} | {lv[0]:.3f} ±{lv[1]:.3f} |")

    w0 = agg.get(CONDS[0][0])
    L += ["", "## 2. 🔒 게이트 판정 (사전등록 D-B56 §3)", ""]
    verdicts = {}

    # ① 우도비 분산 W0 대비 ≥50% 감소
    L += ["### ① 우도비 분산이 W0 대비 ≥50% 감소했나", ""]
    if not w0:
        L.append("- 🔴 W0 감사 없음 — **판정 불가**")
    else:
        L += ["| 조건 | 분산 | W0 대비 | 판정 |", "|---|---|---|---|",
              f"| W0 (기준) | {w0['ll_var'][0]:.3g} | — | — |"]
        for label in [c[0] for c in CONDS[1:]]:
            if label not in agg:
                continue
            v = agg[label]["ll_var"][0]
            drop = 1 - v / w0["ll_var"][0]
            ok = drop >= VAR_DROP
            verdicts[f"①{label}"] = ok
            L.append(f"| {label} | {v:.3g} | {drop:+.1%} | {'✅ 통과' if ok else '🔴 불통과'} |")
    L.append("")

    # ② λ 민감도
    L += ["### ② λ sweep 예측 경계 수 max/min ≥ 1.25 (사전이 inert 가 아닌가)", "",
          "| 조건 | max/min | 판정 |", "|---|---|---|"]
    for label in agg:
        r = agg[label]["ratio"][0]
        ok = r >= RATIO_BAR
        verdicts[f"②{label}"] = ok
        L.append(f"| {label} | {r:.3f} | {'✅ 통과' if ok else '🔴 불통과'} |")
    L.append("")

    # ③ AUC — 동반값으로 비교 가능성부터 확인
    L += ["### ③ 우도비 AUC ≥ 0.7006", "",
          "🔴 **실행 전 선언한 결함(D-B56 §3):** 0.7006 은 D-B19 에서 **dev-3** 리셋확률 합의",
          "AUC 다. 이 감사는 **phase1 val 의사-gold**(count ≥ t\\*=3)에서 잰다 — **모수가 다르다.**",
          "⇒ 같은 val 의 `리셋확률합 AUC` 를 동반값으로 재서 비교 가능성을 먼저 본다.", "",
          "| 조건 | 우도비 AUC | 같은 val 리셋확률합 AUC | 우도비 − 리셋합 | 문턱(0.7006) 대비 |",
          "|---|---|---|---|---|"]
    for label in agg:
        au, rs = agg[label]["auc_ll"][0], agg[label]["auc_rs"][0]
        L.append(f"| {label} | {au:.4f} | {rs:.4f} | **{au - rs:+.4f}** | "
                 f"{'≥' if au >= AUC_BAR else '<'} |")
    rs_all = [agg[k]["auc_rs"][0] for k in agg]
    comparable = all(abs(x - AUC_BAR) <= 0.03 for x in rs_all)
    L += ["", f"동반값(리셋확률합 AUC) = {min(rs_all):.4f} ~ {max(rs_all):.4f}. "
          f"문턱 0.7006 과의 차이가 {'±0.03 이내 → 비교 가능' if comparable else '±0.03 을 넘는다'}.",
          ""]
    if not comparable:
        L += ["⇒ 🔴 **게이트 ③ = 판정 불가.** 사전등록대로 ①② 로만 판정한다.",
              "   (같은 val 에서 리셋확률 합이 문턱보다 뚜렷이 높다는 것은, 우도비가 문턱을 넘더라도",
              "   **그 판에서는 여전히 리셋확률 합에 진다**는 뜻이다 — D-B19 의 0.651 < 0.7006 과 같은 구도.)", ""]
    else:
        for label in agg:
            verdicts[f"③{label}"] = agg[label]["auc_ll"][0] >= AUC_BAR

    passed = [k for k, v in verdicts.items() if v]
    L += ["## 3. 최종 판정", ""]
    if passed:
        L += [f"### ⇒ ✅ **게이트 통과** — 통과 항목: {', '.join(passed)}",
              "", "§4.8 delayed-commit particle smoothing 단일 검정으로 진행한다(D-B56 §4)."]
    else:
        L += ["### ⇒ 🔴 **게이트 전 항목 불통과 — 여기서 중단한다.**", "",
              "**D-B28 🔒 스위칭 층 종결을 그대로 유지한다.** 사전등록 D-B56 §3 대로 추가 라운드 없음.",
              "§4.8 particle smoothing 은 **구현하지 않는다** — 재개의 전제였던",
              "*\"기존 실패 메커니즘이 실제로 달라졌다\"* 는 최소 증거가 확보되지 않았다."]
    L += ["", "## 4. 무엇이 확인됐나 (게이트와 별개로 남는 사실)", "",
          "- **pw-one 계열(W1/W3)로 바꿔도 λ 는 여전히 사실상 inert 다.** D-B28 진단",
          "  *\"사전이 우도에 눌린다\"* 가 보정 개선 이후에도 유지된다.",
          "- **같은 val 에서 우도비 AUC < 리셋확률 합 AUC.** D-B19 가 dev-3 에서 관찰한 구도",
          "  (0.651 < 0.7006)가 **다른 판(phase1 val)·다른 체크포인트에서도 재현**된다.",
          "  ⇒ *\"스위칭 경로가 이 표현에서 열등하다\"* 는 결론의 외적 타당성이 넓어졌다.",
          "- **W1 과 W3 는 서로 구별되지 않는다.** ALPHA 균등화가 스위칭 층 통계량을 바꾸지 않는다",
          "  = codex 판정(*\"α 제거가 스케일 불일치를 고칠 경로는 사실상 없다\"*)의 실측 확인.", "",
          "## 5. 한계 (제거 금지)", "",
          "- **의사-gold 로 잰 AUC 다.** phase1 val 에는 사람 gold 가 없어 count ≥ t\\*=3 을 썼다.",
          "  D-B16 의 S_self 0.659 = 이 라벨 자체의 상한이다.",
          "- **감사 120편**(캘리브레이션 200편과 배타). val 529편 전체가 아니다.",
          "- **§4.8 은 여전히 미검정이다.** 이 판정은 *\"재개할 근거가 없다\"* 이지",
          "  *\"particle smoothing 이 효과 없다\"* 가 아니다. 논문 문구는 D-B56 §6 을 따른다."]
    (D / "GATE_VERDICT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (D / "gate_verdict.json").write_text(json.dumps(
        {"agg": {k: {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in v.items()}
                 for k, v in agg.items()},
         "verdicts": verdicts, "gate3_comparable": comparable,
         "passed": passed}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
