#!/usr/bin/env python3
"""M5 Phase 2 — 학습 코퍼스 ablation REPORT 생성 (fiction / share / both).

입력 = slot_seed_sweep.py --dump 이 append 한 sweep_*.jsonl + x10_place_ceiling_probe.py 의 x10_gate.json
출력 = outputs/experiments/m5_phase2_corpus_ablation/REPORT.md

usage: uv run python scripts/eacl_phase0/corpus_ablation_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/experiments/m5_phase2_corpus_ablation"
AXES = ["protagonist", "space", "time", "causality", "intentionality"]
ORDER = ["fiction_repro", "fiction", "share", "both", "both_matched", "both_silver", "share_maskproto"]


def load_runs() -> dict:
    runs = {}
    for p in sorted(OUT.glob("sweep_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                runs[r["tag"]] = r          # 같은 tag 재실행 시 최신이 이김
    return runs


def ms(v):
    a = np.array(v, dtype=float)
    return f"{a.mean():.2f} ± {a.std():.2f}"


def main() -> int:
    runs = load_runs()
    if not runs:
        raise SystemExit(f"sweep_*.jsonl 없음: {OUT}")
    gate = json.loads((OUT / "x10_gate.json").read_text()) if (OUT / "x10_gate.json").exists() else {}
    tags = [t for t in ORDER if t in runs] + [t for t in runs if t not in ORDER]

    L = ["# M5 Phase 2 — 학습 코퍼스 ablation (fiction / share / both)\n",
         "**질문.** SHARE 스크린플레이 대사 데이터를 학습에 넣으면 슬롯 정렬이 나아지나? "
         "place/time 2축은 원 대본 **슬러그라인(씬 헤딩) 외부 gold**로 교체하고 "
         "protagonist/causality/goal 3축은 모달리티 내 M3(S5) silver 유지 (ledger **D-M8 §990** 사양).\n",
         "**지위.** 본편 학습 코퍼스 결정이 아니라 **X1-0(정보 상한 게이트)의 SHARE 파일럿 진단** = "
         "D-M8 (a)~(d)의 경험적 검정. 결과와 무관하게 본편 코퍼스는 D-M8대로 fiction 단일 유지.\n"]

    # ── 1. setup
    L += ["\n## 1. 실험 setup\n",
          "- **데이터.** fiction 10 docs / 3090 전이 (Pieman·Tunnel·Monkey michelmann + Tier2 7편, "
          "M3(S5) silver anchor). SHARE 12 docs / 971 전이 (874 유효, 97 = 10% 마스킹) — "
          "`share_valid_probe30` ∩ IMSDB 대본 ∩ 슬러그라인 파싱 성공.",
          "- **place/time 외부 gold.** 원 대본 슬러그라인 `INT./EXT. 장소 - 시간` → 턴 단위 씬 정렬 "
          "(부분문자열 매칭 + 단조 LIS-DP + 씬 내 불변 보간). `build_share_turn_scene.py`. "
          "**회귀 테스트: 턴 라벨을 세션으로 재집계 → place 76/96(79%), time 38/96(40%) "
          "= D-M7 실측(76/92=83%, 39/92=42%) 재현 ✅**",
          "- **모델.** `FactoredSlot` (EntNet typed keyed slots, D1 per-axis extractor routing), "
          "M5 = 5 슬롯 / M4 = monolithic 통제군. frozen mpnet 임베딩. variant=weak (bottleneck 16, pw_cap 60).",
          "- **평가 고정 (모델만이 처치).** val = fiction 2문서(tunnel_michelmann, the_lady_with_the_dog) "
          "조건 불문 동일 · probe fit + label-correlation ceiling B = **항상 fiction-train** · "
          "pos_weight = fiction 고정 · chunk_len/max_steps로 업데이트 수 고정.",
          "- **지표.** own-axis AUPRC (슬롯 k → 축 k, held-out val) + "
          "factoring gain = M4 excess − M5 excess (>0 = D1 라우팅이 monolithic이 엉킨 걸 분리). "
          "*(Score 규약과 별개 — Phase 2는 분절이 아니라 표현 프로브라 AUPRC.)*\n"]

    # ── 2. gate
    if gate:
        L += ["\n## 2. 선행 GATE — X1-0 정보 상한 프로브\n",
              "**질문.** 슬러그라인 gold는 각본가 저작 = 최고 품질 라벨이다. 그런데 그게 **학습 가능한** "
              "타깃인가? frozen mpnet Δ특징에서 라벨을 직접 로지스틱 회귀(doc 단위 LOO CV)로 예측해 "
              "**입력이 신호를 나르는지**만 측정 (슬롯 모델의 상한 — 같은 입력의 저차원 요약이므로).\n",
              "| 프로브 | n | 양성 | base | AUPRC | lift |",
              "|---|---|---|---|---|---|"]
        for k, v in gate.items():
            L.append(f"| {k} | {v['n']} | {v['pos']} | {v['base']:.3f} | {v['auprc']:.3f} | "
                     f"**{v['lift']:.2f}x** |")
        L += ["\n**판정: GATE FAIL (SHARE place/time).** 동일 프로브가 fiction에선 3.4x/4.9x로 작동하는데 "
              "SHARE에선 1.16x/0.90x(chance) → **대사 임베딩에 place/time 신호가 없다.** "
              "SHARE의 place 양성 44개·time 14개는 '적어서 부족한' 게 아니라 **원리적으로 학습 불가**.\n",
              "**이 게이트가 D-M7보다 강한 이유.** D-M7의 근거는 M3 recall 4%/0% → "
              "*\"추출기가 나쁜 것 아니냐\"*는 반박이 가능했다. 여기선 **frozen 임베딩 자체에 선형 신호가 0**이므로 "
              "추출기와 무관 = **M3-독립적 확증**. plan Phase X의 X1-0 판정 규칙(\"미달 축은 프로그램에서 제외\")이 "
              "그대로 적용된다.\n"]

    # ── 3. results
    L += ["\n## 3. 결과 — 조건별 own-axis AUPRC (M5, held-out fiction val)\n",
          "| 조건 | train docs | train 전이 | " + " | ".join(a[:5] for a in AXES) + " | factoring gain |",
          "|---|---|---|" + "---|" * 6]
    for t in tags:
        r = runs[t]
        L.append(f"| **{t}** | {r['n_docs']} | {r['n_trans']} | " +
                 " | ".join(ms(r["own_axis"][a]) for a in AXES) +
                 f" | {ms(r['gains'])} |")
    L.append("\n*(mean ± std over seeds; probe·ceiling·val 전부 고정이므로 조건 간 차이는 학습 코퍼스 효과)*\n")

    # deltas vs fiction
    if "fiction" in runs:
        base = runs["fiction"]
        L += ["\n### fiction 대비 변화 (Δ)\n",
              "| 조건 | " + " | ".join(a[:5] for a in AXES) + " | gain |",
              "|---|" + "---|" * 6]
        for t in tags:
            if t in ("fiction", "fiction_repro"):
                continue
            r = runs[t]
            cells = []
            for a in AXES:
                d = np.mean(r["own_axis"][a]) - np.mean(base["own_axis"][a])
                sd = np.std(base["own_axis"][a])
                mark = "**" if abs(d) > 2 * max(sd, 0.005) else ""
                cells.append(f"{mark}{d:+.3f}{mark}")
            dg = np.mean(r["gains"]) - np.mean(base["gains"])
            L.append(f"| {t} | " + " | ".join(cells) + f" | {dg:+.3f} |")
        L.append("\n*(**굵게** = fiction 시드 노이즈의 2σ 밖)*\n")

    L += ["\n## 4. 해석 · 판정 · 한계\n",
          "> 아래는 수치 확인 후 작성 (자동 생성 표만으로는 REPORT 미완성 — CLAUDE.md 규칙).\n"]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {OUT/'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
