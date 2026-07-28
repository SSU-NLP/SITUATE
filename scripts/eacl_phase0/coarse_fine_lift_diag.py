#!/usr/bin/env python3
"""coarse vs fine 경계별 축 lift 진단 — "인과 단절은 coarse 경계와만 붙는다" 가설 검정.

🔴 지위: **탐색적 진단**(dev-3 = 연습장, D-M13③ 무제한 열람). 확증 근거 아님.

## 왜 이걸 하나 (사용자 제기)
지금까지 causality 축은 **단일 gold** 에 대해 평가돼 lift 1.07~2.09 로 약했고 8~9번 기각됐다.
그런데 Zacks 계열은 **인과 단절이 주로 coarse(큰 덩어리) 경계와 강하게 붙고**, fine 경계는
하위목표 전환·장소/시간/인물 변화에서 온다고 본다. 사용자 예:
> "라이터를 켰다 → 불이 붙었다 → 그걸 들고 옆방으로 갔다" — 인과는 안 끊겼지만 공간이 바뀌어
> 경계가 생긴다. "요리한다" 안에서 재료손질 → 굽기 → 플레이팅 전환마다 사람은 경계를 찍는다.

⇒ **가설:** 우리 gold 가 fine 을 포함하면 causality lift 가 낮게 나오는 것이 **당연**하며,
축이 틀린 게 아니라 **비교 대상(경계 층위)이 틀린 것**일 수 있다.

## 방법 — 주석자 합의 강도로 coarse/fine 분리 (새 라벨 0, API 0)
`data/eacl_phase0/{doc}_michelmann/sourcedata/alldata*.mat` = **(주석자 × 시간샘플)** 이진 행렬.
열 합 = 그 시점에 버튼을 누른 인원 수. 각 gold 경계 시각 주변 창에서 **최대 동시 인원**을 그
경계의 **합의 강도**로 삼고, 중앙값으로 coarse(상위) / fine(하위) 을 가른다.
그 두 집합 각각에 대해 5축 + 실험 축의 lift 를 따로 계산한다.

**해석 규칙 (측정 전 선언):** causality lift 가 **coarse 에서 유의하게 높고 fine 에서 낮으면**
사용자 가설을 지지 = 축은 살아 있고 층위가 문제. **양쪽 다 낮으면** 층위 문제가 아니다.
⚠️ 표본이 작다(경계 18~30개를 둘로 쪼갬). **확증이 아니라 방향 탐지용**이다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
import llm_s5_states as s5  # noqa: E402

BASE = REPO / "outputs/eacl_phase0"
DATA = REPO / "data/eacl_phase0"
AXES = ["protagonist_present", "space", "time", "causality", "intentionality"]
# (문서, alldata 파일, 총 재생시간 추정용 = 열 수). 열은 시간샘플이며 gold 는 초 단위 onset 을 갖는다.
SRC = {"pieman": "alldata_run1.mat", "tunnel": "alldata.mat"}
WIN_S = 3.0     # 경계 시각 ±3초 창에서 최대 동시 인원 (Michelmann 계열 관행 범위)


def consensus(doc: str, onsets: np.ndarray) -> np.ndarray | None:
    """각 경계 onset(초) 의 합의 강도 = ±WIN_S 창 내 최대 동시 누름 인원."""
    f = SRC.get(doc)
    if not f or not (DATA / f"{doc}_michelmann/sourcedata/{f}").exists():
        return None
    m = sio.loadmat(DATA / f"{doc}_michelmann/sourcedata/{f}", squeeze_me=True)
    a = m[[k for k in m if not k.startswith("__")][0]]
    counts = a.sum(0)                                  # [T] 시점별 인원
    total_s = float(onsets.max()) * 1.05               # 열 인덱스 ↔ 초 선형 매핑 근사
    hz = len(counts) / total_s
    out = []
    for t in onsets:
        lo, hi = int(max(0, (t - WIN_S) * hz)), int(min(len(counts), (t + WIN_S) * hz))
        out.append(float(counts[lo:hi].max()) if hi > lo else 0.0)
    return np.asarray(out)


def axis_fires(doc: str) -> tuple[dict, np.ndarray, np.ndarray]:
    d = BASE / f"{doc}_michelmann_s5_q36_35b_v2" / doc
    st = {"protagonist": {int(x["item_id"]): x for x in s5.read(d / "protagonist_states.jsonl")},
          "causality": {int(x["item_id"]): x for x in s5.read(d / "causality_states.jsonl")},
          "time": {int(x["item_id"]): x for x in s5.read(d / "time_states.jsonl")},
          "space": s5.carry_forward({int(x["item_id"]): x for x in s5.read(d / "space_states.jsonl")},
                                    "location", "unspecified"),
          "intentionality": s5.carry_forward(
              {int(x["item_id"]): x for x in s5.read(d / "intentionality_states.jsonl")},
              "active_goal", "none")}
    master = sorted(s5.read(BASE / f"{doc}_michelmann_master.jsonl"), key=lambda r: int(r["transition"]))
    fires = {a: [] for a in AXES}
    gold, onset = [], []
    for r in master:
        i = int(r["transition"])
        if any(i not in st[a] or i - 1 not in st[a] for a in st):
            continue
        fires["protagonist_present"].append(s5.reset_protagonist_present(st["protagonist"][i-1], st["protagonist"][i]))
        fires["space"].append(s5.reset_space(st["space"][i-1], st["space"][i]))
        fires["time"].append(s5.reset_time(st["time"][i-1], st["time"][i]))
        fires["causality"].append(s5.reset_causality(st["causality"][i-1], st["causality"][i]))
        fires["intentionality"].append(s5.reset_intentionality(st["intentionality"][i-1], st["intentionality"][i]))
        gold.append(int(r["boundary"]))
        onset.append(float(r.get("cur_onset_s") or 0.0))
    return ({a: np.asarray(v) for a, v in fires.items()},
            np.asarray(gold), np.asarray(onset))


def lift(fire: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.sum() == 0 or neg.sum() == 0 or fire[neg].mean() == 0:
        return float("nan")
    return float(fire[pos].mean() / fire[neg].mean())


def main() -> None:
    L = ["# coarse vs fine 경계별 축 lift — \"인과 단절은 coarse 와만 붙는가\"\n",
         "> 🔴 **탐색적 진단**(dev-3 = 연습장). 확증 근거 아님. 새 라벨 0 · API 0.",
         "> 합의 강도 = 경계 onset ±3초 창의 **최대 동시 누름 인원**(주석자×시간 원자료).",
         "> coarse = 합의 강도 상위 절반 · fine = 하위 절반 (문서별 중앙값 분할).\n"]
    for doc in ("pieman", "tunnel", "monkey"):
        fires, gold, onset = axis_fires(doc)
        gi = np.flatnonzero(gold == 1)
        con = consensus(doc, onset[gi])
        L.append(f"\n## {doc} — 경계 {len(gi)}개\n")
        if con is None:
            L.append(f"> ⚠️ 주석자 원자료 없음(`alldata*.mat` 부재) — coarse/fine 분리 불가. 전체만 보고.\n")
            L += ["| 축 | 전체 lift |", "|---|---|"]
            for a in AXES:
                L.append(f"| {a} | {lift(fires[a], gold == 1, gold == 0):.2f} |")
            continue
        med = float(np.median(con))
        coarse_i, fine_i = gi[con >= med], gi[con < med]
        neg = gold == 0
        cm, fm = np.zeros(len(gold), bool), np.zeros(len(gold), bool)
        cm[coarse_i], fm[fine_i] = True, True
        L.append(f"합의 강도: 중앙값 **{med:.0f}명** · 범위 {con.min():.0f}~{con.max():.0f} · "
                 f"coarse {cm.sum()}개 / fine {fm.sum()}개\n")
        L += ["| 축 | 전체 lift | **coarse lift** | **fine lift** | coarse−fine |", "|---|---|---|---|---|"]
        for a in AXES:
            al, cl, fl = (lift(fires[a], gold == 1, neg), lift(fires[a], cm, neg), lift(fires[a], fm, neg))
            mark = " **←**" if a == "causality" else ""
            L.append(f"| {a}{mark} | {al:.2f} | **{cl:.2f}** | **{fl:.2f}** | {cl-fl:+.2f} |")
    L += ["\n## 해석 규칙 (측정 전 선언)\n",
          "- causality lift 가 **coarse 에서 높고 fine 에서 낮으면** → 사용자 가설 지지."
          " 축은 살아 있고 **경계 층위**가 문제.",
          "- **양쪽 다 낮으면** → 층위 문제가 아니다. 기존 기각 판정 유지.",
          "\n⚠️ 표본이 작다(경계 18~30개를 둘로 분할). **확증이 아니라 방향 탐지용.**",
          "⚠️ 합의 강도는 열 인덱스↔초 **선형 매핑 근사**로 계산했다(원 샘플링레이트 미확인)."]
    p = REPO / "outputs/experiments/m4m5_phase1_anchor/COARSE_FINE_LIFT.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
