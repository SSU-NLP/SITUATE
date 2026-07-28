#!/usr/bin/env python3
"""데이터셋 × 방법 Score 표 (공식 DTS Score, deployable).

## 무엇을
서사(dev-3)와 DTS 대화 4종에 대해 **같은 방법들**을 같은 채점기로 돌려 한 표에 놓는다.
지금까지 이 대조가 흩어져 있었다(M3 수치는 D-M13, M1 은 registry §2, DTS 는 별도).

## 방법
| 방법 | 규칙 | 작동점 |
|---|---|---|
| M3-S5 (동결) | 5축 리셋 합 `count ≥ t*` | **t\\*=3 동결** |
| M3-S5 t\\*=2 / 4 | 같음, 임계만 변경 | 진단용(동결 아님) |
| 축 단독 | 그 축이 리셋되면 경계 | — |
| M1-CosDrift | `idm_score ≥ δ*` | **δ\\*=0.5594 동결**(TIAGE 캘리브) |
| M1-CosDrift(적응) | 문서 내 상위 q 분위수 | q = 그 데이터의 gold 밀도 ⚠️ oracle |

🔴 **deployable 규약(CLAUDE.md):** 메인은 경계 **개수를 스스로 정하는** 규칙이다. 마지막 행
(M1 적응)은 gold 밀도를 쓰므로 **oracle 누출**이며 **참고용**으로만 싣는다.
🔴 dev-3 는 **동결 v3.1 재조합**(D-B14) 사용 — DTS(v3.1 직접추출)와 조건을 맞춘다.
🔴 DTS 는 D-M5 에서 **scope 밖** 선언 — 성능 주장에 쓰지 않는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
import llm_s5_states as s5  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

BASE = REPO / "outputs/eacl_phase0"
DTS = BASE / "dts_s5_q36_35b_v31"
AXES = ["protagonist_present", "space", "time", "causality", "intentionality"]
DELTA_STAR = 0.5594          # M1 canonical (registry §2)
DEV3 = ("pieman", "tunnel", "monkey")


def load_dev3() -> dict[str, dict]:
    """동결 v3.1 재조합(D-B14): fix_space_H 의 space + fix_time 의 time + v2 나머지 3축."""
    out = {}
    for doc in DEV3:
        def S(tag, axis):
            return {int(x["item_id"]): x for x in
                    s5.read(BASE / f"{doc}_michelmann_s5_{tag}" / doc / f"{axis}_states.jsonl")}
        st = {"protagonist": S("q36_35b_v2", "protagonist"),
              "causality": S("q36_35b_v2", "causality"),
              "time": S("fix_time", "time"),
              "space": s5.carry_forward(S("fix_space_H", "space"), "location", "unspecified"),
              "intentionality": s5.carry_forward(S("q36_35b_v2", "intentionality"),
                                                 "active_goal", "none")}
        R, g, idm = [], [], []
        for r in sorted(s5.read(BASE / f"{doc}_michelmann_master.jsonl"),
                        key=lambda r: int(r["transition"])):
            i = int(r["transition"])
            if any(i not in st[a] or i - 1 not in st[a] for a in st):
                continue
            R.append([s5.reset_protagonist_present(st["protagonist"][i-1], st["protagonist"][i]),
                      s5.reset_space(st["space"][i-1], st["space"][i]),
                      s5.reset_time(st["time"][i-1], st["time"][i]),
                      s5.reset_causality(st["causality"][i-1], st["causality"][i]),
                      s5.reset_intentionality(st["intentionality"][i-1], st["intentionality"][i])])
            g.append(int(r["boundary"]))
            idm.append(float(r.get("idm_score") or 0.0))
        out[doc] = dict(R=np.array(R, np.int8), gold=np.array(g, np.int8), idm=np.array(idm))
    return out


def load_dts(name: str) -> dict[str, dict]:
    rows = [json.loads(l) for l in open(DTS / name / "reset_matrix.jsonl") if l.strip()]
    docs: dict[str, dict] = {}
    for r in rows:
        d = docs.setdefault(r["doc_id"], dict(R=[], gold=[], idm=[]))
        d["R"].append([int(r[a]) for a in AXES])
        d["gold"].append(int(r["boundary"]))
        d["idm"].append(float(r.get("idm_score") or 0.0))
    return {k: dict(R=np.array(v["R"], np.int8), gold=np.array(v["gold"], np.int8),
                    idm=np.array(v["idm"])) for k, v in docs.items()}


def score(docs: dict, pred_fn) -> float:
    ys, ps = [], []
    for v in docs.values():
        b = pred_fn(v)
        ys.append(np.concatenate([v["gold"], [0]]).tolist())
        ps.append(np.concatenate([b, [0]]).tolist())
    return score_dialogues(ys, ps)["score"]


def main() -> None:
    data = {"dev-3 (서사)": load_dev3()}
    for s in ("tiage", "dialseg711", "superseg", "ami"):
        data[s] = load_dts(s)

    methods = [
        ("**M3-S5 count≥3 (동결)**", lambda v: (v["R"].sum(1) >= 3).astype(np.int8)),
        ("M3-S5 count≥2", lambda v: (v["R"].sum(1) >= 2).astype(np.int8)),
        ("M3-S5 count≥4", lambda v: (v["R"].sum(1) >= 4).astype(np.int8)),
        ("축 단독: protagonist", lambda v: v["R"][:, 0].astype(np.int8)),
        ("축 단독: space", lambda v: v["R"][:, 1].astype(np.int8)),
        ("축 단독: time", lambda v: v["R"][:, 2].astype(np.int8)),
        ("축 단독: causality", lambda v: v["R"][:, 3].astype(np.int8)),
        ("축 단독: intentionality", lambda v: v["R"][:, 4].astype(np.int8)),
        ("**M1-CosDrift δ\\*=0.5594 (동결)**", lambda v: (v["idm"] >= DELTA_STAR).astype(np.int8)),
        ("전부 경계 (하한)", lambda v: np.ones(len(v["gold"]), np.int8)),
        ("경계 없음 (하한)", lambda v: np.zeros(len(v["gold"]), np.int8)),
    ]

    L = ["# 데이터셋 × 방법 Score 표 (공식 DTS Score, deployable)\n",
         "> 🔴 **탐색적.** dev-3 = 연습장(D-M13③) · **DTS 는 D-M5 에서 scope 밖 선언** —"
         " 성능 주장에 쓰지 않는다.",
         "> dev-3 는 **동결 v3.1 재조합**(D-B14), DTS 는 v3.1 직접추출 — **조건 일치**.",
         "> deployable = 경계 개수를 방법이 스스로 정한다. gold 개수 미사용.\n",
         "\n| | " + " | ".join(data) + " |", "|---|" + "---|" * len(data)]
    dens = ["**gold 밀도**"] + [f"{np.concatenate([v['gold'] for v in d.values()]).mean():.4f}"
                             for d in data.values()]
    L.append("| " + " | ".join(dens) + " |")
    L.append("| " + " | ".join(["**대화/문서 수**"] + [str(len(d)) for d in data.values()]) + " |")
    L.append("|---|" + "---|" * len(data))
    for name, fn in methods:
        L.append(f"| {name} | " + " | ".join(f"{score(d, fn):.4f}" for d in data.values()) + " |")

    L += ["\n> **경계 없음/전부 경계** = 자명한 하한. 어떤 방법이든 이보다 나아야 의미가 있다.",
          "> gold 밀도가 데이터마다 크게 다르다(ami 0.025 ~ tiage 0.170) — Score 절대값을"
          " **데이터셋 간에 비교하지 말 것**. 같은 열(같은 데이터) 안에서만 방법을 비교한다.",
          "\n⚠️ M1 δ\\*=0.5594 는 **TIAGE 에서 캘리브**된 값이다(registry §2). 다른 데이터에서는"
          " 작동점이 안 맞을 수 있으며 이는 M1 의 실제 약점이지 채점 오류가 아니다.",
          "⚠️ ami 는 대화 2편뿐이다."]
    p = REPO / "outputs/experiments/m4m5_phase1_anchor/SCORE_MATRIX.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
