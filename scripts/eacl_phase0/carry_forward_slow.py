#!/usr/bin/env python3
"""P0-S4 후속: 느린 변수(place, goal)에 코드 carry-forward를 적용해 재측정.

死因 진단(A-검증): place/goal은 여러 문장에 걸쳐 지속되나 가끔만 명시되는 "느린" 변수라
target-only 추출에서 과소계측(unspecified 86%)됐다. 최종 factored-slot 모델은 이를 슬롯의
state-carry로 푼다. Phase 0에는 슬롯이 없으므로, 코드가 기계적으로 "직전 명시값 유지"를 흉내낸다.

오염 안전:
- LLM 출력(문장 단독 enum)은 그대로. carry는 100% 코드.
- **경계-무관(boundary-agnostic) carry**: 경계 지식을 전혀 쓰지 않는다. unspecified/none이면 직전
  명시값을 잇고, 새 명시값이 나오면 덮어쓴다. (경계에서 강제 리셋하는 방식은 carry 범위 자체가
  경계 지식이라 boundary-proxy 오염 → 사용 안 함.)
- protagonist/time/causality(빠른/조밀 축)는 carry 안 함. place/goal만.
"""
from __future__ import annotations
import argparse, json, importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("s4", Path(__file__).with_name("llm_enum_states.py"))
_s4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s4)
AXES, reset = _s4.AXES, _s4.reset

CARRY = {"place": ("setting_type", "unspecified"), "goal": ("goal_category", "none")}


def read(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def carry_forward(states: dict[int, dict], field: str, null: str) -> dict[int, dict]:
    out, last = {}, None
    for i in sorted(states):
        s = dict(states[i])
        if s[field] == null:
            if last is not None:
                s[field] = last  # 직전 명시값 유지 (경계 무관)
        else:
            last = s[field]      # 새 명시값으로 갱신
        out[i] = s
    return out


def auc(y: list[int], score: list[float]) -> float:
    pos = [a for yy, a in zip(y, score) if yy == 1]
    neg = [a for yy, a in zip(y, score) if yy == 0]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("master", type=Path)
    ap.add_argument("states_dir", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    master = {int(r["transition"]): r for r in read(args.master)}
    n = len(master) + 1  # sentences = transitions + 1
    base = {ax: {x["item_id"]: x for x in read(args.states_dir / f"{ax}_states.jsonl")} for ax in AXES}
    carried = dict(base)
    for ax, (field, null) in CARRY.items():
        carried[ax] = carry_forward(base[ax], field, null)

    rows = []
    for i in range(1, n):
        r = master[i]
        row = {"doc_id": r["doc_id"], "transition": i,
               "boundary": int(r["boundary"]), "idm_score": float(r["idm_score"])}
        for ax in AXES:
            row[ax] = reset(ax, carried[ax][i - 1], carried[ax][i])
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")

    y = [r["boundary"] for r in rows]
    print(f"=== carry-forward 재측정 (place, goal) -> {args.out} ===")
    for ax in AXES:
        b = read(args.states_dir / f"{ax}_states.jsonl")
        note = ""
        if ax in CARRY:
            f, nu = CARRY[ax]
            u0 = sum(1 for x in b if x[f] == nu)
            uc = sum(1 for i in carried[ax] if carried[ax][i][f] == nu)
            note = f"  ({f} {nu}: {u0}->{uc})"
        print(f"  {ax:12s} reset {sum(r[ax] for r in rows):3d}/{len(rows)}  AUC {auc(y,[r[ax] for r in rows]):.3f}{note}")
    total = [sum(r[ax] for ax in AXES) for r in rows]
    print(f"  {'5축 합':12s} AUC {auc(y, total):.3f}   (baseline no-carry 0.643 / IDM 0.914)")


if __name__ == "__main__":
    main()
