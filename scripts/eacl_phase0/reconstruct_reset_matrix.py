"""완료된 doc들의 축별 state 캐시에서 reset_matrix.jsonl 재구성 (LLM 호출 0).

llm_s5_states.py는 reset_matrix를 전 doc 완료 후에만 쓴다. probe가 예산초과로 중단되면
캐시는 남아도 reset_matrix가 없다. 이 스크립트는 5축 전부 완료된 doc만 골라 frozen 리셋
규칙(llm_s5_states의 carry_forward + reset_*)을 그대로 적용해 reset_matrix를 만든다.

usage: uv run python scripts/eacl_phase0/reconstruct_reset_matrix.py \
         --master outputs/eacl_phase0/share_valid_probe30_master.jsonl \
         --s5-dir outputs/eacl_phase0/share_valid_probe30_s5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import llm_s5_states as s5  # 같은 디렉토리


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--s5-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.s5_dir / "reset_matrix.jsonl"

    master = [json.loads(l) for l in open(args.master)]
    by_doc: dict[str, list[dict]] = {}
    for r in master:
        by_doc.setdefault(r["doc_id"], []).append(r)

    matrix: list[dict] = []
    used = 0
    for did, rows in by_doc.items():
        rows.sort(key=lambda r: int(r["transition"]))
        n = len(rows) + 1  # texts = transitions + 1
        ddir = args.s5_dir / did
        if not (ddir / "story_lists.json").exists():
            continue
        states: dict[str, dict[int, dict]] = {}
        ok = True
        for ax in s5.AXES:
            p = ddir / f"{ax}_states.jsonl"
            if not p.exists():
                ok = False; break
            st = {int(x["item_id"]): x for x in s5.read(p)}
            if len(st) < n:
                ok = False; break
            states[ax] = st
        if not ok:
            continue
        # carry-forward (frozen, code)
        states["place"] = s5.carry_forward(states["place"], "location", "unspecified")
        states["goal"] = s5.carry_forward(states["goal"], "active_goal", "none")
        by_t = {int(r["transition"]): r for r in rows}
        for i in range(1, n):
            r = by_t[i]
            pc = s5.reset_protagonist_charset(states["protagonist"][i - 1], states["protagonist"][i])
            pp = s5.reset_perspective(states["protagonist"][i - 1], states["protagonist"][i])
            matrix.append({
                "doc_id": did, "transition": i, "boundary": int(r["boundary"]),
                "idm_score": (None if r.get("idm_score") is None else float(r["idm_score"])),
                "protagonist_charset": pc, "protagonist_perspective": pp,
                "protagonist": int(pc or pp),
                "place": s5.reset_place(states["place"][i - 1], states["place"][i]),
                "time": s5.reset_time(states["time"][i - 1], states["time"][i]),
                "causality": s5.reset_causality(states["causality"][i - 1], states["causality"][i]),
                "goal": s5.reset_goal(states["goal"][i - 1], states["goal"][i]),
            })
        used += 1

    s5.write(out, matrix)
    nb = sum(r["boundary"] for r in matrix)
    print(f"reconstruct: {used} complete docs, {len(matrix)} transitions, {nb} boundaries -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
