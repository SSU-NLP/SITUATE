"""캐시된 축 파일에서 `reset_matrix.jsonl` 을 **LLM 없이** 조립한다.

왜 별도 스크립트인가 (2026-08-03):
`run_label.sh` Phase C 는 `llm_s5_states.py` 를 통합 master 로 1회 돌려 reset_matrix 를 만든다.
그 경로에는 두 문제가 있다.
  ① **단일 실패점** — matrix 를 메모리에 쌓아 **패스 끝에 한 번** 쓴다. 11,842편은 8시간이고,
     그 사이 문서 하나가 실패하면 전부 날아간다(실측: 320편 80,641행 누적 후 소실).
  ② **캐시 미스 시 LLM 재호출** — 축 파일이 잘려 있으면 모델을 다시 부르고, 거기서 JSON 이
     깨지면 런이 죽는다(실측: `pg1202_003` protagonist 80/821 에서 크래시).

축 파일이 **완전하면** reset_matrix 는 결정적 계산이다. 동결 함수를 그대로 import 해 쓰므로
재구현이 아니며 결과가 달라질 이유가 없다. 문서 단위로 즉시 append 하여 중간 실패에도 이어간다.

⚠️ 불완전한 문서는 **계산하지 않고 건너뛴다**(라벨을 지어내지 않는다). 목록을 따로 낸다.

    python scripts/eacl_phase1/assemble_reset_matrix.py <master.jsonl> <s5_dir> --out <파일>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
import llm_s5_states as S  # noqa: E402  — 동결 로직을 그대로 쓴다


def states_of(ddir: Path, axis: str) -> dict[int, dict] | None:
    rows = S.read(ddir / f"{axis}_states.jsonl")
    if not rows:
        return None
    return {int(x["item_id"]): x for x in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("master", type=Path)
    ap.add_argument("s5_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skipped-out", type=Path, default=None)
    a = ap.parse_args()

    by_doc: dict[str, list[dict]] = {}
    for line in a.master.open(encoding="utf-8"):
        r = json.loads(line)
        by_doc.setdefault(str(r["doc_id"]), []).append(r)

    n_row = n_doc = 0
    skipped: list[str] = []
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as fh:
        for doc_id, rows in by_doc.items():
            rows.sort(key=lambda x: int(x["transition"]))
            texts = S.sequence(rows)
            ddir = a.s5_dir / doc_id
            st = {ax: states_of(ddir, ax) for ax in S.AXES}
            # 완전성: 모든 축이 {0..len(texts)-1} 를 빠짐없이 가져야 한다
            if any(s is None or set(s) != set(range(len(texts))) for s in st.values()):
                skipped.append(doc_id)
                continue

            # carry-forward — 동결 순서 그대로 (llm_s5_states.main 과 동일)
            st["space"] = S.carry_forward(st["space"], "location", "unspecified")
            st["intentionality"] = S.carry_forward(st["intentionality"], "active_goal", "none")

            by_t = {int(r["transition"]): r for r in rows}
            for i in range(1, len(texts)):
                r = by_t[i]
                pc = S.reset_protagonist_charset(st["protagonist"][i - 1], st["protagonist"][i])
                pp = S.reset_perspective(st["protagonist"][i - 1], st["protagonist"][i])
                ppr = S.reset_protagonist_present(st["protagonist"][i - 1], st["protagonist"][i])
                cm = S.reset_causal_marker(st["causality"][i - 1], st["causality"][i])
                row = {"doc_id": doc_id, "transition": i, "boundary": int(r["boundary"]),
                       "idm_score": (None if r.get("idm_score") is None else float(r["idm_score"])),
                       "protagonist_charset": pc, "protagonist_perspective": pp,
                       "protagonist": int(pc or pp),
                       "space": S.reset_space(st["space"][i - 1], st["space"][i]),
                       "time": S.reset_time(st["time"][i - 1], st["time"][i]),
                       "causality": S.reset_causality(st["causality"][i - 1], st["causality"][i]),
                       "intentionality": S.reset_intentionality(st["intentionality"][i - 1],
                                                                st["intentionality"][i]),
                       "protagonist_present": ppr, "causal_marker": cm}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_row += 1
            n_doc += 1

    print(f"조립 {n_doc:,}편 · {n_row:,}전이 -> {a.out}")
    print(f"건너뜀(축 불완전) {len(skipped)}편")
    for d in skipped[:10]:
        print(f"   {d}")
    if a.skipped_out:
        a.skipped_out.write_text("".join(f"{d}\n" for d in skipped), encoding="utf-8")


if __name__ == "__main__":
    main()
