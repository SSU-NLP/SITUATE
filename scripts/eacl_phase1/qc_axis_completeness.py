"""축 파일 **완전성** 검사 — 존재만으로 완료 판정하는 구멍을 막는다.

왜 필요한가 (2026-08-03 발견):
`llm_s5_states.py` 는 배치마다 `write(sp, ...)` 로 `{axis}_states.jsonl` 을 덮어쓴다.
즉 문서를 다 끝내기 전에도 **파일은 이미 존재한다**. 그런데 감독기 `supervise_delta.sh`
의 `remaining()` 과 드라이버의 resume 판정은 `.exists()` 만 본다 — 워커가 중간에 죽으면
잘린 축 파일이 "완료"로 집계되어 미완인 채 완주 선언이 날 수 있다.

정상 완료 조건: 축 파일의 item_id 집합 == {0 .. n_transitions}
(`sequence()` 가 previous 1개 + current n개 = n+1 개 상태를 만든다).

읽기 전용이다 — 아무것도 고치지 않고 목록만 낸다.

    python scripts/eacl_phase1/qc_axis_completeness.py <master.jsonl|샤드디렉터리> <out_dir> [--fix-list 파일]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

AXES = ("protagonist", "space", "time", "causality", "intentionality")


def load_expected(src: Path) -> dict[str, int]:
    """doc_id → 기대 상태 개수(= 전이 수 + 1)."""
    files = sorted(src.glob("*.jsonl")) if src.is_dir() else [src]
    n_tr: dict[str, int] = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                d = str(r["doc_id"])
                n_tr[d] = max(n_tr.get(d, 0), int(r["transition"]))
    return {d: n + 1 for d, n in n_tr.items()}


def check(doc_dir: Path, axis: str, want: int) -> str | None:
    """None = 정상. 문자열 = 문제 사유."""
    p = doc_dir / f"{axis}_states.jsonl"
    if not p.exists():
        return "missing"
    try:
        ids = {int(json.loads(l)["item_id"]) for l in open(p, encoding="utf-8") if l.strip()}
    except (json.JSONDecodeError, KeyError, ValueError):
        return "unreadable"
    if ids != set(range(want)):
        return f"partial {len(ids)}/{want}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="master jsonl 또는 샤드 디렉터리")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--fix-list", type=Path, default=None,
                    help="재처리가 필요한 doc_id 를 이 파일에 쓴다")
    a = ap.parse_args()

    expected = load_expected(a.src)
    bad: dict[str, list[str]] = {}
    n_ok = 0
    for doc_id, want in sorted(expected.items()):
        reasons = [f"{ax}:{r}" for ax in AXES
                   if (r := check(a.out_dir / doc_id, ax, want)) is not None]
        if reasons:
            bad[doc_id] = reasons
        else:
            n_ok += 1

    n_partial = sum(1 for v in bad.values() if any("partial" in x for x in v))
    print(f"문서 {len(expected)}편 · 완전 {n_ok} · 문제 {len(bad)} "
          f"(그중 **잘린 축 파일 보유** {n_partial}편 — 존재만 보면 완료로 오집계된다)")
    for doc_id, reasons in list(bad.items())[:20]:
        print(f"  {doc_id}: {', '.join(reasons)}")
    if len(bad) > 20:
        print(f"  ... 외 {len(bad) - 20}편")

    if a.fix_list:
        a.fix_list.write_text("".join(f"{d}\n" for d in bad), encoding="utf-8")
        print(f"재처리 목록 → {a.fix_list} ({len(bad)}편)")


if __name__ == "__main__":
    main()
