"""VESSL Job 산출물(/shared)을 로컬 주석 디렉터리로 **완전성 기준** 병합한다.

왜 `cp -rn` 이면 안 되는가 (2026-08-03 발견):
`supervise_delta.sh` 는 `cp -rn "$d"/pg*/ "$OUT"/` 로 병합한다. `-n` = no-clobber 라
**로컬에 이미 있는 파일은 건드리지 않는다.** 그런데 `llm_s5_states.py` 는 배치마다
`{axis}_states.jsonl` 을 덮어써서, 워커가 중간에 죽으면 **잘린 축 파일이 남는다**.
그 상태에서 VESSL 이 같은 문서를 완주해도 완전한 파일이 잘린 파일을 못 덮는다
→ 영구적으로 truncated 인데 존재하므로 "완료"로 집계된다.

이 스크립트는 **item_id 집합이 더 완전한 쪽**을 남긴다. 반복 실행해도 안전하다
(항상 더 완전한 쪽으로만 수렴하고, 동률이면 아무것도 안 한다).

    python scripts/eacl_phase1/merge_vessl_out.py <out_dir> [--src-glob '/shared/hiontop/out/delta-cap20*'] [--apply]

기본은 dry-run 이다. `--apply` 를 줘야 실제로 쓴다.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

AXES = ("protagonist", "space", "time", "causality", "intentionality")


def item_ids(p: Path) -> set[int] | None:
    """축 파일의 item_id 집합. 없거나 못 읽으면 None."""
    if not p.exists():
        return None
    try:
        return {int(json.loads(l)["item_id"]) for l in open(p, encoding="utf-8") if l.strip()}
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path, help="로컬 주석 디렉터리 (병합 목적지)")
    ap.add_argument("--src-glob", default="/shared/hiontop/out/delta-cap20*",
                    help="VESSL 산출 디렉터리 glob")
    ap.add_argument("--apply", action="store_true", help="실제로 쓴다 (기본은 dry-run)")
    a = ap.parse_args()

    srcs = sorted(p for p in Path("/").glob(a.src_glob.lstrip("/")) if p.is_dir())
    print(f"소스 {len(srcs)}개: {', '.join(s.name for s in srcs)}")

    n_new_doc = n_new_axis = n_upgrade = n_keep = n_stage1 = 0
    upgrades: list[str] = []
    for src in srcs:
        for ddir in sorted(src.glob("pg*")):
            if not ddir.is_dir():
                continue
            dst = a.out_dir / ddir.name
            if not dst.exists():
                n_new_doc += 1
                if a.apply:
                    dst.mkdir(parents=True, exist_ok=True)

            sl = ddir / "story_lists.json"
            if sl.exists() and not (dst / "story_lists.json").exists():
                n_stage1 += 1
                if a.apply:
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sl, dst / "story_lists.json")

            for ax in AXES:
                s_ids = item_ids(ddir / f"{ax}_states.jsonl")
                if s_ids is None:
                    continue
                d_ids = item_ids(dst / f"{ax}_states.jsonl")
                if d_ids is None:
                    n_new_axis += 1
                elif len(s_ids) > len(d_ids):
                    n_upgrade += 1
                    upgrades.append(f"{ddir.name}/{ax}: 로컬 {len(d_ids)} → VESSL {len(s_ids)}")
                else:
                    n_keep += 1
                    continue
                if a.apply:
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ddir / f"{ax}_states.jsonl", dst / f"{ax}_states.jsonl")

    mode = "적용" if a.apply else "DRY-RUN (쓰지 않음)"
    print(f"[{mode}] 새 문서 {n_new_doc} · stage1 신규 {n_stage1} · 축 신규 {n_new_axis} · "
          f"**잘린 로컬 파일 교체 {n_upgrade}** · 유지 {n_keep}")
    for u in upgrades[:15]:
        print(f"  ↑ {u}")
    if len(upgrades) > 15:
        print(f"  ... 외 {len(upgrades) - 15}건")
    if not a.apply and (n_new_doc or n_new_axis or n_upgrade or n_stage1):
        print("\n실제 반영하려면 --apply 를 붙여 다시 실행한다.")


if __name__ == "__main__":
    main()
