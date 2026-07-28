#!/usr/bin/env python3
"""옛 human_sample 을 Stage-0 정제본에 **같은 순서로 재정렬**한 블라인드 sheet 생성.

목적(사용자 지정 2026-07-25): 저번 블라인드 도구와 **똑같은 문장·똑같은 순서**로 라벨링하되, Stage-0 로
사라진 잡물(N=7)은 제외하고, 모델 라벨은 **새 정제본 재추출(stage0)** 값으로 answer_key 에 담는다.

- 순서 = 옛 `sheet.jsonl` 행 순서 그대로(잡물 7행만 스킵) → 93행.
- 문장 = 옛 문장(정제본에 그대로 존재하는 것만). (doc, sentence) → 새 정제본 item_id 로 remap.
- 모델 라벨(answer_key) = 새 stage0 run 의 축별 필드값 · 중립여부.
- choices = 새 stage0 story_lists 의 closed list + 중립토큰(yesno 모드에선 미표시).
- 번역 = 옛 translations.json 재사용(문장 동일, sentence 매칭). gloss = 옛 list_gloss.json 재사용.

산출: <out>/{sheet.jsonl, answer_key.jsonl, sheet.csv, translations.json, list_gloss.json}
채점/웹툴은 기존 `human_sample_score.py` / `human_sample_webpage.py --mode yesno` 그대로 사용.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# mirrors human_sample_build.py (동일 필드·중립토큰)
PERSPECTIVE = ["first", "second", "third", "none"]
TEMPORAL_SHIFT = ["absent", "sequence", "jump", "simultaneity"]
ASPECT = ["inceptive", "ongoing", "completive"]
AXIS_CFG = {
    "space":          {"field": "location",              "neutral": "unspecified", "choices": "locations"},
    "intentionality": {"field": "active_goal",           "neutral": "none",        "choices": "goals"},
    "time":           {"field": "temporal_shift_marker", "neutral": "absent",      "choices": TEMPORAL_SHIFT},
    "protagonist":    {"field": "perspective",           "neutral": "none",        "choices": PERSPECTIVE},
    "causality":      {"field": "aspect",                "neutral": "ongoing",     "choices": ASPECT},
}


def read_jsonl(p: Path) -> list[dict]:
    p = Path(p)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def sequence(rows: list[dict]) -> list[str]:
    rows = sorted(rows, key=lambda x: int(x["transition"]))
    return [rows[0]["previous"], *[x["current"] for x in rows]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-sheet", type=Path,
                    default=REPO / "outputs/experiments/2026-07-22_gate1b_repcorpus/human_sample/sheet.jsonl")
    ap.add_argument("--old-translations", type=Path,
                    default=REPO / "outputs/experiments/2026-07-22_gate1b_repcorpus/human_sample/translations.json")
    ap.add_argument("--old-gloss", type=Path,
                    default=REPO / "outputs/experiments/2026-07-22_gate1b_repcorpus/human_sample/list_gloss.json")
    ap.add_argument("--new-master", type=Path,
                    default=REPO / "outputs/eacl_phase0/repcorpus_master_stage0.jsonl")
    ap.add_argument("--new-run", type=Path,
                    default=REPO / "outputs/eacl_phase0/repcorpus_s5_q36_35b_v2_stage0")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "outputs/experiments/2026-07-25_stage0_relabel_verify/human_sample_matched")
    args = ap.parse_args()

    old = read_jsonl(args.old_sheet)
    old_tr = json.loads(args.old_translations.read_text()) if args.old_translations.exists() else {}

    # new master: (doc, sentence) -> new_item_id
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(args.new_master):
        by_doc[str(r["doc_id"])].append(r)
    new_idx: dict[tuple[str, str], int] = {}
    for doc, rows in by_doc.items():
        for i, s in enumerate(sequence(rows)):
            new_idx.setdefault((doc, s), i)

    # new run: per (doc, axis) item_id -> field value; and per-doc stage1 lists (choices)
    def label_of(doc: str, axis: str, item_id: int) -> str | None:
        for st in read_jsonl(args.new_run / doc / f"{axis}_states.jsonl"):
            if int(st["item_id"]) == item_id:
                return st[AXIS_CFG[axis]["field"]]
        return None

    lists_cache: dict[str, dict] = {}
    def choices_of(doc: str, axis: str) -> list[str]:
        cfg = AXIS_CFG[axis]
        if isinstance(cfg["choices"], list):
            return list(cfg["choices"])
        if doc not in lists_cache:
            lists_cache[doc] = json.loads((args.new_run / doc / "story_lists.json").read_text())["lists"]
        return sorted(set(lists_cache[doc][cfg["choices"]])) + [cfg["neutral"]]

    import re
    DIVIDER_RE = re.compile(r"\*(?:\s*\*)+")

    def emit(doc, axis, sent, density_bin, old_row_id):
        """새 정제본에서 sent 의 item_id 를 찾아 sheet/key/tr 에 추가. 못 찾으면 False."""
        nid = new_idx.get((doc, sent))
        if nid is None:
            return False
        cfg = AXIS_CFG[axis]
        lab = label_of(doc, axis, nid)
        row_id = f"{axis}:{doc}:{nid}"
        sheet.append({"row_id": row_id, "doc_id": doc, "item_id": nid, "axis": axis,
                      "density_bin": density_bin, "sentence": sent,
                      "choices": choices_of(doc, axis), "human_label": "", "notes": ""})
        key.append({"row_id": row_id, "doc_id": doc, "item_id": nid, "axis": axis,
                    "assigned_label": lab, "is_neutral": (lab == cfg["neutral"])})
        # 번역은 human_sample_translate.py 로 sheet 문장 기준 재생성한다(여기서 재사용 금지 —
        # 구분선 쪼갠 조각은 옛 융합문 번역과 다르므로). translations.json 은 여기서 쓰지 않음.
        return True

    sheet, key, tr, skipped, split_kept = [], [], {}, [], []
    for r in old:
        doc, sent, axis = r["doc_id"], r["sentence"], r["axis"]
        if emit(doc, axis, sent, r["density_bin"], r["row_id"]):
            continue                           # 정제본에 그대로 존재 → 그대로 유지
        # verbatim 부재 → 구분선 융합이면 쪼갠 조각(진짜 서사)으로 유지, 아니면 진짜 junk 스킵
        pieces = [p.strip() for p in DIVIDER_RE.split(sent) if p.strip()] if DIVIDER_RE.search(sent) else []
        got = [p for p in pieces if emit(doc, axis, p, r["density_bin"], r["row_id"])]
        if got:
            split_kept.append((axis, doc, r["item_id"], len(got)))
        else:
            skipped.append((axis, doc, r["item_id"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sheet.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sheet), encoding="utf-8")
    (args.out_dir / "answer_key.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in key), encoding="utf-8")
    # translations.json / list_gloss.json 는 여기서 쓰지 않는다 — 빌드 후
    # `human_sample_translate.py` 로 sheet 문장·stage0 run 기준 재생성(구분선 조각 올바로 번역).
    with (args.out_dir / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "axis", "density_bin", "sentence", "choices", "human_label", "notes"])
        for r in sheet:
            w.writerow([r["row_id"], r["axis"], r["density_bin"], r["sentence"],
                        " | ".join(r["choices"]), "", ""])

    from collections import Counter
    print(f"matched sheet = {len(sheet)} rows (옛 순서 유지)")
    print(f"  axis blocks: {dict(Counter(r['axis'] for r in sheet))}")
    print(f"  구분선 케이스 유지(쪼갠 조각): {split_kept}")
    print(f"  진짜 junk 스킵(비서사): {skipped}")
    print(f"-> {args.out_dir}")
    print("  ※ 다음: human_sample_translate.py 로 translations.json·list_gloss.json 재생성 후 webpage 빌드")


if __name__ == "__main__":
    main()
