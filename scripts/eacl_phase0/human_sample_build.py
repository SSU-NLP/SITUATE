#!/usr/bin/env python3
"""①B human 층화표본 — 블라인드 2단계 채점 시트 생성.

목적 (사전등록 §4, PRE_REGISTRATION.md): 숫자로만 본 none/unspecified 라벨이 *정당한 근거 부재*인지
*모델 회피*인지 사람 눈으로 확인. dev-3 gold와 달리 대표 코퍼스(run1)는 무라벨이라, 여기서 사람이
직접 라벨을 매겨(블라인드) 모델 anchor와 대조한다.

🔴 블라인드 2단계 (anchoring 제거):
  1단계  사람이 문장 + 그 문서의 Stage-1 목록(=closed 선택지)만 보고 human_label을 직접 매긴다.
         모델 라벨(assigned_label)은 sheet에 없다 — answer_key.jsonl 에 분리 보관(채점까지 미개봉).
  2단계  human_label vs assigned_label → 축별 일치율 (human_sample_score.py --stage1).
  3단계  불일치 항목만 {model-wrong / human-wrong / ambiguous} 사후 판정 (score.py 가 뽑아준 시트에 기입).

층화 2축 (실행 전 고정):
  (a) 중립값 비율 보존 — 각 축 full 분포의 중립토큰 비율 p 를 표본 20문장에도 반영(round(20p) 중립).
  (b) 대사밀도 bin(극저/저/중/고) 고르게 — 각 (중립/비중립) 그룹 안에서 bin 비례 배분.
  ⚠️ is_neutral(=모델이 중립을 냈나)는 어느 그룹에서 뽑았는지일 뿐, **블라인드 sheet 엔 노출 안 함**
     (노출하면 "모델이 none 냈다"가 새어 anchoring). answer_key 에만 둔다.

축별 채점 대상 필드 = 그 축에서 '중립토큰이 있는' 1차 범주형 필드 (사람이 closed list 에서 고를 수 있는 것):
  space=location(중립 unspecified) · intentionality=active_goal(none) · time=temporal_shift_marker(absent)
  · protagonist=perspective(none) · causality=aspect(ongoing).
  (event_type/characters_present/tense/causal_marker 등 부가 필드는 이 spot-check 대상 아님 — none 정당성이 핵심.)

합격선 = SCORING_RUBRIC.md 에 시트와 동시 동결: 축별 model-wrong(✗) ≤ 20%. 결과 본 뒤 변경 금지
(no-postresult-criterion-change). 검토 중 0.95 소급 임계 제안은 post-hoc 라 불채택(§1B 경위 참조).
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

# mirrors llm_s5_states.py fixed enums (kept local so this stays pure-stdlib)
PERSPECTIVE = ["first", "second", "third", "none"]
TEMPORAL_SHIFT = ["absent", "sequence", "jump", "simultaneity"]
ASPECT = ["inceptive", "ongoing", "completive"]

# per-axis: labeled field in {axis}_states.jsonl · neutral token · choice source
AXIS_CFG = {
    "space":          {"field": "location",              "neutral": "unspecified", "choices": "locations"},
    "intentionality": {"field": "active_goal",           "neutral": "none",        "choices": "goals"},
    "time":           {"field": "temporal_shift_marker", "neutral": "absent",      "choices": TEMPORAL_SHIFT},
    "protagonist":    {"field": "perspective",           "neutral": "none",        "choices": PERSPECTIVE},
    "causality":      {"field": "aspect",                "neutral": "ongoing",     "choices": ASPECT},
}
AXES = tuple(AXIS_CFG)
SEED = 20260723
PER_AXIS = 20


def read_jsonl(p: Path) -> list[dict]:
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def sequence(rows: list[dict]) -> list[str]:
    """Reconstruct the sentence list from transition rows (== llm_s5_states.sequence)."""
    rows = sorted(rows, key=lambda x: int(x["transition"]))
    return [rows[0]["previous"], *[x["current"] for x in rows]]


def _by_bin(recs: list[dict], k: int, rng) -> list[dict]:
    """Take k records spread across density_bins proportional to bin population."""
    if k <= 0 or not recs:
        return []
    groups: dict[str, list] = defaultdict(list)
    for r in recs:
        groups[r["density_bin"]].append(r)
    bins = sorted(groups)
    total = len(recs)
    alloc = {b: k * len(groups[b]) // total for b in bins}
    rem = k - sum(alloc.values())
    for b in sorted(bins, key=lambda b: k * len(groups[b]) / total - alloc[b], reverse=True)[:rem]:
        alloc[b] += 1
    out: list[dict] = []
    for b in bins:
        g = groups[b][:]
        rng.shuffle(g)
        out += g[: min(alloc[b], len(g))]
    if len(out) < k:  # small bins under-filled -> top up from the remainder
        chosen = {id(x) for x in out}
        pool = [r for r in recs if id(r) not in chosen]
        rng.shuffle(pool)
        out += pool[: k - len(out)]
    return out[:k]


def stratified_sample(recs: list[dict], k: int, rng) -> list[dict]:
    """Preserve the neutral-token ratio, then spread each group across density bins."""
    neutral = [r for r in recs if r["is_neutral"]]
    nonneu = [r for r in recs if not r["is_neutral"]]
    p = len(neutral) / len(recs) if recs else 0.0
    k_neu = min(round(k * p), len(neutral))
    k_non = k - k_neu
    if k_non > len(nonneu):  # not enough non-neutral -> shift the slack to neutral
        k_non = len(nonneu)
        k_neu = min(len(neutral), k - k_non)
    return _by_bin(neutral, k_neu, rng) + _by_bin(nonneu, k_non, rng)


def main() -> None:
    import random

    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path,
                    default=root / "outputs/eacl_phase0/repcorpus_s5_eismv2_run1")
    ap.add_argument("--master", type=Path,
                    default=root / "outputs/eacl_phase0/repcorpus_master.jsonl")
    ap.add_argument("--manifest", type=Path,
                    default=root / "outputs/experiments/2026-07-22_gate1b_repcorpus/manifest.json")
    ap.add_argument("--out-dir", type=Path,
                    default=root / "outputs/experiments/2026-07-22_gate1b_repcorpus/human_sample")
    ap.add_argument("--per-axis", type=int, default=PER_AXIS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    density_bin = {d["doc_id"]: d["bin"] for d in json.loads(args.manifest.read_text())}

    # sentences per doc (item_id -> text)
    master = read_jsonl(args.master)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in master:
        by_doc[str(r["doc_id"])].append(r)
    sents = {doc: sequence(rows) for doc, rows in by_doc.items()}

    # build the candidate pool per axis
    pool: dict[str, list[dict]] = {a: [] for a in AXES}
    for doc_dir in sorted(p for p in args.run_dir.iterdir() if p.is_dir()):
        doc = doc_dir.name
        if doc not in sents:
            continue
        lists = json.loads((doc_dir / "story_lists.json").read_text())["lists"]
        for axis, cfg in AXIS_CFG.items():
            field, neutral = cfg["field"], cfg["neutral"]
            if isinstance(cfg["choices"], list):
                choices = list(cfg["choices"])
            else:  # per-doc closed list from Stage 1 + the neutral token
                choices = sorted(set(lists[cfg["choices"]])) + [neutral]
            for st in read_jsonl(doc_dir / f"{axis}_states.jsonl"):
                i = int(st["item_id"])
                if i >= len(sents[doc]):
                    continue
                label = st[field]
                pool[axis].append({
                    "row_id": f"{axis}:{doc}:{i}",
                    "doc_id": doc, "item_id": i, "axis": axis,
                    "sentence": sents[doc][i],
                    "density_bin": density_bin.get(doc, "?"),
                    "choices": choices,
                    "assigned_label": label,
                    "is_neutral": (label == neutral),
                })

    # stratified sample per axis
    rng = random.Random(args.seed)
    sheet, key = [], []
    summary = {}
    for axis in AXES:
        recs = pool[axis]
        picked = stratified_sample(recs, args.per_axis, rng)
        p_full = sum(r["is_neutral"] for r in recs) / len(recs) if recs else 0.0
        p_samp = sum(r["is_neutral"] for r in picked) / len(picked) if picked else 0.0
        summary[axis] = {"pool": len(recs), "picked": len(picked),
                         "neutral_ratio_full": round(p_full, 3),
                         "neutral_ratio_sample": round(p_samp, 3),
                         "bins": dict(sorted(Counter(r["density_bin"] for r in picked).items()))}
        for r in picked:
            sheet.append({"row_id": r["row_id"], "doc_id": r["doc_id"], "item_id": r["item_id"],
                          "axis": axis, "density_bin": r["density_bin"], "sentence": r["sentence"],
                          "choices": r["choices"], "human_label": "", "notes": ""})
            key.append({"row_id": r["row_id"], "doc_id": r["doc_id"], "item_id": r["item_id"],
                        "axis": axis, "assigned_label": r["assigned_label"],
                        "is_neutral": r["is_neutral"]})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sheet.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sheet), encoding="utf-8")
    (args.out_dir / "answer_key.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in key), encoding="utf-8")
    (args.out_dir / "SCORING_RUBRIC.md").write_text(_rubric(args, summary), encoding="utf-8")

    # spreadsheet-friendly CSV (fill the human_label column, then score --labels sheet.csv)
    with (args.out_dir / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "axis", "density_bin", "sentence", "choices", "human_label", "notes"])
        for r in sheet:
            w.writerow([r["row_id"], r["axis"], r["density_bin"], r["sentence"],
                        " | ".join(r["choices"]), "", ""])

    print(f"sheet.jsonl = {len(sheet)} rows ({len(AXES)}축 × {args.per_axis})")
    for axis in AXES:
        s = summary[axis]
        print(f"  {axis:14s} pool={s['pool']:4d}  neutral full={s['neutral_ratio_full']:.2f} "
              f"sample={s['neutral_ratio_sample']:.2f}  bins={s['bins']}")
    print(f"-> {args.out_dir}")


def _rubric(args, summary: dict) -> str:
    lines = [
        "# ①B human 층화표본 — 채점 규칙 (시트와 동시 동결)",
        "",
        f"> seed={args.seed} · 축당 {args.per_axis}문장 × {len(AXES)}축 = {len(AXES) * args.per_axis}문장.",
        "> 생성 = `scripts/eacl_phase0/human_sample_build.py`. 결과 본 뒤 이 규칙 변경 금지",
        "> (`no-postresult-criterion-change`).",
        "",
        "## 🔴 합격선 (결과 보기 전 동결)",
        "- **축별 model-wrong(✗) ≤ 20%.** (3단계에서 '모델이 명백히 틀림'으로 판정된 것만 카운트.)",
        "- 검토 중 0.95 소급 임계 제안이 있었으나 post-hoc 기준이라 **불채택** — 이 spot-check 는 라벨",
        "  타당성(특히 none/unspecified 정당성) 확인용이며 결정성 임계와 무관.",
        "",
        "## 라벨링 방식 (2026-07-23, 사용자 선택)",
        "- **1차 = 예/아니오 웹도구** `label_web.html`(생성 `human_sample_webpage.py --mode yesno`, Artifact 로 open).",
        "  각 문장에 축별 질문 '이 문장에 [축]의 근거가 있나?' → 예(근거 있음)/아니오(근거 없음)/애매. 20지선다·규칙",
        "  암기 없이 중립토큰(none/unspecified/absent/ongoing) **정당성만** 블라인드로 본다 — 사전 fallback(§합격선",
        "  주석의 '중립값 판정만이라도')을 전 축으로 확장한 것. **합격선(≤20%)·표본·seed 불변**(기준변경 아님).",
        "- 대안 = full 20지선다(`--mode full` 또는 sheet.csv 직접). 정밀하나 느림. 채점기는 두 방식 자동 판별.",
        "",
        "## 블라인드 3단계 절차",
        "1. **블라인드 라벨링:** 웹도구(`label_web.html`)에서 문장을 읽고 예/아니오/애매를 누른다(모델 답 미노출).",
        "   끝나면 CSV 내보내기(`row_id,axis,human_label,notes`). full 방식이면 sheet.csv 의 human_label 열 채움.",
        "2. **자동 대조:** `human_sample_score.py --stage1 --labels <내보낸csv>` → 축별 human↔model 일치율",
        "   + `disagreements.jsonl` + `stage1_counts.json`. (yes/no: 아니오↔모델중립, 예↔모델 concrete 로 대조.)",
        "3. **불일치 판정:** `disagreements.jsonl` 의 각 행에 `verdict` ∈ {model-wrong, human-wrong, ambiguous}",
        "   + `error_tag`(예: pronoun-unresolved, mention-vs-stage, avoidance-none) 기입 →",
        "   `human_sample_score.py --final` → 축별 ✗율·PASS/FAIL·오류유형 히스토그램 → RESULT.md.",
        "",
        "## 축별 채점 대상 필드 · 중립토큰",
        "| 축 | 라벨 필드 | 중립토큰 | choices |",
        "|---|---|---|---|",
        "| space | location | unspecified | 문서 locations + unspecified |",
        "| intentionality | active_goal | none | 문서 goals + none |",
        "| time | temporal_shift_marker | absent | absent/sequence/jump/simultaneity |",
        "| protagonist | perspective | none | first/second/third/none |",
        "| causality | aspect | ongoing | inceptive/ongoing/completive |",
        "",
        "## 층화 확인 (생성 시)",
        "| 축 | pool | 중립비율 full | 중립비율 표본 | 밀도 bin 분포 |",
        "|---|---|---|---|---|",
    ]
    for axis in AXES:
        s = summary[axis]
        lines.append(f"| {axis} | {s['pool']} | {s['neutral_ratio_full']:.2f} | "
                     f"{s['neutral_ratio_sample']:.2f} | {s['bins']} |")
    lines.append("")
    lines.append("> `is_neutral` 는 answer_key.jsonl 에만 있고 블라인드 sheet 엔 없음(anchoring 방지).")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
