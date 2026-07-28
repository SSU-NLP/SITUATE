#!/usr/bin/env python3
"""①B human 층화표본 — 채점 (블라인드 2단계 대조 + 불일치 사후판정).

파이프라인 (SCORING_RUBRIC.md):
  build  →  사람이 sheet.jsonl 의 human_label 채움 (블라인드)
  --stage1  →  human_label vs answer_key.assigned_label 자동 대조:
               · 축별 human↔model 일치율 (labeled 행 기준)
               · 중립토큰 하위분석 (모델-중립 / 사람-중립 일치)
               · disagreements.jsonl 생성 (verdict·error_tag 공란) — 3단계 입력
  --final   →  disagreements.jsonl 의 verdict 채운 것 읽어:
               · 축별 model-wrong(✗)율 → 합격선 ≤20% PASS/FAIL
               · human-wrong / ambiguous 분해 + error_tag 히스토그램
               · RESULT.md 작성

합격선(≤20%)·절차는 build 시 SCORING_RUBRIC.md 에 동결 — 여기서 재정의하지 않는다.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

VERDICTS = ("model-wrong", "human-wrong", "ambiguous")
PASS_MAX_WRONG = 0.20  # frozen in SCORING_RUBRIC.md (do not change post-result)


def read_jsonl(p: Path) -> list[dict]:
    p = Path(p)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def load_labels(path: Path) -> dict[str, str]:
    """{row_id: human_label} from a filled .csv (spreadsheet) or .jsonl."""
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            return {r["row_id"]: str(r.get("human_label", "")).strip()
                    for r in csv.DictReader(f) if str(r.get("human_label", "")).strip()}
    return {r["row_id"]: str(r.get("human_label", "")).strip()
            for r in read_jsonl(path) if str(r.get("human_label", "")).strip()}


def stage1(d: Path, labels_path: Path | None) -> None:
    sheet = {r["row_id"]: r for r in read_jsonl(d / "sheet.jsonl")}
    key = {r["row_id"]: r for r in read_jsonl(d / "answer_key.jsonl")}
    # human_label source: explicit --labels (csv/jsonl), else a filled sheet.csv, else sheet.jsonl itself
    src = labels_path or (d / "sheet.csv" if load_labels(d / "sheet.csv") else d / "sheet.jsonl")
    hl = load_labels(src)
    print(f"human_label 출처: {src}")
    for rid, r in sheet.items():
        r["human_label"] = hl.get(rid, "")
    labeled = [r for r in sheet.values() if str(r.get("human_label", "")).strip()]
    print(f"labeled {len(labeled)}/{len(sheet)} rows")
    if not labeled:
        print("사람이 human_label 을 아직 안 채웠습니다 — sheet 채운 뒤 다시 --stage1.")
        return

    # mode: yes/no(중립 정당성) vs full(정확 라벨 일치)
    lset = {str(r["human_label"]).strip().lower() for r in labeled}
    yn = lset <= {"yes", "no", "unclear", "y", "n", "예", "아니오", "애매"}
    NO = {"no", "n", "아니오"}
    UNCLEAR = {"unclear", "애매"}
    print(f"모드: {'yes/no (중립 정당성)' if yn else 'full (정확 라벨 일치)'}")

    per: dict[str, dict] = defaultdict(lambda: {"n": 0, "agree": 0,
                                                "model_neutral": 0, "model_neutral_agree": 0})
    disagreements: list[dict] = []
    for r in labeled:
        rid = r["row_id"]
        k = key[rid]
        h = str(r["human_label"]).strip()
        m = str(k["assigned_label"]).strip()
        neu = bool(k["is_neutral"])
        a = per[r["axis"]]
        a["n"] += 1
        if yn:
            hl = h.lower()
            unclear = hl in UNCLEAR
            # human "no"=근거없음 ↔ 모델 중립 ; human "yes"=근거있음 ↔ 모델 non-neutral
            agree = (not unclear) and ((hl in NO) == neu)
        else:
            unclear = False
            agree = (h == m)
        a["agree"] += int(agree)
        if neu:
            a["model_neutral"] += 1
            a["model_neutral_agree"] += int(agree)
        if not agree:
            disagreements.append({
                "row_id": rid, "axis": r["axis"], "doc_id": r["doc_id"], "item_id": r["item_id"],
                "sentence": r["sentence"], "human_label": h,
                "model_label": m, "model_is_neutral": neu,
                # yes+중립 = 모델이 근거 있는데 none(회피 후보); no+비중립 = 모델이 과하게 라벨
                "verdict": "ambiguous" if unclear else "", "error_tag": ""})

    write_jsonl(d / "disagreements.jsonl", disagreements)
    (d / "stage1_counts.json").write_text(
        json.dumps({axis: per[axis]["n"] for axis in per}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'축':14s} {'n':>3s} {'일치율':>7s}  {'모델중립 일치':>12s}")
    for axis in sorted(per):
        a = per[axis]
        agr = a["agree"] / a["n"] if a["n"] else 0.0
        mn = f"{a['model_neutral_agree']}/{a['model_neutral']}" if a["model_neutral"] else "-"
        print(f"{axis:14s} {a['n']:>3d} {agr:>7.2%}  {mn:>12s}")
    print(f"\ndisagreements.jsonl = {len(disagreements)}행 → verdict 채운 뒤 --final")


def final(d: Path) -> None:
    counts_p = d / "stage1_counts.json"
    if not counts_p.exists():
        print("stage1_counts.json 없음 — 먼저 --stage1 을 실행하세요.")
        return
    labeled_by_axis = Counter(json.loads(counts_p.read_text()))
    dis = read_jsonl(d / "disagreements.jsonl")
    unlabeled = [r for r in dis if str(r.get("verdict", "")).strip() not in VERDICTS]
    if unlabeled:
        print(f"⚠️ verdict 미기입/무효 {len(unlabeled)}행 (허용값 {VERDICTS}). 채운 뒤 재실행.")
        for r in unlabeled[:5]:
            print(f"   {r['row_id']}  verdict={r.get('verdict','')!r}")
        return

    per: dict[str, Counter] = defaultdict(Counter)
    tags: dict[str, Counter] = defaultdict(Counter)
    for r in dis:
        per[r["axis"]][r["verdict"].strip()] += 1
        if r["verdict"].strip() == "model-wrong" and str(r.get("error_tag", "")).strip():
            tags[r["axis"]][r["error_tag"].strip()] += 1

    rows, all_pass = [], True
    for axis in sorted(labeled_by_axis):
        n = labeled_by_axis[axis]
        mw, hw, am = per[axis]["model-wrong"], per[axis]["human-wrong"], per[axis]["ambiguous"]
        wrong_rate = mw / n if n else 0.0
        passed = wrong_rate <= PASS_MAX_WRONG
        all_pass &= passed
        rows.append((axis, n, mw, hw, am, wrong_rate, passed))

    lines = ["# ①B human 층화표본 — 채점 결과 (RESULT)", "",
             f"> 합격선 = 축별 model-wrong(✗) ≤ {PASS_MAX_WRONG:.0%} (SCORING_RUBRIC.md 동결).",
             f"> 채점 = `human_sample_score.py --final`. 판정 = **{'PASS' if all_pass else 'FAIL'}** (전 축).",
             "",
             "## 축별 결과",
             "| 축 | n(labeled) | 일치(=n−불일치) | ✗ model-wrong | human-wrong | △ ambiguous | ✗율 | 판정 |",
             "|---|---|---|---|---|---|---|---|"]
    for axis, n, mw, hw, am, wr, passed in rows:
        agree = n - (mw + hw + am)
        lines.append(f"| {axis} | {n} | {agree} | {mw} | {hw} | {am} | {wr:.1%} | "
                     f"{'✅ PASS' if passed else '🔴 FAIL'} |")
    lines += ["", "## 오류유형 히스토그램 (model-wrong 만)"]
    any_tag = False
    for axis in sorted(tags):
        if tags[axis]:
            any_tag = True
            lines.append(f"- **{axis}**: " + ", ".join(f"{t}×{c}" for t, c in tags[axis].most_common()))
    if not any_tag:
        lines.append("- (model-wrong 없음 또는 error_tag 미기입)")
    lines += ["", "## 해석 / 한계 (제거 금지)",
              "- **질적 발견(문면에서만 드러나는 설계 이슈)은 `FINDINGS.md` 참조** — space 어휘공백, "
              "intentionality 3종(pervasive-goal·대사 화제/목적·carry 방향), 라벨링 규약. 다음 VERSION 재료.",
              "- 이 표본은 라벨 타당성 spot-check(축당 20, n=100). 축 단위 결론은 소표본 노이즈.",
              "- 중립토큰(none/unspecified) 정당성 확인이 1차 목적 — model-wrong 중 `avoidance-none`",
              "  태그가 있으면 '근거 있는데 회피'한 경우이므로 프롬프트 수정 후보.",
              "- 블라인드 라벨링이라 human-wrong/ambiguous 는 사람·문장 난이도 몫이며 모델 결함 아님."]
    (d / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {d / 'RESULT.md'}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path,
                    default=root / "outputs/experiments/2026-07-22_gate1b_repcorpus/human_sample")
    ap.add_argument("--labels", type=Path, default=None,
                    help="채운 human_label 출처(.csv/.jsonl). 생략 시 sheet.csv→sheet.jsonl 순 자동탐색")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage1", action="store_true", help="블라인드 대조 → disagreements.jsonl")
    g.add_argument("--final", action="store_true", help="verdict 채운 뒤 ✗율·PASS/FAIL → RESULT.md")
    args = ap.parse_args()
    if args.stage1:
        stage1(args.dir, args.labels)
    else:
        final(args.dir)


if __name__ == "__main__":
    main()
