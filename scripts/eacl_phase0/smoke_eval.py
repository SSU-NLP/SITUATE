#!/usr/bin/env python3
"""스모크 평가 (관문①-c §3.0) — 단일 축 재추출이 '맞는 자리에' 리셋을 냈는가.

주지표 = A↔B 리셋 F1 (deepseek-신 프롬프트 ↔ fable5). 리셋 '수'·none-rate는 실패를 못 거르므로
(dsv4pro 리셋 15 ≈ fable5 17인데 F1 0.437) 보조로만. 사용자 정정 2026-07-21.

한 (doc, axis)에 대해:
  1. A↔B 리셋 F1: 신(deepseek-eismv2) vs fable5.  [주] — dsv4pro↔fable5 baseline 대비 상승해야.
  2. none/unspecified rate: 신 vs fable5 vs dsv4pro.  [보조·방향]
  3. 리셋 수: 신.  [보조·폭주(≥40) 탐지]
  4. count≥3 혼합: 신 축 + dsv4pro 나머지 4축 → dsv4pro 전(0.105류) 대비 상향.  [구조]

용법:
  python smoke_eval.py --axis intentionality --doc pieman \
    --new  outputs/eacl_phase0/pieman_michelmann_s5_eismv2 \
    --fable5 outputs/eacl_phase0/pieman_michelmann_s5 \
    --dsv4pro outputs/eacl_phase0/pieman_michelmann_s5_dsv4pro
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("s5", Path(__file__).with_name("llm_s5_states.py"))
_s5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s5)

# axis -> (states filename stem, field, null, reset fn). fable5/dsv4pro use OLD file stem.
AXIS = {
    "space": {"new_stem": "space", "old_stem": "place", "field": "location",
              "null": "unspecified", "reset": _s5.reset_space},
    "intentionality": {"new_stem": "intentionality", "old_stem": "goal", "field": "active_goal",
                       "null": "none", "reset": _s5.reset_intentionality},
}
# dsv4pro reset_matrix column name for each smoke axis (OLD keys)
OLD_COL = {"space": "place", "intentionality": "goal"}
OTHER4 = {"space": ["protagonist_perspective", "time", "causality", "goal"],
          "intentionality": ["protagonist_perspective", "place", "time", "causality"]}


def read(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def reset_series(states_dir: Path, doc: str, stem: str, field: str, null: str, resetfn) -> dict[int, int]:
    rows = read(Path(states_dir) / doc / f"{stem}_states.jsonl")
    st = {r["item_id"]: r for r in rows}
    cf = _s5.carry_forward(st, field, null)
    n = len(st)
    return {i: resetfn(cf[i - 1], cf[i]) for i in range(1, n)}, rows


def binary_f1(a: dict[int, int], b: dict[int, int]) -> tuple[float, int]:
    common = sorted(set(a) & set(b))
    tp = sum(a[i] and b[i] for i in common)
    fp = sum(b[i] and not a[i] for i in common)  # b=신 예측, a=fable5 기준
    fn = sum(a[i] and not b[i] for i in common)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, len(common)


def none_rate(states_dir, doc, stem, field, null):
    rows = read(Path(states_dir) / doc / f"{stem}_states.jsonl")
    if not rows:
        return None
    return sum(1 for r in rows if r.get(field) == null) / len(rows)


def count3_mixture(new_reset: dict[int, int], dsv4pro_dir: Path, doc: str, axis: str):
    """신 축 reset + dsv4pro 나머지 4축 → count≥3 비율. dsv4pro 전5축 대비."""
    rm = [r for r in read(Path(dsv4pro_dir) / "reset_matrix.jsonl") if str(r["doc_id"]) == doc]
    rm = {int(r["transition"]): r for r in rm}
    others = OTHER4[axis]
    mix_ge3 = dsv_ge3 = 0
    common = sorted(set(new_reset) & set(rm))
    for t in common:
        base = sum(int(rm[t][a]) for a in others)
        mix_ge3 += (base + new_reset[t]) >= 3
        dsv_ge3 += sum(int(rm[t][a]) for a in _s5_frozen_old()) >= 3
    n = len(common)
    return (mix_ge3 / n if n else 0.0), (dsv_ge3 / n if n else 0.0), n


def _s5_frozen_old():
    return ["protagonist_perspective", "place", "time", "causality", "goal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=list(AXIS))
    ap.add_argument("--doc", required=True)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--fable5", required=True, type=Path)
    ap.add_argument("--dsv4pro", required=True, type=Path)
    args = ap.parse_args()
    cfg = AXIS[args.axis]

    new_r, new_rows = reset_series(args.new, args.doc, cfg["new_stem"], cfg["field"], cfg["null"], cfg["reset"])
    fab_r, _ = reset_series(args.fable5, args.doc, cfg["old_stem"], cfg["field"], cfg["null"], cfg["reset"])
    dsv_r, _ = reset_series(args.dsv4pro, args.doc, cfg["old_stem"], cfg["field"], cfg["null"], cfg["reset"])

    f1_new, n = binary_f1(fab_r, new_r)
    f1_dsv, _ = binary_f1(fab_r, dsv_r)   # baseline (실패한 라벨러)

    print(f"=== SMOKE EVAL: {args.doc} / {args.axis} ({len(new_rows)} 문장) ===")
    print(f"[주지표] A↔B 리셋 F1 (vs fable5):")
    print(f"    신(eismv2)  = {f1_new:.3f}   <- 이게 올라가야")
    print(f"    dsv4pro(전) = {f1_dsv:.3f}   (baseline, 실패 라벨러)")
    print(f"    판정: {'✅ 상승' if f1_new > f1_dsv + 0.02 else '🔴 미상승' if f1_new < f1_dsv else '△ 미미'}"
          f" (Δ {f1_new - f1_dsv:+.3f})")
    nr_new = none_rate(args.new, args.doc, cfg["new_stem"], cfg["field"], cfg["null"])
    nr_fab = none_rate(args.fable5, args.doc, cfg["old_stem"], cfg["field"], cfg["null"])
    nr_dsv = none_rate(args.dsv4pro, args.doc, cfg["old_stem"], cfg["field"], cfg["null"])
    print(f"[보조] {cfg['null']}-rate: 신 {nr_new:.3f} | fable5 {nr_fab:.3f} | dsv4pro {nr_dsv:.3f}")
    n_reset = sum(new_r.values())
    print(f"[보조] 신 리셋 수 = {n_reset}/{len(new_r)}  {'🔴 폭주(≥40)' if n_reset >= 40 else 'ok'}")
    mix, dsv3, nc = count3_mixture(new_r, args.dsv4pro, args.doc, args.axis)
    print(f"[구조] count≥3 혼합(신 {args.axis}+dsv4pro 4축) = {mix:.3f}  vs  dsv4pro 전5축 {dsv3:.3f}  "
          f"(n={nc}; 상향이면 구조 회복)")


if __name__ == "__main__":
    main()
