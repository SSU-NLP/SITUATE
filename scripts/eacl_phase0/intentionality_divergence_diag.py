#!/usr/bin/env python3
"""goal-divergence 진단 (관문①-a · README 다음단계 #2 · Q5 재정의 선행).

intentionality(구 goal) 축이 (1) target-only 추출에서 얼마나 식별 가능한지, (2) 라벨러 간
얼마나 발산하는지, (3) 제거 시 dev Score 가 얼마나 취약한지를 dev-3(pieman/tunnel/monkey)에서
측정한다. 이 진단은 Q5(현행유지 / 좁힘 / Indexter-frame) 결정의 근거일 뿐, **재정의 동결은
dev-3 Gate C 에서** 한다(스펙: EACL/s5-eism-redesign-guide.md §3.5, §6.1).

세 측정:
  1. 식별가능성 — active_goal 이 null(none)이 아닌 문장 비율. space(구 place)의 86% unspecified
     같은 死변수 여부. (states 파일에서 계산, 예산 0.)
  2. 라벨 발산 — 두 라벨러(dir A vs B)의 intentionality reset 계열 일치(Cohen κ + raw). enum 집합이
     모델마다 달라 값 자체 비교는 부적절 → 리셋(코드 판정) 수준에서 비교. (dir 2개 필요.)
  3. 취약성 — deployable Score(count≥t*) 를 5축 vs intentionality 제거 4축으로 계산해 ΔScore.
     참조(구 fable5): −goal pieman 0.598→0.546, dialseg 0.634→0.477 = 신호 실재하나 취약.

예산: 단일 dir 진단(1·3)은 **예산 0**(기존 산출물 재사용). 모델간 발산(2)만 두 번째 dir 필요.

용법:
  # fable5 기존 산출로 식별성+취약성 (구 키; 예산 0)
  python intentionality_divergence_diag.py --dir outputs/eacl_phase0/{doc}_michelmann_s5 --legacy-keys \
      --docs pieman_michelmann tunnel_michelmann monkey_michelmann
  # 신 deepseek eismv2 산출 (신 키) + fable5 대비 발산
  python intentionality_divergence_diag.py --dir <deepseek_eismv2_dir> --dir-b <fable5_dir> --legacy-keys-b ...
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

# frozen deployable axes (perspective-only protagonist, D-B8)
NEW_FROZEN = ["protagonist_perspective", "space", "time", "causality", "intentionality"]
OLD_FROZEN = ["protagonist_perspective", "place", "time", "causality", "goal"]
T_STAR = 3


def read(p: Path) -> list[dict]:
    p = Path(p)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def end_turn_labels(n_sent: int, bt: list[int]) -> list[int]:
    y = [0] * n_sent
    for i in bt:
        y[i - 1] = 1
    return y


def deployable_score(rows: list[dict], axes: list[str], tstar: int | None):
    """Official deployable Score over `axes` (count≥t*). tstar None → argmax on this doc."""
    rows = sorted(rows, key=lambda r: int(r["transition"]))
    n_sent = len(rows) + 1
    ref_bt = [int(r["transition"]) for r in rows if int(r["boundary"])]
    gold = end_turn_labels(n_sent, ref_bt)
    count = {int(r["transition"]): sum(int(r[a]) for a in axes) for r in rows}

    def sc(t):
        bt = [i for i in count if count[i] >= t]
        return score_dialogues([gold], [end_turn_labels(n_sent, bt)])["score"]

    if tstar is None:
        tstar = max(range(1, len(axes) + 1), key=sc)
    return sc(tstar), tstar


def none_rate(states_dir: Path, doc: str, int_file: str, field: str, null: str):
    rows = read(Path(states_dir) / doc / f"{int_file}_states.jsonl")
    if not rows:
        return None
    n = len(rows)
    none = sum(1 for r in rows if r.get(field) == null)
    return {"n": n, "none": none, "none_rate": none / n, "specified_rate": 1 - none / n}


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1 = sum(a) / n; pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def cfg(legacy: bool):
    return (("goal", "place", OLD_FROZEN) if legacy else ("intentionality", "space", NEW_FROZEN))


def load(roots: list[Path], docs: list[str]) -> dict[str, tuple[Path, list[dict]]]:
    """Each root holds one doc's reset_matrix.jsonl + <doc>/<axis>_states.jsonl. Align roots↔docs."""
    out = {}
    for root, doc in zip(roots, docs):
        rows = [r for r in read(Path(root) / "reset_matrix.jsonl") if str(r["doc_id"]) == doc]
        out[doc] = (Path(root), rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, nargs="+", type=Path,
                    help="primary s5 out_dir(s), one per doc (each: reset_matrix.jsonl + <doc>/<axis>_states.jsonl)")
    ap.add_argument("--dir-b", nargs="+", type=Path, default=None,
                    help="second labeler out_dir(s), one per doc, for divergence (optional)")
    ap.add_argument("--docs", nargs="+", required=True, help="doc_ids (subdir names), aligned to --dir order")
    ap.add_argument("--legacy-keys", action="store_true", help="primary dir uses old place/goal keys (fable5)")
    ap.add_argument("--legacy-keys-b", action="store_true", help="dir-b uses old place/goal keys")
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/experiments/2026-07-21_s5_eism_redesign/goal_divergence_REPORT.md")
    args = ap.parse_args()
    assert len(args.dir) == len(args.docs), "--dir 와 --docs 개수 일치 필요"

    int_ax, space_ax, frozen = cfg(args.legacy_keys)
    A = load(args.dir, args.docs)
    by_doc = {doc: rows for doc, (_, rows) in A.items()}
    root_of = {doc: root for doc, (root, _) in A.items()}

    lines = ["# goal-divergence 진단 (관문①-a)", "",
             f"- primary dirs: {', '.join(f'`{r}`' for r in args.dir)} (keys: {'legacy place/goal' if args.legacy_keys else 'eism space/intentionality'})",
             f"- dir-b: {', '.join(f'`{r}`' for r in args.dir_b)}" if args.dir_b else "- dir-b: (없음 — 발산 측정 skip)",
             f"- docs: {', '.join(args.docs)}", f"- t\\*={T_STAR}", "",
             "## 1. 식별가능성 (active_goal none-rate; space 대비)", "",
             "| doc | int none-rate | int specified | space none-rate |",
             "|---|---|---|---|"]
    id_rows = {}
    for doc in args.docs:
        gi = none_rate(root_of[doc], doc, int_ax, "active_goal", "none")
        gs = none_rate(root_of[doc], doc, space_ax, "location", "unspecified")
        id_rows[doc] = (gi, gs)
        if gi:
            lines.append(f"| {doc} | {gi['none_rate']:.3f} | {gi['specified_rate']:.3f} | "
                         f"{gs['none_rate']:.3f} |" if gs else
                         f"| {doc} | {gi['none_rate']:.3f} | {gi['specified_rate']:.3f} | — |")

    lines += ["", "## 3. 취약성 (deployable Score: 5축 vs intentionality 제거 4축)", "",
              "| doc | 5축 Score (t*) | −int 4축 Score (t*) | Δ(=int 기여) | 5축@t\\*=3 | −int@t\\*=3 | Δ@3 |",
              "|---|---|---|---|---|---|---|"]
    ablated = [a for a in frozen if a != int_ax]
    frag = {}
    for doc in args.docs:
        rows = by_doc.get(doc)
        if not rows:
            lines.append(f"| {doc} | (reset_matrix 없음) | | | | | |")
            continue
        s5, t5 = deployable_score(rows, frozen, None)
        s4, t4 = deployable_score(rows, ablated, None)
        s5f, _ = deployable_score(rows, frozen, T_STAR)
        s4f, _ = deployable_score(rows, ablated, T_STAR)
        frag[doc] = s5 - s4
        lines.append(f"| {doc} | {s5:.3f} (t*={t5}) | {s4:.3f} (t*={t4}) | {s5 - s4:+.3f} | "
                     f"{s5f:.3f} | {s4f:.3f} | {s5f - s4f:+.3f} |")

    if args.dir_b:
        assert len(args.dir_b) == len(args.docs), "--dir-b 와 --docs 개수 일치 필요"
        int_ax_b, _, _ = cfg(args.legacy_keys_b)
        B = load(args.dir_b, args.docs)
        rmb = {doc: {int(r["transition"]): r for r in rows} for doc, (_, rows) in B.items()}
        lines += ["", "## 2. 라벨 발산 (intentionality reset: primary vs dir-b)", "",
                  "| doc | n共通 | reset-rate A | reset-rate B | raw 일치 | Cohen κ |",
                  "|---|---|---|---|---|---|"]
        for doc in args.docs:
            ra = {int(r["transition"]): int(r[int_ax]) for r in by_doc.get(doc, [])}
            rb = {t: int(r[int_ax_b]) for t, r in rmb.get(doc, {}).items()}
            common = sorted(set(ra) & set(rb))
            if not common:
                lines.append(f"| {doc} | 0 | | | | |")
                continue
            a = [ra[t] for t in common]; b = [rb[t] for t in common]
            raw = sum(x == y for x, y in zip(a, b)) / len(common)
            lines.append(f"| {doc} | {len(common)} | {sum(a)/len(a):.3f} | {sum(b)/len(b):.3f} | "
                         f"{raw:.3f} | {cohen_kappa(a, b):.3f} |")

    # 판정 보조 요약 (Q5 는 codex 결정 — 여기선 수치만)
    lines += ["", "## 요약 (Q5 결정은 codex 위임; 수치만)", ""]
    if id_rows:
        mn = min(v[0]["none_rate"] for v in id_rows.values() if v[0])
        mx = max(v[0]["none_rate"] for v in id_rows.values() if v[0])
        lines.append(f"- 식별성: intentionality none-rate {mn:.2f}~{mx:.2f} "
                     f"(space 死변수 기준 ~0.86 대비 참고).")
    if frag:
        lines.append(f"- 취약성: intentionality 기여 ΔScore {min(frag.values()):+.3f}~{max(frag.values()):+.3f} "
                     f"(양수·유의하면 '신호 실재하나 취약' → 죽이지 않는 재정의 필요).")
    lines.append("- 발산: dir-b 제공 시 κ 낮으면 라벨러 불안정 → 재정의가 안정화해야 함.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[written] {args.out}", flush=True)


if __name__ == "__main__":
    main()
