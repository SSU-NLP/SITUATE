#!/usr/bin/env python3
"""Anchor-labeler equivalence check — is labeler B a drop-in for labeler A?

Motivation (2026-07-20): swapping the frozen S5 anchor labeler (claude-fable-5 →
deepseek-v4-pro) for cost. codex judged this a **measurement-instrument change**, not an
implementation swap: M3's boundary function depends directly on the labeler's output.

Gate structure per codex review (`outputs/reports/2026-07-20_codex_labeler_swap_review.md`):

  Gate A  run integrity    every sentence labelled on every axis, no dup/missing ids
  Gate B  5-axis equality  per-axis reset P/R/F1, co-reset structure, 5-bit vector match,
                           coordinated-count distribution, place granularity (tier 3)
  Gate C  decision equality  final M3 boundaries (exact and ±1), t* crossing,
                           **non-inferiority of DTS Score vs HUMAN gold**

Why Gate C needs human gold: agreement with fable5 only proves implementation equivalence.
"Fable의 오류까지 그대로 복제할 필요는 없다" — a labeler that disagrees with fable5 but scores
as well against human annotation is acceptable. Neither labeler is ground truth.

⚠️ Place string comparison is literal. `duane_library_steps` vs `duane_library` counts as a
disagreement here though it may be the same location; codex flagged that Stage-1 lists must be
semantically aligned before reading the specified-only agreement number. Treat that metric as a
lower bound and audit `disagreements` by hand.

No API calls — reads frozen `*_states.jsonl` from two S5 output dirs plus the gold master.

Usage:
  uv run python scripts/eacl_phase0/labeler_agreement.py --all --json out.json
  uv run python scripts/eacl_phase0/labeler_agreement.py --doc pieman \
      --a outputs/eacl_phase0/pieman_michelmann_s5 \
      --b outputs/eacl_phase0/pieman_michelmann_s5_dsv4pro
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

AXES = ["protagonist_perspective", "space", "time", "causality", "intentionality"]
T_STAR = 3
DEV_DOCS = ["pieman", "tunnel", "monkey"]

# Frozen 2026-07-20, BEFORE seeing deepseek results. All bidirectional.
# Tier-2 place thresholds are DIAGNOSTIC only (codex: marginals cannot authorize a swap).
DIAG = {"unspecified_pp": 5.0, "reset_per_specified": 0.15, "unique_rel": 0.5}
# Gate C is the actual authorization.
GATE_C = {"boundary_f1_min": 0.80, "score_drop_max": 0.03, "tstar_crossing_max": 0.10}

AXIS_FILE = {a: ("protagonist" if a == "protagonist_perspective" else a) for a in AXES}


def read_states(s5_dir: Path, doc: str, axis: str) -> dict[int, dict[str, Any]]:
    path = s5_dir / doc / f"{AXIS_FILE[axis]}_states.jsonl"
    if not path.exists():
        raise FileNotFoundError(str(path))
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {r["item_id"]: r for r in rows}, len(rows)


def carry_forward(vals: dict[int, str], null: str) -> dict[int, str | None]:
    out: dict[int, str | None] = {}
    last: str | None = None
    for k in sorted(vals):
        v = vals[k]
        if v != null:
            last = v
        out[k] = last
    return out


def axis_resets(st: dict[int, dict[str, Any]], axis: str) -> dict[int, int]:
    """Frozen per-axis reset rules (llm_s5_states.py:243-273). Transition i pairs item i-1 → i."""
    ks = sorted(st)
    r: dict[int, int] = {}
    if axis in ("space", "intentionality"):
        field, null = ("location", "unspecified") if axis == "space" else ("active_goal", "none")
        cf = carry_forward({k: st[k][field] for k in ks}, null)
        for a, b in zip(ks, ks[1:]):
            r[b] = int(cf[a] is not None and cf[b] is not None and cf[a] != cf[b])
    elif axis == "protagonist_perspective":
        for a, b in zip(ks, ks[1:]):
            pa, pb = st[a]["perspective"], st[b]["perspective"]
            r[b] = int(pa != pb and pa != "none" and pb != "none")
    elif axis == "time":
        for a, b in zip(ks, ks[1:]):
            r[b] = int(st[b]["temporal_shift_marker"] in {"sequence", "jump", "simultaneity"})
    elif axis == "causality":
        for a, b in zip(ks, ks[1:]):
            r[b] = int(st[a]["event_type"] != st[b]["event_type"] or st[b]["aspect"] == "inceptive")
    return r


def prf(pred: set[int], ref: set[int]) -> tuple[float, float, float]:
    tp = len(pred & ref)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(ref) if ref else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def prf_tol(pred: set[int], ref: set[int], tol: int = 1) -> float:
    """F1 with ±tol transition tolerance (codex: resets can shift by a sentence)."""
    tp_p = sum(any(abs(x - y) <= tol for y in ref) for x in pred)
    tp_r = sum(any(abs(x - y) <= tol for y in pred) for x in ref)
    p = tp_p / len(pred) if pred else 0.0
    r = tp_r / len(ref) if ref else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def cond_entropy(pairs: list[tuple[str, str]]) -> float:
    joint, marg, n = Counter(pairs), Counter(x for x, _ in pairs), len(pairs)
    return -sum((c / n) * math.log2(c / marg[x]) for (x, _), c in joint.items()) if n else 0.0


def end_turn_labels(n_sent: int, bts: list[int]) -> list[int]:
    y = [0] * n_sent
    for i in bts:
        y[i - 1] = 1
    return y


def compare_doc(doc: str, a_dir: Path, b_dir: Path, master: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"doc": doc, "gate_a": {}, "gate_b": {}, "gate_c": {}, "tier3": {}}

    # ---------- Gate A: run integrity ----------
    A: dict[str, Any] = {}
    B: dict[str, Any] = {}
    ga: dict[str, Any] = {"missing_files": [], "row_counts": {}}
    for tag, d, store in (("a", a_dir, A), ("b", b_dir, B)):
        for ax in AXES:
            try:
                st, n = read_states(d, doc, ax)
            except FileNotFoundError as e:
                ga["missing_files"].append(str(e))
                continue
            store[ax] = st
            ga["row_counts"][f"{tag}:{ax}"] = n
    if ga["missing_files"]:
        out["gate_a"] = ga | {"pass": False}
        return out
    n_sent = len(A["space"])
    ga["n_sentences"] = n_sent
    ga["consistent_row_counts"] = len(set(ga["row_counts"].values())) == 1
    ga["contiguous_ids"] = all(sorted(A[ax]) == list(range(n_sent)) for ax in AXES) and all(
        sorted(B[ax]) == list(range(n_sent)) for ax in AXES
    )
    ga["pass"] = ga["consistent_row_counts"] and ga["contiguous_ids"]
    out["gate_a"] = ga

    rA = {ax: axis_resets(A[ax], ax) for ax in AXES}
    rB = {ax: axis_resets(B[ax], ax) for ax in AXES}
    trans = sorted(rA["space"])

    # ---------- Gate B: 5-axis equivalence ----------
    gb: dict[str, Any] = {"per_axis": {}}
    for ax in AXES:
        sa = {t for t in trans if rA[ax][t]}
        sb = {t for t in trans if rB[ax][t]}
        f, p, r = prf(sb, sa)
        gb["per_axis"][ax] = {
            "resets_a": len(sa), "resets_b": len(sb),
            "rate_a": len(sa) / len(trans), "rate_b": len(sb) / len(trans),
            "f1": f, "precision": p, "recall": r,
        }
    kA = {t: sum(rA[ax][t] for ax in AXES) for t in trans}
    kB = {t: sum(rB[ax][t] for ax in AXES) for t in trans}
    gb["count_hist_a"] = {k: sum(1 for t in trans if kA[t] == k) for k in range(6)}
    gb["count_hist_b"] = {k: sum(1 for t in trans if kB[t] == k) for k in range(6)}
    gb["mean_abs_count_diff"] = sum(abs(kA[t] - kB[t]) for t in trans) / len(trans)
    gb["vector_exact_match_rate"] = sum(
        all(rA[ax][t] == rB[ax][t] for ax in AXES) for t in trans
    ) / len(trans)
    gb["mean_hamming"] = sum(
        sum(rA[ax][t] != rB[ax][t] for ax in AXES) for t in trans
    ) / len(trans)
    # co-reset structure: which axis PAIRS fire together (codex §4 "축 간 상관구조 변화")
    gb["co_reset"] = {}
    for x, y in combinations(AXES, 2):
        ca = sum(1 for t in trans if rA[x][t] and rA[y][t])
        cb = sum(1 for t in trans if rB[x][t] and rB[y][t])
        gb["co_reset"][f"{x[:4]}+{y[:4]}"] = {"a": ca, "b": cb, "delta": cb - ca}
    out["gate_b"] = gb

    # ---------- Gate C: decision equivalence + human-gold non-inferiority ----------
    rows = [json.loads(x) for x in master.read_text().splitlines() if x.strip()]
    gold_bt = [int(r["transition"]) for r in rows if int(r["boundary"])]
    gold = end_turn_labels(n_sent, gold_bt)
    bA = {t for t in trans if kA[t] >= T_STAR}
    bB = {t for t in trans if kB[t] >= T_STAR}
    f, p, r = prf(bB, bA)
    sc_a = score_dialogues([gold], [end_turn_labels(n_sent, sorted(bA))])
    sc_b = score_dialogues([gold], [end_turn_labels(n_sent, sorted(bB))])
    gc = {
        "n_boundaries_a": len(bA), "n_boundaries_b": len(bB), "n_boundaries_gold": len(gold_bt),
        "density_a": len(bA) / len(trans), "density_b": len(bB) / len(trans),
        "boundary_f1_ab": f, "boundary_precision_ab": p, "boundary_recall_ab": r,
        "boundary_f1_ab_tol1": prf_tol(bB, bA),
        "tstar_crossing_rate": sum(
            1 for t in trans if (kA[t] >= T_STAR) != (kB[t] >= T_STAR)
        ) / len(trans),
        "score_vs_human_a": sc_a["score"], "score_vs_human_b": sc_b["score"],
        "score_delta": sc_b["score"] - sc_a["score"],
        "f1_vs_human_a": prf(bA, set(gold_bt))[0], "f1_vs_human_b": prf(bB, set(gold_bt))[0],
    }
    gc["pass"] = (
        gc["boundary_f1_ab"] >= GATE_C["boundary_f1_min"]
        and gc["score_delta"] >= -GATE_C["score_drop_max"]
        and gc["tstar_crossing_rate"] <= GATE_C["tstar_crossing_max"]
    )
    out["gate_c"] = gc

    # ---------- tier 3: space granularity (diagnostic) ----------
    la = {k: A["space"][k]["location"] for k in range(n_sent)}
    lb = {k: B["space"][k]["location"] for k in range(n_sent)}
    ua = sum(v == "unspecified" for v in la.values())
    ub = sum(v == "unspecified" for v in lb.values())
    spec = [k for k in la if la[k] != "unspecified" or lb[k] != "unspecified"]
    both = [k for k in la if la[k] != "unspecified" and lb[k] != "unspecified"]
    n_split = n_same = n_merge = n_diff = 0
    for i, j in combinations(both, 2):
        if la[i] == la[j]:
            n_same += 1
            n_split += lb[i] != lb[j]
        else:
            n_diff += 1
            n_merge += lb[i] == lb[j]
    ca, cb = carry_forward(la, "unspecified"), carry_forward(lb, "unspecified")
    fm = mm = nb = 0
    for x, y in zip(range(n_sent), range(1, n_sent)):
        if None in (ca[x], ca[y], cb[x], cb[y]):
            continue
        nb += 1
        ra_, rb_ = ca[x] != ca[y], cb[x] != cb[y]
        fm += rb_ and not ra_
        mm += ra_ and not rb_
    na, nbs = n_sent - ua, n_sent - ub
    t3 = {
        "unspecified_a_pct": 100 * ua / n_sent, "unspecified_b_pct": 100 * ub / n_sent,
        "unspecified_delta_pp": 100 * (ub - ua) / n_sent,
        "unique_a": len(set(la.values())), "unique_b": len(set(lb.values())),
        "reset_per_specified_a": gb["per_axis"]["place"]["resets_a"] / na if na else 0.0,
        "reset_per_specified_b": gb["per_axis"]["place"]["resets_b"] / nbs if nbs else 0.0,
        "specified_only_agreement": (
            sum(la[k] == lb[k] for k in spec) / len(spec) if spec else None
        ),
        "n_specified_either": len(spec),
        "false_move_rate": fm / nb if nb else None, "missed_move_rate": mm / nb if nb else None,
        "split_rate": n_split / n_same if n_same else None,
        "merge_rate": n_merge / n_diff if n_diff else None,
        "H_b_given_a": cond_entropy([(la[k], lb[k]) for k in both]),
        "H_a_given_b": cond_entropy([(lb[k], la[k]) for k in both]),
        "disagreements": [{"item_id": k, "a": la[k], "b": lb[k]} for k in la if la[k] != lb[k]],
    }
    for tag, d in (("a", a_dir), ("b", b_dir)):
        sl = d / doc / "story_lists.json"
        if sl.exists():
            t3[f"locations_{tag}"] = json.loads(sl.read_text())["lists"]["locations"]
    out["tier3"] = t3
    return out


def report(res: dict[str, Any]) -> None:
    doc = res["doc"]
    print(f"\n{'=' * 76}\n### {doc}\n{'=' * 76}")
    ga = res["gate_a"]
    print(f"[GATE A — run integrity] {'PASS' if ga.get('pass') else 'FAIL'}")
    if ga.get("missing_files"):
        for m in ga["missing_files"]:
            print(f"  missing: {m}")
        return
    print(f"  n_sentences={ga['n_sentences']}  row counts consistent="
          f"{ga['consistent_row_counts']}  contiguous ids={ga['contiguous_ids']}")

    gb = res["gate_b"]
    print("\n[GATE B — 5-axis equivalence]")
    print(f"  {'axis':<26}{'A':>6}{'B':>6}{'F1':>8}{'P':>7}{'R':>7}")
    for ax, v in gb["per_axis"].items():
        print(f"  {ax:<26}{v['resets_a']:>6}{v['resets_b']:>6}"
              f"{v['f1']:>8.3f}{v['precision']:>7.3f}{v['recall']:>7.3f}")
    print(f"  coordinated-count hist A: {gb['count_hist_a']}")
    print(f"  coordinated-count hist B: {gb['count_hist_b']}")
    print(f"  mean|kA-kB|={gb['mean_abs_count_diff']:.3f}  "
          f"5bit exact match={gb['vector_exact_match_rate']:.3f}  "
          f"mean Hamming={gb['mean_hamming']:.3f}")
    worst = sorted(gb["co_reset"].items(), key=lambda kv: -abs(kv[1]["delta"]))[:5]
    print("  co-reset pairs (largest drift): " +
          "  ".join(f"{k}:{v['a']}→{v['b']}" for k, v in worst))

    gc = res["gate_c"]
    print(f"\n[GATE C — decision equivalence] {'PASS' if gc['pass'] else 'FAIL'}")
    print(f"  boundaries      A={gc['n_boundaries_a']}  B={gc['n_boundaries_b']}  "
          f"gold={gc['n_boundaries_gold']}")
    print(f"  A↔B boundary F1 {gc['boundary_f1_ab']:.3f} "
          f"(P={gc['boundary_precision_ab']:.3f} R={gc['boundary_recall_ab']:.3f})  "
          f"±1: {gc['boundary_f1_ab_tol1']:.3f}   [min {GATE_C['boundary_f1_min']}]")
    print(f"  t* crossing     {gc['tstar_crossing_rate']:.3f}   "
          f"[max {GATE_C['tstar_crossing_max']}]")
    print(f"  DTS Score vs HUMAN gold   A={gc['score_vs_human_a']:.4f}  "
          f"B={gc['score_vs_human_b']:.4f}  Δ={gc['score_delta']:+.4f}   "
          f"[max drop {GATE_C['score_drop_max']}]")
    print(f"  boundary F1 vs human      A={gc['f1_vs_human_a']:.3f}  B={gc['f1_vs_human_b']:.3f}")

    t3 = res["tier3"]
    print("\n[TIER 3 — place granularity, DIAGNOSTIC ONLY]")
    print(f"  unspecified  A={t3['unspecified_a_pct']:.1f}%  B={t3['unspecified_b_pct']:.1f}%  "
          f"(Δ={t3['unspecified_delta_pp']:+.1f}pp)  [diag limit ±{DIAG['unspecified_pp']}]")
    print(f"  unique       A={t3['unique_a']}  B={t3['unique_b']}")
    print(f"  reset/spec   A={t3['reset_per_specified_a']:.2f}  "
          f"B={t3['reset_per_specified_b']:.2f}")
    ag = t3["specified_only_agreement"]
    print(f"  specified-only agreement {ag:.1%} (n={t3['n_specified_either']}) "
          f"— LOWER BOUND, string-literal" if ag is not None else "  specified-only n/a")
    for k in ("false_move_rate", "missed_move_rate", "split_rate", "merge_rate"):
        v = t3[k]
        print(f"  {k:<20} {v:.3f}" if v is not None else f"  {k:<20} n/a")
    print(f"  H(B|A)={t3['H_b_given_a']:.3f}  H(A|B)={t3['H_a_given_b']:.3f} bits")
    if "locations_a" in t3:
        print(f"  Stage1 A ({len(t3['locations_a'])}): {t3['locations_a']}")
        print(f"  Stage1 B ({len(t3['locations_b'])}): {t3['locations_b']}")
    if t3["disagreements"]:
        print(f"  place disagreements ({len(t3['disagreements'])}):")
        for d in t3["disagreements"][:12]:
            print(f"    item {d['item_id']:>4}: A={d['a']:<26} B={d['b']}")
        if len(t3["disagreements"]) > 12:
            print(f"    … +{len(t3['disagreements']) - 12} more")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", action="append")
    ap.add_argument("--a", type=Path)
    ap.add_argument("--b", type=Path)
    ap.add_argument("--master", type=Path)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    jobs = []
    for d in (DEV_DOCS if args.all else (args.doc or DEV_DOCS)):
        jobs.append((
            d,
            args.a or REPO / f"outputs/eacl_phase0/{d}_michelmann_s5",
            args.b or REPO / f"outputs/eacl_phase0/{d}_michelmann_s5_dsv4pro",
            args.master or REPO / f"outputs/eacl_phase0/{d}_michelmann_master.jsonl",
        ))

    results = []
    for doc, a, b, m in jobs:
        res = compare_doc(doc, a, b, m)
        results.append(res)
        report(res)

    done = [r for r in results if r["gate_a"].get("pass")]
    if done:
        print(f"\n{'=' * 76}\nVERDICT (per-doc; codex: judge by WORST doc, not the mean)")
        for r in done:
            print(f"  {r['doc']:<10} gate C {'PASS' if r['gate_c']['pass'] else 'FAIL'}  "
                  f"(A↔B F1={r['gate_c']['boundary_f1_ab']:.3f}, "
                  f"ΔScore={r['gate_c']['score_delta']:+.4f})")

    if args.json and results:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
