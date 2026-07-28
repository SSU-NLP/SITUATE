"""Aggregate recall/*.json + qa/*.json + segment meta into consolidated tables."""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "outputs/experiments/2026-07-15_locomo_segmenter_swap"
METHODS = ["m1", "surprise", "m3", "hybrid"]
MODES = ["online", "offline"]
CATS = [("1", "multi-hop"), ("2", "temporal"), ("3", "open-domain"),
        ("4", "single-hop"), ("5", "adversarial")]
LABEL = {"m1": "M1-CosDrift", "surprise": "surprise-gpt2xl", "m3": "M3-5axis", "hybrid": "Hybrid"}


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def main() -> None:
    lines = []
    P = lines.append

    # 1) density
    P("### Density (turns/segment) — online c calibrated on dev, applied to test\n")
    P("| method | mode | c(dev) | dev tps | test tps |")
    P("|---|---|---|---|---|")
    for m in METHODS:
        meta = load(BASE / f"segments/{m}__online.meta.json")
        if meta:
            P(f"| {LABEL[m]} | online | {meta.get('c_calibrated_on_dev')} | "
              f"{meta.get('dev_density')} | {meta.get('test_density')} |")
    P("")

    # 2) recall@k overall by method x mode (test)
    P("### Evidence recall@k — TEST split, overall (density-matched)\n")
    P("| method | mode | r@1 | r@2 | r@3 | r@5 |")
    P("|---|---|---|---|---|---|")
    for m in METHODS:
        for mode in MODES:
            r = load(BASE / f"recall/{m}__{mode}.json")
            if not r:
                continue
            ov = r["splits"]["test"]["overall"]
            P(f"| {LABEL[m]} | {mode} | " + " | ".join(f"{ov[f'recall@{k}']:.3f}" for k in [1, 2, 3, 5]) + " |")
    P("")

    # 3) recall@5 per category (test, online)
    P("### Evidence recall@5 by category — TEST, ONLINE (deployable)\n")
    hdr = "| method | " + " | ".join(n for _, n in CATS) + " | overall |"
    P(hdr); P("|" + "---|" * (len(CATS) + 2))
    for m in METHODS:
        r = load(BASE / f"recall/{m}__online.json")
        if not r:
            continue
        bc = r["splits"]["test"]["by_category"]
        cells = " | ".join(f"{bc[c]['recall@5']:.3f}" if bc[c]['recall@5'] is not None else "-" for c, _ in CATS)
        ov = r["splits"]["test"]["overall"]["recall@5"]
        P(f"| {LABEL[m]} | {cells} | **{ov:.3f}** |")
    P("")

    # 4) GPT4Score by category (test, online)
    P("### Final answer GPT4Score (×10, /100) by category — TEST, ONLINE\n")
    have_qa = any((BASE / f"qa/{m}__online.json").exists() for m in METHODS)
    if not have_qa:
        P("_(QA still running)_\n")
    else:
        hdr = "| method | " + " | ".join(n for _, n in CATS) + " | overall | ctx_tok |"
        P(hdr); P("|" + "---|" * (len(CATS) + 3))
        for m in METHODS:
            q = load(BASE / f"qa/{m}__online.json")
            if not q:
                continue
            bc = q["by_category"]
            cells = " | ".join(f"{bc[c]['gpt4score_x10']:.1f}" if bc[c]['gpt4score_x10'] is not None else "-" for c, _ in CATS)
            ov = q["overall_gpt4score_x10"]
            P(f"| {LABEL[m]} | {cells} | **{ov:.2f}** | {q['avg_ctx_tokens']:.0f} |")
    P("")

    out = "\n".join(lines)
    (BASE / "SUMMARY_TABLES.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
