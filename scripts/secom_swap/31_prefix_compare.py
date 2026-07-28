"""full-M3 vs prefix-online-M3 evidence recall, on the SAME conversations.

QA needs the LLM (budget exhausted) but recall is LLM-free, so we compare the two
M3 variants at matched online density on the test convs that have prefix states
(44,47,48; 49/50 missing due to budget). Online threshold c is calibrated on dev
(all dev convs have prefix). Reports recall@k per category, full vs prefix.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/secom_swap"))
seg = __import__("25_segment")

BASE = REPO_ROOT / "outputs/experiments/2026-07-15_locomo_segmenter_swap"
DATA = BASE / "data/locomo_turns.jsonl"
SCORES = BASE / "scores"
CATNAME = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
DEV = ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43"]
TEST_AVAIL = ["conv-44", "conv-47", "conv-48"]  # 49,50 missing (budget)
TOPKS = [1, 2, 3, 5]


def seg_text(s):
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in s)


def segments_for(variant, target_ids, enc, convs_qa):
    """variant: 'full' or 'prefix'. Returns recall dict per category."""
    all_cids = DEV + TEST_AVAIL
    data = {cid: seg.load_conv(cid, variant) for cid in all_cids}
    c = seg.calibrate_c(DEV, data, "m3", 5.0)
    dev_tps = round(seg.density(DEV, data, "m3", "online", c=c), 3)
    rec = {k: defaultdict(list) for k in TOPKS}
    seg_counts = []
    for cid in target_ids:
        turns, cos, surp, sess, m3 = data[cid]
        bnd = seg.online_boundaries("m3", cos, surp, m3, sess, c)
        segs = seg.build_segments(turns, sess, bnd)
        seg_counts.append((len(turns), len(segs)))
        seg_txt = [seg_text(s) for s in segs]
        seg_dia = [set(t["dia_id"] for t in s) for s in segs]
        semb = np.asarray(enc.encode(seg_txt, normalize_embeddings=True, show_progress_bar=False))
        qa = convs_qa[cid]
        qs = [q["question"] for q in qa]
        qemb = np.asarray(enc.encode(qs, normalize_embeddings=True, show_progress_bar=False))
        order = np.argsort(-(qemb @ semb.T), axis=1)
        for qi, q in enumerate(qa):
            gold = set(q["evidence"])
            if not gold:
                continue
            for k in TOPKS:
                got = set().union(*[seg_dia[j] for j in order[qi][:k]]) if k else set()
                rec[k][q["category"]].append(len(gold & got) / len(gold))
    tt = sum(a for a, _ in seg_counts); ts = sum(b for _, b in seg_counts)
    return rec, c, dev_tps, tt / ts


def main() -> None:
    convs = {json.loads(l)["conversation_id"]: json.loads(l) for l in DATA.open()}
    convs_qa = {cid: convs[cid]["qa"] for cid in convs}
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1",
                              device="cuda" if torch.cuda.is_available() else "cpu")

    res = {}
    for variant in ["full", "prefix"]:
        rec, c, dev_tps, test_tps = segments_for(variant, TEST_AVAIL, enc, convs_qa)
        res[variant] = {"rec": rec, "c": c, "dev_tps": dev_tps, "test_tps": test_tps}
        print(f"[{variant}] c={c} dev_tps={dev_tps} test_tps(3conv)={test_tps:.2f}")

    def r(variant, k, cat):
        v = res[variant]["rec"][k][cat]
        return float(np.mean(v)) if v else None

    print(f"\n=== full-M3 vs prefix-online-M3 · recall@k · TEST subset {TEST_AVAIL} ===")
    cats = sorted(CATNAME)
    print(f"{'':13} " + "  ".join(f"full@{k} pref@{k}" for k in [1, 5]))
    for cat in cats:
        f1, p1, f5, p5 = r("full", 1, cat), r("prefix", 1, cat), r("full", 5, cat), r("prefix", 5, cat)
        def fmt(x): return f"{x:.3f}" if x is not None else "  -  "
        print(f"{CATNAME[cat]:13} {fmt(f1)}   {fmt(p1)}     {fmt(f5)}   {fmt(p5)}")
    for k in TOPKS:
        af = [v for cat in cats for v in res["full"]["rec"][k][cat]]
        ap = [v for cat in cats for v in res["prefix"]["rec"][k][cat]]
        print(f"overall recall@{k}: full={np.mean(af):.3f}  prefix={np.mean(ap):.3f}  "
              f"Δ={np.mean(ap)-np.mean(af):+.3f}")
    out = {"test_subset": TEST_AVAIL,
           "full": {"c": res["full"]["c"], "test_tps": res["full"]["test_tps"],
                    "recall": {str(k): {str(cat): r("full", k, cat) for cat in cats} for k in TOPKS},
                    "overall": {str(k): float(np.mean([v for cat in cats for v in res["full"]["rec"][k][cat]])) for k in TOPKS}},
           "prefix": {"c": res["prefix"]["c"], "test_tps": res["prefix"]["test_tps"],
                      "recall": {str(k): {str(cat): r("prefix", k, cat) for cat in cats} for k in TOPKS},
                      "overall": {str(k): float(np.mean([v for cat in cats for v in res["prefix"]["rec"][k][cat]])) for k in TOPKS}}}
    (BASE / "prefix_vs_full_recall.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> prefix_vs_full_recall.json")


if __name__ == "__main__":
    main()
