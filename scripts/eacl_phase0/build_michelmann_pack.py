#!/usr/bin/env python3
"""Build Phase-0 masters from the Michelmann GPT_event_share dataset (Pieman/Monkey/Tunnel).

Decision (ledger, 2026-07-13): unify ALL THREE stories on this single dataset with the
**author-filtered** human event boundaries (replicated from the paper's
`loaders.Boundary_Loader.get_Manoj_filtered_bounds_ms`), so dev (Pieman) and test
(Monkey/Tunnel) share one boundary-derivation rule.

Frozen sentence policy (same as Pieman dev): punctuation split
(build_narrative_pack.split_sentences) + word-onset alignment (difflib). A gold boundary
time (ms) maps to the last word with onset<=t -> that word's sentence is segment-final ->
boundary at the following transition (collisions collapse to one, as in Pieman).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import scipy.io as sio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pieman_pack import split_sentences, align_words_to_sentences  # noqa: E402

# --- author-filtered human boundaries (ms), replicated from paper loaders.py ---
_PIEMAN = [27118, 42021, 53883, 72791, 97259, 117622, 133494, 185887, 203434, 208573,
           226955, 268269, 279488, 289554, 308024, 329378, 381957, 423366, 431517]
_PIEMAN_FILT = [x for i, x in enumerate(_PIEMAN) if i != 18]  # delete idx 18 -> 18 bounds

_MONKEY_SEG90 = [21700, 69591, 98503, 146510, 175480, 183933, 187968, 234272, 242959,
                 243139, 243231, 284866, 296304, 315029, 359647, 386838, 387207, 387720,
                 387781, 388185, 388974, 442772, 456868, 466100, 466384, 469985, 541682,
                 541714, 581722, 650849, 832964, 881392, 890426, 908610, 914337, 926219,
                 1037478, 1091764, 1115995, 1141312, 1167122, 1167208, 1211324, 1211861,
                 1211987, 1212231, 1212743, 1212872, 1213129, 1306814, 1306865, 1315393,
                 1369944, 1444632, 1468867, 1468923, 1560030, 1631446, 1670747, 1778312,
                 1797724]
_MONKEY_DEL = {8, 9, 10, 11, 15, 16, 18, 19, 20, 14, 22, 23, 26, 29, 40, 42, 43, 44, 45,
               46, 48, 49, 54, 35, 37, 28, 30, 32, 33, 34}
_MONKEY_FILT = [x for i, x in enumerate(_MONKEY_SEG90) if i not in _MONKEY_DEL]

_TUNNEL_TR = [24, 25, 26, 53, 68, 83, 84, 85, 86, 87, 110, 111, 112, 113, 114, 142, 143,
              163, 164, 165, 166, 210, 211, 212, 213, 214, 243, 244, 245, 246, 304, 305,
              306, 307, 308, 419, 420, 421, 422, 454, 455, 499, 500, 501, 502, 503, 541,
              542, 543, 544, 545, 567, 568, 569, 654, 919, 920, 921, 922, 923, 1000, 1001,
              1002, 1003, 1007, 1023]
_TUNNEL_DEL = {0, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 20, 21, 23, 24, 25, 26, 27,
               29, 30, 31, 32, 34, 35, 36, 38, 39, 41, 42, 44, 45, 46, 47, 48, 50, 51, 53,
               55, 56, 58, 59, 60, 61, 63}
_TUNNEL_FILT = [x * 1.5 * 1000 for i, x in enumerate(_TUNNEL_TR) if i not in _TUNNEL_DEL]

GOLD_MS = {"pieman": _PIEMAN_FILT, "monkey": _MONKEY_FILT, "tunnel": _TUNNEL_FILT}

# gold B: raw button-press boundary peaks + 5s merge (phoebsc boundaryDetection rule,
# borrowed to minimize researcher DOF). Raw sources are the {Story}Bounds*.txt files (ms).
_RAW_FILE = {"pieman": "PiemanBoundsRun2.txt", "monkey": "MonkeyBounds.txt",
             "tunnel": "TunnelBounds.txt"}


def merge_5s(times_ms: list[float], window_ms: float = 5000.0) -> list[float]:
    """Cluster boundary times within `window_ms`; keep the cluster mean (phoebsc 5s merge)."""
    ts = sorted(times_ms)
    if not ts:
        return []
    clusters, cur = [], [ts[0]]
    for t in ts[1:]:
        if t - cur[-1] <= window_ms:
            cur.append(t)
        else:
            clusters.append(cur); cur = [t]
    clusters.append(cur)
    return [sum(c) / len(c) for c in clusters]


def raw_gold_ms(story: str, sd: Path) -> list[float]:
    raw = [float(x) for x in (sd / _RAW_FILE[story]).read_text(encoding="utf-8-sig").split()]
    return merge_5s(raw)


def load_mat_col(path: Path, key: str):
    m = sio.loadmat(str(path))
    k = key if key in m else [x for x in m if not x.startswith("__")][0]
    return [v for v in m[k].flatten()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("story", choices=["pieman", "monkey", "tunnel"])
    ap.add_argument("--gold", choices=["filtered", "raw"], default="filtered",
                    help="filtered=author-curated (gold A); raw=raw peaks + 5s merge (gold B)")
    ap.add_argument("--data-root", type=Path, default=Path("data/eacl_phase0"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    S = args.story
    ddir = args.data_root / f"{S}_michelmann"
    sd = ddir / "sourcedata"
    title = S.capitalize()
    gold_ms = GOLD_MS[S] if args.gold == "filtered" else raw_gold_ms(S, sd)
    suffix = "" if args.gold == "filtered" else "_raw"

    words = [str(w[0]).strip().lower() if hasattr(w, "__len__") and len(w) else str(w).strip().lower()
             for w in load_mat_col(sd / "words.mat", "words")]
    onsets = [float(x) for x in load_mat_col(sd / "word_onsets.mat", "onsets")]
    offsets = [float(x) for x in load_mat_col(sd / "word_offsets.mat", "offsets")]
    n = min(len(words), len(onsets), len(offsets))
    words, onsets, offsets = words[:n], onsets[:n], offsets[:n]

    text = (ddir / f"{title}Clean.txt").read_text(encoding="utf-8-sig")
    sentences = split_sentences(text)
    word_sent = align_words_to_sentences(words, sentences)

    def time_to_word(t_s: float) -> int:
        # last word with a valid (non-NaN) onset <= t; NaN onsets (forced-alignment
        # failures, ~3.7% in Tunnel) are skipped, not treated as a stop.
        lo = 0
        for i, on in enumerate(onsets):
            if on == on and on <= t_s:  # on==on filters NaN
                lo = i
        return lo

    boundary_after: dict[int, int] = {}
    for ms in gold_ms:
        wi = time_to_word(ms / 1000.0)
        sid = word_sent[wi]
        boundary_after[sid] = boundary_after.get(sid, 0) + 1
    collisions = {s: c for s, c in boundary_after.items() if c > 1}
    dropped_last = boundary_after.pop(len(sentences) - 1, 0)

    # per-sentence time span
    sent_on = [None] * len(sentences); sent_off = [None] * len(sentences)
    for i, sid in enumerate(word_sent):
        if sent_on[sid] is None:
            sent_on[sid] = onsets[i]
        sent_off[sid] = offsets[i]

    rows = []
    for s in range(1, len(sentences)):
        rows.append({
            "doc_id": S, "transition": s,
            "previous": sentences[s - 1], "current": sentences[s],
            "previous_ko_reference": "", "current_ko_reference": "",
            "boundary": int((s - 1) in boundary_after),
            "prev_onset_s": sent_on[s - 1], "cur_onset_s": sent_on[s],
            "idm_score": None, "protagonist": None, "place": None, "time": None,
            "causality": None, "goal": None, "notes": "",
        })
    out = args.out or (Path("outputs/eacl_phase0") / f"{S}_michelmann{suffix}_master.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    meta = {"doc_id": S, "source": "michelmann_gpt_event_share",
            "gold": args.gold + ("" if args.gold == "raw" else "-author-filtered"),
            "n_words": n, "n_sentences": len(sentences), "n_transitions": len(rows),
            "n_gold_boundaries_ms": len(gold_ms),
            "n_boundary_transitions": sum(r["boundary"] for r in rows),
            "collisions": collisions, "dropped_at_final": dropped_last,
            "boundary_transitions": [r["transition"] for r in rows if r["boundary"]]}
    (out.parent / f"{S}_michelmann{suffix}_pack_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
