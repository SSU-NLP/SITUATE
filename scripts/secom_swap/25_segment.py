"""Density-matched segmentation — ONLINE (deployable) and OFFLINE (diagnostic).

Methods (per-transition raw signal, higher = more boundary-like):
  m1        CosDrift delta_eff                 (causal, online)
  surprise  gpt2-xl proxy-LM surprise          (causal, online)
  m3        5-axis coordinated reset count 0-5 (Stage1 lists are global -> offline instrument; caveat)
  hybrid    normalized-sum: z(surprise)+z(count)

Segmentation is per-session (SeCom convention): session breaks are automatic
boundaries; running statistics reset each session (fresh segmenter per session).

Two modes:
  --mode online   Streaming rule: at transition t, boundary iff running-z(raw_t) >= c
                  (z uses PAST transitions only, Welford; warmup W). For hybrid the
                  decision stat is run-z(surp)+run-z(count). c is a per-method scalar
                  CALIBRATED ON DEV to hit the common target density; frozen for test.
                  -> online, no future lookahead, density matched *on average*.
  --mode offline  Global top-N per conversation (exact density match). Sees the whole
                  conversation -> NOT online; a placement/ranking diagnostic only.

Output: segments/<method>__<mode>.json  + meta (c, per-conv density, split).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/eacl_phase0"))
from llm_s5_states import (  # noqa: E402
    carry_forward, reset_causality, reset_goal, reset_perspective,
    reset_place, reset_time,
)

BASE = REPO_ROOT / "outputs/experiments/2026-07-15_locomo_segmenter_swap"
DATA = BASE / "data/locomo_turns.jsonl"
SCORES = BASE / "scores"
M3STATES = BASE / "m3_states"
OUTDIR = BASE / "segments"
AXES5 = ["protagonist", "place", "time", "causality", "goal"]

DEV = ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43"]
TEST = ["conv-44", "conv-47", "conv-48", "conv-49", "conv-50"]
WARMUP = 3


def m3_counts(cid, n_turns, sess):
    d = M3STATES / cid
    if not (d / "goal.jsonl").exists():
        return None

    def load(ax):
        return {int(json.loads(l)["item_id"]): json.loads(l) for l in (d / f"{ax}.jsonl").open()}

    S = {ax: load(ax) for ax in AXES5}
    if any(len(S[ax]) < n_turns for ax in AXES5):
        return None
    S["place"] = carry_forward(S["place"], "location", "unspecified")
    S["goal"] = carry_forward(S["goal"], "active_goal", "none")
    out = np.full(n_turns, np.nan)
    for i in range(1, n_turns):
        if sess[i] != sess[i - 1]:
            continue
        out[i] = float(
            reset_perspective(S["protagonist"][i - 1], S["protagonist"][i])
            + reset_place(S["place"][i - 1], S["place"][i])
            + reset_time(S["time"][i - 1], S["time"][i])
            + reset_causality(S["causality"][i - 1], S["causality"][i])
            + reset_goal(S["goal"][i - 1], S["goal"][i])
        )
    return out


def load_conv(cid, m3_variant="full"):
    turns = json.loads((SCORES / f"{cid}.turns.json").read_text())
    z = np.load(SCORES / f"{cid}.npz")
    cos, surp, sess = z["cos"], z["surp"], z["sess"]
    n = len(turns)
    if m3_variant == "prefix":
        p = SCORES / f"{cid}.m3prefix.json"
        if not p.exists():
            m3 = None
        else:
            arr = json.loads(p.read_text())
            m3 = np.array([np.nan if v is None else float(v) for v in arr])
    else:
        m3 = m3_counts(cid, n, sess)
    return turns, cos, surp, sess, m3


class Welford:
    __slots__ = ("n", "mean", "M2")

    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0

    def z(self, x):
        if self.n < WARMUP:
            return None
        var = self.M2 / self.n
        sd = var ** 0.5
        return 0.0 if sd == 0 else (x - self.mean) / sd

    def push(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)


def online_boundaries(method, cos, surp, m3, sess, c):
    """Return set of internal transition indices flagged as boundaries (online)."""
    n = len(sess)
    bnd = set()
    if method == "hybrid":
        ws, wc = Welford(), Welford()
    else:
        w = Welford()
        raw = {"m1": cos, "surprise": surp, "m3": m3}[method]
    for i in range(1, n):
        if sess[i] != sess[i - 1]:
            if method == "hybrid":
                ws, wc = Welford(), Welford()
            else:
                w = Welford()
            continue
        if method == "hybrid":
            if not (np.isfinite(surp[i]) and np.isfinite(m3[i])):
                continue
            zs, zc = ws.z(surp[i]), wc.z(m3[i])
            stat = None if (zs is None or zc is None) else zs + zc
            ws.push(surp[i]); wc.push(m3[i])
        else:
            if not np.isfinite(raw[i]):
                continue
            stat = w.z(raw[i]); w.push(raw[i])
        if stat is not None and stat >= c:
            bnd.add(i)
    return bnd


def offline_boundaries(method, cos, surp, m3, sess, target_tps, n_sessions, n_turns):
    target_segs = max(n_sessions, round(n_turns / target_tps))
    n_internal = max(0, target_segs - n_sessions)
    if method == "m1":
        score = cos
    elif method == "surprise":
        score = surp
    elif method == "m3":
        score = m3.copy()
        cand = [i for i in range(1, n_turns) if sess[i] == sess[i - 1] and np.isfinite(cos[i])]
        cvals = np.array([cos[i] for i in cand]); rng = (cvals.max() - cvals.min()) or 1.0
        for i in cand:
            if np.isfinite(score[i]):
                score[i] = score[i] + 1e-3 * (cos[i] - cvals.min()) / rng
    else:  # hybrid: global z-sum
        def gz(x):
            m = [x[i] for i in range(1, n_turns) if sess[i] == sess[i - 1] and np.isfinite(x[i])]
            mu, sd = np.mean(m), np.std(m) or 1.0
            return mu, sd
        mus, sds = gz(surp); muc, sdc = gz(m3)
        score = np.full(n_turns, np.nan)
        for i in range(1, n_turns):
            if sess[i] == sess[i - 1] and np.isfinite(surp[i]) and np.isfinite(m3[i]):
                score[i] = (surp[i] - mus) / sds + (m3[i] - muc) / sdc
    cand = [i for i in range(1, n_turns)
            if sess[i] == sess[i - 1] and np.isfinite(score[i])]
    cand.sort(key=lambda i: score[i], reverse=True)
    return set(cand[:n_internal])


def build_segments(turns, sess, bnd):
    segs, cur = [], [turns[0]]
    for i in range(1, len(turns)):
        if sess[i] != sess[i - 1] or i in bnd:
            segs.append(cur); cur = [turns[i]]
        else:
            cur.append(turns[i])
    segs.append(cur)
    return segs


def density(cids, data, method, mode, c=None, target_tps=None):
    tot_turns = tot_segs = 0
    for cid in cids:
        turns, cos, surp, sess, m3 = data[cid]
        n = len(turns); ns = int(len(set(sess.tolist())))
        if mode == "online":
            bnd = online_boundaries(method, cos, surp, m3, sess, c)
        else:
            bnd = offline_boundaries(method, cos, surp, m3, sess, target_tps, ns, n)
        segs = build_segments(turns, sess, bnd)
        tot_turns += n; tot_segs += len(segs)
    return tot_turns / tot_segs


def calibrate_c(dev_ids, data, method, target_tps):
    """Pick scalar c so that dev density ~ target_tps (grid search)."""
    best, best_gap = 0.0, 1e9
    for c in np.arange(-0.5, 3.01, 0.1):
        tps = density(dev_ids, data, method, "online", c=float(c))
        gap = abs(tps - target_tps)
        if gap < best_gap:
            best_gap, best = gap, float(c)
    return round(best, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["m1", "surprise", "m3", "hybrid", "m3prefix", "hybridprefix"])
    ap.add_argument("--mode", required=True, choices=["online", "offline"])
    ap.add_argument("--target_tps", type=float, default=5.0)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    convs = [json.loads(l) for l in DATA.open()]
    cids = [c["conversation_id"] for c in convs]
    variant = "prefix" if args.method in ("m3prefix", "hybridprefix") else "full"
    data = {cid: load_conv(cid, variant) for cid in cids}
    # normalized method for the boundary logic ("m3prefix"->"m3", "hybridprefix"->"hybrid")
    base = {"m3prefix": "m3", "hybridprefix": "hybrid"}.get(args.method, args.method)
    if base in ("m3", "hybrid"):
        miss = [cid for cid in cids if data[cid][4] is None]
        if miss:
            raise SystemExit(f"M3 states incomplete for {miss}")

    meta = {"method": args.method, "mode": args.mode, "target_tps": args.target_tps,
            "dev": DEV, "test": TEST}
    c = None
    if args.mode == "online":
        c = calibrate_c(DEV, data, base, args.target_tps)
        meta["c_calibrated_on_dev"] = c
        meta["dev_density"] = round(density(DEV, data, base, "online", c=c), 3)
        meta["test_density"] = round(density(TEST, data, base, "online", c=c), 3)

    out_rows = []
    per_conv_density = {}
    for cid in cids:
        turns, cos, surp, sess, m3 = data[cid]
        n = len(turns); ns = int(len(set(sess.tolist())))
        if args.mode == "online":
            bnd = online_boundaries(base, cos, surp, m3, sess, c)
        else:
            bnd = offline_boundaries(base, cos, surp, m3, sess, args.target_tps, ns, n)
        segs = build_segments(turns, sess, bnd)
        out_rows.append({"conversation_id": cid, "segments": segs})
        per_conv_density[cid] = round(n / len(segs), 2)
    meta["per_conv_density"] = per_conv_density
    save = OUTDIR / f"{args.method}__{args.mode}.json"
    save.write_text(json.dumps(out_rows, ensure_ascii=False))
    (OUTDIR / f"{args.method}__{args.mode}.meta.json").write_text(json.dumps(meta, indent=2))
    tt = sum(len(data[r['conversation_id']][0]) for r in out_rows)
    ts = sum(len(r["segments"]) for r in out_rows)
    extra = f"c={c} dev_tps={meta.get('dev_density')} test_tps={meta.get('test_density')}" if c is not None else ""
    print(f"{args.method}/{args.mode}: {ts} segs / {tt} turns = {tt/ts:.2f} tps  {extra}")


if __name__ == "__main__":
    main()
