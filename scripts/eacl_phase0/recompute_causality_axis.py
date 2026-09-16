#!/usr/bin/env python3
r"""앵커 인과 축 규칙 교체 — **API 0**. 저장된 `aspect` 라벨에서 reset 컬럼만 다시 계산한다.

임베딩(`phase1_emb.npy`)은 문장도 인코더도 그대로이므로 **재생성하지 않고 심볼릭 링크로 재사용**한다.
바뀌는 것은 `{split}_index.npz` 의 `resets` 5번째 열(causality) 하나뿐이다.

규칙:
  aspcomp  : reset = (b.aspect == "completive")            # F3
  nonong   : reset = (b.aspect != "ongoing")               # F4
  frozen   : reset = (a.event_type != b.event_type) or (b.aspect == "inceptive")   # 검증용 재현

🔒 기존 산출(`phase1_index.npz`)은 읽기만 한다. 새 split 이름으로만 쓴다.

    .venv/bin/python scripts/eacl_phase0/recompute_causality_axis.py --rule aspcomp
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
DATA = REPO / "outputs/experiments/m4m5_phase1_anchor/data"
SRC = REPO / "outputs/eacl_phase0/phase1_all_v31"
CAUSALITY_COL = 3   # AXES = [protagonist_present, space, time, causality, intentionality]


def rule_fn(rule: str):
    if rule == "aspcomp":
        return lambda a, b: int(b.get("aspect") == "completive")
    if rule == "nonong":
        return lambda a, b: int(b.get("aspect") != "ongoing")
    if rule == "frozen":
        return lambda a, b: int(a.get("event_type") != b.get("event_type")
                                or b.get("aspect") == "inceptive")
    raise SystemExit(f"🔴 알 수 없는 규칙: {rule}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True, choices=["aspcomp", "nonong", "frozen"])
    ap.add_argument("--base-split", default="phase1")
    a = ap.parse_args()
    fn = rule_fn(a.rule)
    split = a.base_split if a.rule == "frozen" else f"{a.base_split}_{a.rule}"
    out_ix = DATA / f"{split}_index.npz"
    if a.rule != "frozen" and out_ix.exists():
        raise SystemExit(f"🔴 이미 있다: {out_ix.name} — 덮어쓰지 않는다")

    t0 = time.time()
    ix = np.load(DATA / f"{a.base_split}_index.npz", allow_pickle=True)
    doc_ids = [str(x) for x in ix["doc_ids"]]
    so, to = ix["sent_offsets"], ix["trans_offsets"]
    R = ix["resets"].copy()
    old = R[:, CAUSALITY_COL].mean()

    n_miss = 0
    for i, doc in enumerate(doc_ids):
        p = SRC / doc / "causality_states.jsonl"
        if not p.exists():
            n_miss += 1
            continue
        rows = sorted((json.loads(l) for l in p.read_text().splitlines() if l.strip()),
                      key=lambda r: int(re.search(r"(\d+)$", str(r["item_id"])).group(1)))
        t0i, t1i = int(to[i]), int(to[i + 1])
        n_tr = t1i - t0i
        if len(rows) < n_tr + 1:
            n_miss += 1
            continue
        R[t0i:t1i, CAUSALITY_COL] = np.asarray(
            [fn(rows[t], rows[t + 1]) for t in range(n_tr)], np.int8)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1:,}/{len(doc_ids):,} …", flush=True)

    new = R[:, CAUSALITY_COL].mean()
    bnd_old = (ix["resets"].sum(1) >= 3).mean()
    bnd_new = (R.sum(1) >= 3).mean()
    print(f"\n규칙 {a.rule} · 문서 {len(doc_ids):,} (라벨 누락 {n_miss})")
    print(f"  인과 발화율 {old:.4f} -> {new:.4f}")
    print(f"  앵커 경계율(count≥3) {bnd_old:.4f} -> {bnd_new:.4f}")
    print(f"  동시 리셋 분포 " + " ".join(
        f"{v}:{(R.sum(1)==v).mean():.4f}" for v in range(6)))

    if a.rule == "frozen":
        same = float((R[:, CAUSALITY_COL] == ix["resets"][:, CAUSALITY_COL]).mean())
        print(f"  🟢 동결 재현 검증: 기존 열과 일치율 {same:.6f}")
        return

    np.savez(out_ix, doc_ids=np.array(doc_ids), sent_offsets=so, trans_offsets=to,
             resets=R, states=ix["states"],
             **({"state_cols": ix["state_cols"]} if "state_cols" in ix.files else {}))
    emb = DATA / f"{split}_emb.npy"
    if not emb.exists():
        emb.symlink_to(DATA / f"{a.base_split}_emb.npy")   # 임베딩 재사용(재인코딩 없음)
    print(f"\n-> {out_ix.name} (+ {emb.name} 심볼릭 링크) · {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
