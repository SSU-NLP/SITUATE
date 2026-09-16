#!/usr/bin/env python3
r"""앵커 축 규칙 교체 (인과 × 의도 교차) — **API 0**. 저장된 라벨에서 reset 열만 재계산.

인과: frozen(현행) · aspcomp(`b.aspect == completive`)
의도: frozen(현행 = carry∞ + none 가드) · i2(**carry 없음 + 채택만**: `a=='none' and b!='none'`)

임베딩은 심볼릭 링크로 재사용(재인코딩 0). 기존 산출은 읽기만 한다.

    .venv/bin/python scripts/eacl_phase0/recompute_axes.py --causality aspcomp --intent i2
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "outputs/experiments/m4m5_phase1_anchor/data"
SRC = REPO / "outputs/eacl_phase0/phase1_all_v31"   # 기본 = D-B15 Qwen 앵커
COL = {"causality": 3, "intentionality": 4}


def read_states(doc: str, axis: str, src: Path | None = None):
    p = (src or SRC) / doc / f"{axis}_states.jsonl"
    if not p.exists():
        return None
    return sorted((json.loads(l) for l in p.read_text().splitlines() if l.strip()),
                  key=lambda r: int(re.search(r"(\d+)$", str(r["item_id"])).group(1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--causality", default="frozen",
                    choices=["frozen", "aspcomp", "nonong", "chainstrict", "chaincf"])
    ap.add_argument("--chain-src", default="outputs/eacl_phase0/phase1_chain_gptoss_or_min_v31",
                    help="사슬 라벨 디렉토리(문서별 causalchain_states.jsonl)")
    ap.add_argument("--intent", default="frozen", choices=["frozen", "i2"])
    # ── 다른 라벨러 앵커에도 쓰기 위한 파라미터화 (D-B57) ────────────────────
    #  원래는 SRC(=Qwen 앵커)와 split "phase1" 이 하드코딩이라 이 스크립트가
    #  D-B15 코퍼스 전용이었다. 축 재조작화 자체는 라벨러와 무관하므로 열어 둔다.
    ap.add_argument("--src-dir", default="", help="축 상태 파일 디렉터리(기본 = phase1_all_v31)")
    ap.add_argument("--from-split", default="phase1", help="입력 split 이름(기본 phase1)")
    a = ap.parse_args()
    src = Path(a.src_dir) if a.src_dir else SRC
    tag = "_".join(x for x in [None if a.causality == "frozen" else a.causality,
                               None if a.intent == "frozen" else a.intent] if x)
    if not tag:
        raise SystemExit("🔴 둘 다 frozen 이면 기존 phase1 과 같다")
    split = f"{a.from_split}_{tag}"
    out_ix = DATA / f"{split}_index.npz"
    if out_ix.exists():
        raise SystemExit(f"🔴 이미 있다: {out_ix.name}")

    t0 = time.time()
    ix = np.load(DATA / f"{a.from_split}_index.npz", allow_pickle=True)
    doc_ids = [str(x) for x in ix["doc_ids"]]
    to = ix["trans_offsets"]
    R = ix["resets"].copy()
    old = R[:, COL["causality"]].mean(), R[:, COL["intentionality"]].mean()

    miss = 0
    for i, doc in enumerate(doc_ids):
        t0i, t1i = int(to[i]), int(to[i + 1])
        n_tr = t1i - t0i
        if a.causality in ("chainstrict", "chaincf"):
            cp = REPO / a.chain_src / doc / "causalchain_states.jsonl"
            if not cp.exists():
                miss += 1
            else:
                rows = sorted((json.loads(l) for l in cp.read_text().splitlines() if l.strip()),
                              key=lambda r: int(r["item_id"]))
                ch = [r.get("causal_chain") or "none" for r in rows]
                if a.causality == "chaincf":      # none 을 직전 사슬로 보간(느린 축 규약)
                    last = "none"; cc = []
                    for x in ch:
                        if x != "none":
                            last = x
                        cc.append(last)
                    ch = cc
                if len(ch) < n_tr + 1:
                    miss += 1
                else:
                    R[t0i:t1i, COL["causality"]] = np.asarray(
                        [int(ch[t] != ch[t + 1] and "none" not in (ch[t], ch[t + 1]))
                         for t in range(n_tr)], np.int8)
        elif a.causality in ("aspcomp", "nonong"):
            want = "completive" if a.causality == "aspcomp" else None
            rows = read_states(doc, "causality", src)
            if rows is None or len(rows) < n_tr + 1:
                miss += 1
            else:
                R[t0i:t1i, COL["causality"]] = np.asarray(
                    [int(rows[t + 1].get("aspect") == want) if want
                     else int(rows[t + 1].get("aspect") != "ongoing") for t in range(n_tr)], np.int8)
        if a.intent == "i2":
            rows = read_states(doc, "intentionality", src)
            if rows is None or len(rows) < n_tr + 1:
                miss += 1
            else:
                # 🔴 carry-forward 미적용(원본 상태) · 채택만: none -> 목표
                g = [r.get("active_goal") or "none" for r in rows]
                R[t0i:t1i, COL["intentionality"]] = np.asarray(
                    [int(g[t] == "none" and g[t + 1] != "none") for t in range(n_tr)], np.int8)

    new = R[:, COL["causality"]].mean(), R[:, COL["intentionality"]].mean()
    print(f"조건 인과={a.causality} · 의도={a.intent} (라벨 누락 {miss})")
    print(f"  인과 발화율 {old[0]:.4f} -> {new[0]:.4f} · 의도 {old[1]:.4f} -> {new[1]:.4f}")
    print(f"  앵커 경계율(count≥3) {(ix['resets'].sum(1)>=3).mean():.4f} -> {(R.sum(1)>=3).mean():.4f}")

    np.savez(out_ix, doc_ids=np.array(doc_ids), sent_offsets=ix["sent_offsets"],
             trans_offsets=to, resets=R, states=ix["states"],
             **({"state_cols": ix["state_cols"]} if "state_cols" in ix.files else {}))
    emb = DATA / f"{split}_emb.npy"
    if not emb.exists():
        emb.symlink_to(DATA / f"{a.from_split}_emb.npy")
    print(f"-> {out_ix.name} · {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
