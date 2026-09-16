#!/usr/bin/env python3
r"""gpt-oss 앵커로 학습 split 구성 — **재인코딩 0** (D-B58).

## 왜 이 스크립트가 필요한가
`build_anchor_dataset.py` 는 문장을 **다시 인코딩**한다(2.78M 문장, GPU 필요). 그런데
gpt-oss 코퍼스는 **문서·문장이 Qwen 코퍼스와 동일**하고 바뀐 것은 라벨뿐이다.
⇒ `phase1` 의 doc 순서·offset 을 그대로 쓰고 **`phase1_emb.npy` 를 심볼릭으로 재사용**한다.
(`recompute_axes.py` 가 축 재조작화에 쓰는 것과 같은 수법.)

🔴 **resets 와 states 를 둘 다 gpt-oss 라벨에서 읽는다.** states(보조 enum 타깃)를 Qwen 것으로
두면 주 손실은 gpt-oss·보조 손실은 Qwen 이 되어 조건이 섞인다.

## 전제
`phase1` 의 문서 집합이 gpt-oss 완비 집합의 **부분집합**이어야 offset 이 그대로 맞는다.
아니면 중단한다(임의로 잘라내면 offset 이 어긋나 임베딩 재사용이 깨진다).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from build_anchor_dataset import AXES, load_anchors  # noqa: E402

DATA = REPO / "outputs/experiments/m4m5_phase1_anchor/data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-dir", required=True, help="gpt-oss 앵커 디렉터리")
    ap.add_argument("--from-split", default="phase1", help="doc 순서·offset·임베딩을 빌려올 split")
    ap.add_argument("--tag", required=True, help="산출 split 이름 접미사 (예: gptoss)")
    a = ap.parse_args()
    split = f"{a.from_split}_{a.tag}"
    out_ix = DATA / f"{split}_index.npz"
    if out_ix.exists():
        raise SystemExit(f"🔴 이미 있다: {out_ix.name}")

    ix = np.load(DATA / f"{a.from_split}_index.npz", allow_pickle=True)
    doc_ids = [str(x) for x in ix["doc_ids"]]
    so, to = ix["sent_offsets"], ix["trans_offsets"]
    n_sent = [int(so[i + 1] - so[i]) for i in range(len(doc_ids))]
    print(f"[base] {a.from_split}: {len(doc_ids):,}편 · {int(so[-1]):,}문장 · {int(to[-1]):,}전이",
          flush=True)

    resets_parts, states_parts, dropped = load_anchors(doc_ids, n_sent, Path(a.anchor_dir))
    R = np.concatenate(resets_parts)
    S = np.concatenate(states_parts)

    if not dropped:
        # 문서 집합이 완전히 같다 → offset·임베딩을 그대로 재사용(복사 0)
        keep = list(range(len(doc_ids)))
        assert len(R) == int(to[-1]) and len(S) == int(so[-1]), "offset 불일치"
        new_so, new_to = so, to
        emb = DATA / f"{split}_emb.npy"
        if not emb.exists():
            emb.symlink_to(DATA / f"{a.from_split}_emb.npy")
        print(f"[emb] 심볼릭 재사용 (복사 0)", flush=True)
    else:
        # 🔴 라벨 불완전 문서를 **빼고** 새 offset 을 만든다. 임베딩은 그 문서 구간만
        #    빼고 슬라이스 복사한다(재인코딩 0, GPU 0). 무엇을 뺐는지 반드시 보고한다.
        drop = set(dropped)
        keep = [i for i, d in enumerate(doc_ids) if d not in drop]
        print(f"[drop] gpt-oss 라벨 불완전 {len(dropped)}편 제외: {dropped}", flush=True)
        ns = [n_sent[i] for i in keep]
        new_so = np.asarray([0] + list(np.cumsum(ns)), dtype=np.int64)
        new_to = np.asarray([0] + list(np.cumsum([n - 1 for n in ns])), dtype=np.int64)
        src = np.load(DATA / f"{a.from_split}_emb.npy", mmap_mode="r")
        dst = np.lib.format.open_memmap(DATA / f"{split}_emb.npy", mode="w+",
                                        dtype=src.dtype, shape=(int(new_so[-1]), src.shape[1]))
        for j, i in enumerate(keep):
            dst[int(new_so[j]):int(new_so[j + 1])] = src[int(so[i]):int(so[i + 1])]
        dst.flush(); del dst
        print(f"[emb] 슬라이스 복사 완료 {int(new_so[-1]):,}문장", flush=True)
        doc_ids = [doc_ids[i] for i in keep]

    np.savez(out_ix, doc_ids=np.array(doc_ids), sent_offsets=new_so, trans_offsets=new_to,
             resets=R, states=S,
             **({"state_cols": ix["state_cols"]} if "state_cols" in ix.files else {}))

    print(f"[done] {split}: {len(doc_ids):,}편 · 전이 {len(R):,}")
    for i, ax in enumerate(AXES):
        print(f"  {ax:22s} 발화율 {R[:, i].mean():.4f} (기준 {ix['resets'][:, i].mean():.4f})")
    print(f"  경계율(count>=3) {(R.sum(1) >= 3).mean():.4f} "
          f"(기준 {(ix['resets'].sum(1) >= 3).mean():.4f})")
    print(f"-> {out_ix}")


if __name__ == "__main__":
    main()
