#!/usr/bin/env python3
"""EM-LLM 문장별 surprise 를 **test3 3편**에 대해 계산해 캐시한다.

dev3 는 `2026-07-31_m9_attn_entity/cache/{doc}__emllm.npz` 에 이미 있으나 test3 는 없다.
같은 LM·같은 함수(`scripts/segment_manual_emllm.py::encode_doc`)를 **재사용**해 조건을 맞춘다
— 다른 코드로 뽑으면 dev3/test3 가 비교 불가가 된다.

LM = `Qwen/Qwen2.5-7B` (`segment_manual_emllm.py:41` DEFAULT_LM, EM-LLM 논문 7B급 정합).

    .venv/bin/python EACL/methods/emllm/compute_surprise_test3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))

CACHE = REPO / "outputs/experiments/2026-07-31_m9_attn_entity/cache"
DOCS = ("run", "gohitler", "mymotherslife")


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from segment_manual_emllm import DEFAULT_LM, encode_doc   # noqa: PLC0415
    from baseline_compare import load_dataset                 # noqa: PLC0415

    data = {d: s for d, s, _g in load_dataset("test3")}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LM={DEFAULT_LM} · device={dev}")
    tok = AutoTokenizer.from_pretrained(DEFAULT_LM)
    # 🔴 `device_map="auto"` 를 쓰면 레이어가 여러 GPU 로 쪼개져 `encode_doc` 의
    # `torch.stack(repr_per_layer)` 이 cuda:0/cuda:1 혼합으로 터진다. **단일 GPU** 에 올린다
    # (Qwen2.5-7B fp16 ≈ 15GB, 80GB 카드에 충분).
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_LM, torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32)
    model.to(dev)
    model.eval()

    CACHE.mkdir(parents=True, exist_ok=True)
    for doc in DOCS:
        out = CACHE / f"{doc}__emllm.npz"
        if out.exists():
            print(f"  {doc}: 이미 있음 — 건너뜀")
            continue
        sents = data[doc]
        with torch.no_grad():
            surprise, A = encode_doc(sents, tok, model, dev)
        surprise = np.asarray(surprise, dtype=float)
        np.savez_compressed(out, surprise=surprise, A_sim=np.asarray(A, dtype=np.float32))
        print(f"  {doc}: 문장 {len(sents)} · surprise {surprise.shape} → {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
