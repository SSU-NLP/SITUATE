#!/usr/bin/env python3
"""EM-LLM 문장별 surprise 를 **임의의 LM 으로** 6편(dev3+test3) 전체에 계산해 캐시한다.

`compute_surprise_test3.py` 의 일반화판. 그쪽은 LM 이 `DEFAULT_LM`(Qwen2.5-7B)에 고정되고
대상이 test3 3편으로 고정돼 있어 D-07 의 "Llama 로 다시 뽑아 비교" 를 할 수 없다.

**D-07 배경**: surprise 는 Qwen2.5-7B 로 계산했는데 EM-LLM 원 논문·우리 LongBench 재현은
Llama 계열이다. *"LongBench 에선 재현되는데 서사에선 무경계 하한 아래"* 라는 대비가
**모델 차이 때문일 가능성**을 배제하려면 같은 파이프라인을 Llama 로 한 번 더 돌려야 한다.

분절 함수는 `scripts/segment_manual_emllm.py::encode_doc` 을 **그대로 재사용**한다 —
다른 코드로 뽑으면 조건이 달라져 비교 불가가 된다.

캐시는 **조건 네임스페이스**로 분리한다(`{doc}__emllm_{tag}.npz`). 기존 Qwen 캐시
`{doc}__emllm.npz` 는 건드리지 않는다.

    # Llama-3.1-8B-Instruct (로컬 존재)
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python EACL/methods/emllm/compute_surprise.py \
        --lm meta-llama/Llama-3.1-8B-Instruct --tag llama31i

🔴 `device_map="auto"` 금지: 레이어가 여러 GPU 로 쪼개지면 `encode_doc` 의
`torch.stack(repr_per_layer)` 이 cuda:0/cuda:1 혼합으로 터진다(D-05). 단일 GPU 에 올린다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))

CACHE = REPO / "outputs/experiments/2026-07-31_m9_attn_entity/cache"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", required=True, help="HF 모델 ID (예: meta-llama/Llama-3.1-8B-Instruct)")
    ap.add_argument("--tag", required=True,
                    help="캐시 네임스페이스. 산출 = {doc}__emllm_{tag}.npz")
    ap.add_argument("--datasets", default="dev3,test3")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--nll-chunk", type=int, default=0,
                    help="토큰 NLL 을 이 행 크기로 나눠 계산(수치 동일, 메모리 절감). "
                         "0=기존 동작. GPU 를 다른 작업과 공유할 때 512~1024 권장")
    ap.add_argument("--verify-chunk-equals", default="",
                    help="지정한 doc 에 대해 chunk 유무 두 경로를 모두 계산해 "
                         "수치가 같은지 확인만 하고 종료한다")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from segment_manual_emllm import encode_doc   # noqa: PLC0415
    from baseline_compare import load_dataset     # noqa: PLC0415

    if torch.cuda.device_count() > 1:
        print(f"🔴 GPU 가 {torch.cuda.device_count()}장 보인다. encode_doc 은 단일 GPU 전제다 "
              f"(D-05). CUDA_VISIBLE_DEVICES 로 1장만 노출해 실행할 것.", file=sys.stderr)
        return 2

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LM={args.lm} · tag={args.tag} · device={dev}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.lm)
    model = AutoModelForCausalLM.from_pretrained(
        args.lm, torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32)
    model.to(dev)
    model.eval()

    CACHE.mkdir(parents=True, exist_ok=True)

    if args.verify_chunk_equals:
        sents = {d: s for ds in args.datasets.split(",")
                 for d, s, _g in load_dataset(ds)}[args.verify_chunk_equals]
        with torch.no_grad():
            s_plain, _ = encode_doc(sents, tok, model, dev, nll_chunk=0)
            s_chunk, _ = encode_doc(sents, tok, model, dev, nll_chunk=args.nll_chunk or 512)
        d = np.abs(np.asarray(s_plain) - np.asarray(s_chunk)).max()
        print(f"  {args.verify_chunk_equals}: max|Δsurprise| = {d:.3e} "
              f"({'✅ 동일' if d == 0.0 else '⚠️ 불일치' if d > 1e-6 else '✅ 부동소수 오차 내'})")
        return 0 if d < 1e-6 else 1

    for ds in args.datasets.split(","):
        for doc, sents, _gold in load_dataset(ds):
            out = CACHE / f"{doc}__emllm_{args.tag}.npz"
            if out.exists() and not args.overwrite:
                print(f"  {doc}: 이미 있음 — 건너뜀", flush=True)
                continue
            t0 = time.time()
            with torch.no_grad():
                surprise, A = encode_doc(sents, tok, model, dev, nll_chunk=args.nll_chunk)
            surprise = np.asarray(surprise, dtype=float)
            np.savez_compressed(out, surprise=surprise, A_sim=np.asarray(A, dtype=np.float32))
            print(f"  {doc}: 문장 {len(sents)} · surprise {surprise.shape} "
                  f"· {time.time()-t0:.1f}s → {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
