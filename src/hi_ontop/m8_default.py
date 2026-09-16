#!/usr/bin/env python3
r"""M8-DirectBoundary — 축 분해 없는 이진 경계 헤드의 **추론 진입점**.

`m5_default.segment()` 와 **같은 계약**을 따른다. 문장 리스트를 넣으면 경계를 돌려준다.

    from hi_ontop.m8_default import segment
    r = segment(["문장1", "문장2", ...], checkpoint=".../M8_pair_seed0.pt")
    r["boundaries"]   # 새 세그먼트를 여는 문장 인덱스
    r["segments"]     # [(start, end_exclusive), ...]
    r["scores"]       # 전이별 s = P(boundary), 길이 n_sent-1

왜 별도 모듈인가: M8 은 FactoredSlot 이 아니다. 상태도 축도 dynamics 도 없는 MLP 이고
점수도 `s = Σ_k P(reset_k)` 가 아니라 **단일 시그모이드 `s = P(boundary)`** 다.
`m5_default._load()` 의 체크포인트 스키마(`mono`/`dynamics`/`args`)와 호환되지 않는다.

## 구성 (학습기 `scripts/eacl_phase0/train_direct_boundary.py` 와 동형이어야 한다)
| 항목 | 값 |
|---|---|
| 인코더 | `configs/m5_default.json` 의 encoder (frozen, L2 정규화) — M4/M5 와 **동일** |
| 헤드 | `Linear(D, hidden) → GELU → Dropout → Linear(hidden, 1)` (`train_direct_boundary.py:49-58`) |
| 전이 특징 | pair: `[z_l, z_r, z_r−z_l, z_l⊙z_r]` (3072-D) · ctx: `+ [z_r−c_l]` (3840-D) |
| `c_l` | 문서 내 **과거 포함 누적평균**의 L2 정규화 — 과거만 쓰므로 온라인 적법 |
| 신호 | `s = P(boundary)` |
| 임계 | M4/M5 와 **동일한 적응임계** `online_nf` c=1.0 (D-B39) |

🔴 헤드·특징 정의는 학습기에서 **손으로 옮긴 것**이다(학습기를 import 하면
`train_anchor_m4m5` 의 앵커 데이터 경로까지 끌려와 추론 환경에서 깨진다).
`selftest()` 가 학습기 정의와 수치가 같은지 검증한다 — 학습기를 고치면 여기도 고칠 것.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from hi_ontop.hi_ontop_v2 import adaptive_boundaries

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/m5_default.json"


class DirectBoundaryHead(nn.Module):
    """전이 특징 -> 경계 로짓 1개. 상태 없음 · 축 없음 · dynamics 없음.

    `train_direct_boundary.DirectBoundaryHead` 와 **같은 모듈 이름·같은 순서**여야
    `state_dict` 키(`net.0.weight` …)가 맞는다.
    """

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def feats(zl: torch.Tensor, zr: torch.Tensor, cl: torch.Tensor | None) -> torch.Tensor:
    """전이 특징. `cl` 이 있으면 과거 누적문맥 항을 덧붙인다(과거만 사용 = 온라인 적법)."""
    parts = [zl, zr, zr - zl, zl * zr]
    if cl is not None:
        parts.append(zr - cl)
    return torch.cat(parts, -1)


def causal_context(emb: torch.Tensor) -> torch.Tensor:
    """단일 문서의 **과거 포함 누적평균** (L2 정규화). c_t = normalize(mean(z_0..z_t))."""
    cs = torch.cumsum(emb.float(), 0) / torch.arange(
        1, emb.shape[0] + 1, device=emb.device, dtype=torch.float32).unsqueeze(1)
    return torch.nn.functional.normalize(cs, dim=1).to(emb.dtype)


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _load(device: str, checkpoint: str) -> tuple:
    cfg = load_config()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(cfg["encoder"], device=device)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    variant = ck["variant"]
    assert variant in ("pair", "ctx"), f"알 수 없는 M8 변형: {variant}"
    in_dim = 768 * (5 if variant == "ctx" else 4)
    m = DirectBoundaryHead(in_dim, ck.get("hidden", 256), ck.get("dropout", 0.4)).to(device)
    m.load_state_dict(ck["state"])       # 키가 안 맞으면 여기서 죽는다 — 조용한 폴백 없음
    m.eval()
    return enc, m, variant, cfg


@torch.no_grad()
def segment(sentences: list[str], c: float = 1.0, mode: str = "online_nf",
            device: str | None = None, checkpoint: str | None = None) -> dict:
    """문장 리스트 -> 경계. `checkpoint` 필수(M8 은 동결 기본본이 없다)."""
    assert checkpoint, "m8_default.segment 는 checkpoint 를 반드시 받는다"
    n = len(sentences)
    if n < 2:
        return {"boundaries": [], "segments": [(0, n)], "scores": np.zeros(0)}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc, m, variant, _ = _load(device, checkpoint)

    E = torch.tensor(np.asarray(enc.encode(sentences, normalize_embeddings=True, batch_size=64),
                                np.float32), device=device)
    ctx = causal_context(E) if variant == "ctx" else None
    j = torch.arange(0, n - 1, device=device)
    x = feats(E[j], E[j + 1], ctx[j] if ctx is not None else None)
    s = torch.sigmoid(m(x)).cpu().numpy().astype(np.float64)   # [n-1]

    # adaptive_boundaries 규약: 0번 sentinel 을 앞에 붙인 시퀀스를 받는다 (m5_default 와 동일)
    b = np.asarray(adaptive_boundaries(np.concatenate([[0.0], s]).tolist(), c=c, mode=mode),
                   np.int8)[:n - 1]
    bnd = [int(i) + 1 for i in np.flatnonzero(b)]
    cuts = [0] + bnd + [n]
    return {"boundaries": bnd, "segments": list(zip(cuts[:-1], cuts[1:])), "scores": s}


def selftest(checkpoint: str, device: str = "cpu") -> None:
    """이 모듈의 헤드·특징이 **학습기 정의와 수치가 같은지** 검증한다.

    학습기를 import 할 수 있는 환경(레포)에서만 돈다. 스테이징 환경에서는 건너뛴다.
    """
    import sys
    sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
    sys.path.insert(0, str(REPO / "src"))
    import train_direct_boundary as B

    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    variant = ck["variant"]
    in_dim = 768 * (5 if variant == "ctx" else 4)
    ref = B.DirectBoundaryHead(in_dim, ck.get("hidden", 256), ck.get("dropout", 0.4)).to(device)
    ref.load_state_dict(ck["state"]); ref.eval()
    mine = DirectBoundaryHead(in_dim, ck.get("hidden", 256), ck.get("dropout", 0.4)).to(device)
    mine.load_state_dict(ck["state"]); mine.eval()

    g = torch.Generator(device="cpu").manual_seed(0)
    E = torch.nn.functional.normalize(torch.randn(7, 768, generator=g), dim=1).to(device)
    for var, cl in (("pair", None), ("ctx", causal_context(E)[:6])):
        if var != variant:
            continue
        x_mine = feats(E[:6], E[1:7], cl)
        x_ref = B.feats(E[:6], E[1:7], cl)
        assert torch.allclose(x_mine, x_ref, atol=0), "feats 불일치"
        with torch.no_grad():
            assert torch.allclose(mine(x_mine), ref(x_ref), atol=0), "헤드 출력 불일치"
    # causal_context 도 학습기(문서 배열판)와 대조한다
    if variant == "ctx":
        so = np.asarray([0, 7])
        assert torch.allclose(causal_context(E), B.causal_context(E, so, device), atol=1e-6), \
            "causal_context 불일치"
    print(f"[m8 selftest] {Path(checkpoint).name} variant={variant} — 학습기와 수치 동일 ✓")


def main() -> None:
    ap = argparse.ArgumentParser(description="M8-DirectBoundary + 적응임계 분절")
    ap.add_argument("input", nargs="?", help="한 줄 = 한 문장인 텍스트 파일")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--mode", default="online_nf")
    ap.add_argument("--selftest", action="store_true", help="학습기 정의와 대조만 하고 끝낸다")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.checkpoint)
        return
    sents = [x.strip() for x in Path(a.input).read_text(encoding="utf-8").splitlines() if x.strip()]
    r = segment(sents, c=a.c, mode=a.mode, checkpoint=a.checkpoint)
    print(json.dumps({"boundaries": r["boundaries"],
                      "scores": np.round(r["scores"], 4).tolist()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
