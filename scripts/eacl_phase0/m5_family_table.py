#!/usr/bin/env python3
"""M5 계열 시험지 순위표 — **파일명(코드명) 대신 실제 설정**으로 렌더링.

체크포인트 파일명은 내부 네임스페이스(`_phase1_i2_per_slot_slot_pwone`)라 읽어도 뜻이 안 보인다.
저장된 `args` 를 사람이 읽는 설정 이름으로 풀어 쓴다. 기본값(= 동결본과 같은 값)은 적지 않고
**동결본과 다른 항목만** 적는다.

    .venv/bin/python scripts/eacl_phase0/m5_family_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "outputs/experiments/m4m5_phase1_anchor"
OUT = REPO / "outputs/experiments/2026-08-02_m5_family_test3"

# ── 앵커 split -> (라벨러, 의도 축 규칙, 인과 축 규칙, 코퍼스) ─────────────────
#   라벨 자체는 전부 S5 프롬프트 `p0s5-eism-v3.1` 로 뽑았다. 라벨러 모델만 다르다.
#   의도/인과 축의 "규칙"은 같은 라벨에서 리셋 열만 다시 계산한 것(API 0, recompute_axes.py).
SPLIT = {
    "phase1":      ("Qwen3.6-35B", "동결(carry∞)", "동결", "본 10,598편"),
    "phase1_i2":   ("Qwen3.6-35B", "**채택만(i2)**", "동결", "본 10,598편"),
    "phase1_smoke": ("Qwen3.6-35B", "동결(carry∞)", "동결", "🔴 스모크 40편"),
    "phase1_gemma": ("Gemma-4-26B", "동결(carry∞)", "동결", "본 10,542편"),
    "phase1_gptoss": ("gpt-oss-120b", "동결(carry∞)", "동결", "본 10,593편"),
    "phase1_gptoss_i2": ("gpt-oss-120b", "**채택만(i2)**", "동결", "본 10,593편"),
    "phase1_gptoss_chainstrict_i2": ("gpt-oss-120b", "**채택만(i2)**", "사슬(엄격)", "본 10,593편"),
    "phase1_aspcomp": ("Qwen3.6-35B", "동결(carry∞)", "상(相)=완료", "본 10,598편"),
    "phase1_aspcomp_i2": ("Qwen3.6-35B", "**채택만(i2)**", "상(相)=완료", "본 10,598편"),
    "phase1_nonong": ("Qwen3.6-35B", "동결(carry∞)", "상(相)≠진행중", "본 10,598편"),
    "phase1_chainq_i2": ("Qwen3.6-35B", "**채택만(i2)**", "사슬(Qwen 라벨)", "본 10,598편"),
    "phase1_chainoss_i2": ("Qwen3.6-35B", "**채택만(i2)**", "사슬(gpt-oss 라벨)", "본 10,598편"),
}
# ── 구조/학습 설정: (args 키, 동결본 값, 다를 때 쓸 문구) ─────────────────────
FIELD = [
    ("epochs", 8, lambda v: f"**{v} epoch**"),
    ("slot_dim", 128, lambda v: f"슬롯 {v}차원"),
    ("pred_space", "slot", lambda v: "비교공간=관측(슬롯 아님)"),
    ("head_feat", "delta", lambda v: {"resid_signed": "입력=부호 있는 잔차",
                                      "resid_sq": "입력=제곱 잔차",
                                      "pred_centric": "입력=예측중심 비교"}.get(v, f"입력={v}")),
    ("psi", "none", lambda v: {"ema": "ψ 지수평활 상태", "cummean": "ψ 누적평균 상태"}.get(v, v)),
    ("slot_reset", None, lambda v: "슬롯 강제 리셋"),
    ("tagger", "none", lambda v: {"hybrid": "spaCy 태거 **병합**",
                                  "replace": "spaCy 태거 **대체**"}.get(v, f"태거={v}")),
    ("tagger_axes", 5, lambda v: f"태거 {v}축"),
    ("input_path", "axis_extractor", lambda v: "입력경로=공유투영"),
    ("shared_proj_dim", 0, lambda v: f"공유투영 {v}차원"),
    ("pos_weight_mode", "one", lambda v: "손실 양성가중 clip(구식)"),
    ("alpha_mode", "current", lambda v: "축 손실 균등"),
    ("bottleneck", 8, lambda v: f"병목 {v}"),
    ("lr", 0.0003, lambda v: f"lr {v}"),
]
NEUTRAL = {"input_path": (None,), "tagger": (None,), "shared_proj_dim": (None,),
           "alpha_mode": (None,), "tagger_axes": (None,), "pos_weight_mode": ()}


def describe(a: dict) -> str:
    """동결본과 다른 설정만 사람이 읽는 문구로."""
    out = []
    for k, base, fmt in FIELD:
        v = a.get(k)
        if v == base or v in NEUTRAL.get(k, ()):        # 옛 ckpt 의 None = 그 시절 기본값
            continue
        if k == "pos_weight_mode" and v is None:        # 구세대는 clip 가중이 기본이었다
            out.append(fmt(v))
            continue
        if v is None:
            continue
        out.append(fmt(v))
    return " · ".join(out) if out else "기본 그대로"


def main() -> None:
    rows = json.loads((OUT / "results.json").read_text())["rows"]
    dev = {r["ckpt"]: r["adaptive"]["score"]
           for r in json.loads((OUT / "results_dev3.json").read_text())["rows"]}
    dev_rank = {k: i for i, k in enumerate(sorted(dev, key=lambda k: -dev[k]), 1)}
    rows.sort(key=lambda r: -r["adaptive"]["score"])

    L = ["| 순위 | Score | BOR | 앵커 라벨러 | 의도 축 규칙 | 인과 축 규칙 | 코퍼스 | 구조·학습 설정 "
         "(동결본과 다른 것만) | seed | 연습장 순위 |",
         "|---:|---:|---:|---|---|---|---|---|---:|---:|"]
    for i, r in enumerate(rows, 1):
        name = r["ckpt"]
        p = (REPO / "models/m5_default/M5.pt") if name.startswith("m5_default") else \
            EXP / f"{name}.pt"
        a = torch.load(p, map_location="cpu", weights_only=False)["args"]
        lab, intent, caus, corp = SPLIT.get(a.get("split", "phase1"),
                                            ("?", "?", "?", a.get("split", "?")))
        desc = describe(a)
        if name.startswith("m5_default"):
            desc = "**← 우리가 못박아 둔 동결본**"
        deg = " 🔴 경계 0개" if r["adaptive"]["n_pred"] == 0 else ""
        L.append(f"| {i}{deg} | {r['adaptive']['score']:.4f} | {r['adaptive']['bor']:.2f} | {lab} | "
                 f"{intent} | {caus} | {corp} | {desc} | {a.get('seed', '?')} | "
                 f"{dev_rank.get(name, '—')} |")
    (OUT / "table_readable.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
