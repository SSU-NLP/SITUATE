#!/usr/bin/env python3
r"""M5-FactoredSlot + 적응임계 — **기본(default) 분절기 진입점**.

문장 리스트를 넣으면 경계를 돌려준다. LLM 없음 · 온라인 · frozen 인코더.

    from hi_ontop.m5_default import segment
    r = segment(["문장1", "문장2", ...])
    r["boundaries"]   # 새 세그먼트를 여는 문장 인덱스 [1-based 아님, 0-based 문장 번호]
    r["segments"]     # [(start, end_exclusive), ...]
    r["scores"]       # 전이별 연속 점수 s = Σ_k P(reset_k), 길이 n_sent-1
    r["axis_probs"]   # [n_sent-1, 5] 축별 리셋 확률 (protagonist/space/time/causality/intentionality)

CLI:

    python -m hi_ontop.m5_default input.txt            # 한 줄 = 한 문장
    python -m hi_ontop.m5_default input.txt --mode online_nf --c 1.5   # 규칙 변경(비권장)

## 동결된 구성 (`configs/m5_default.json` 과 동일해야 함)
| 항목 | 값 |
|---|---|
| 인코더 | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (frozen, L2 정규화) |
| 체크포인트 | `models/m5_default/M5.pt` = `M5_phase1__personkey_i2_per_slot_gsauthor_ep100_i2e100_seed0` (앵커 11,836편 학습, **의도 축 I2**. 논문 표의 4-seed 중 seed0) |
| 구조 | 5 타입 슬롯 · slot_dim 128 · dynamics `per_slot` · pred_space `obs` |
| 신호 | `s = Σ_k P(reset_k)` (5축 리셋확률 합) |
| 임계 | **적응임계 `online μ+1.0σ`** — `s_t ≥ 지금까지의 평균 + 1.0×표준편차`. **미래 미사용(온라인)** · gold 개수 미사용. c=1.0 은 **DTS 공식 test 36,109 전이**가 고른 값(D-B39 채택) |

## 🔴 알아야 할 한계 (숨기지 말 것)
- ✅ **3 seed**: dev-3 Score **0.4053 ± 0.0046** (직전 동결본 0.3963 · M4 0.3223). 무예측 하한 0.2790.
- 🔴 그 +0.0090 은 **pieman 한 편**에서 나온다(0.578→0.602). tunnel·monkey 는 미미/하락.
  dev-3 에서 고른 값이며 노이즈 하한(0.06) 안이다. 확인은 봉인 시험지 1회 개봉뿐.
- 🔴 **이 규칙은 dev-3 최고점이 아니다.** 고정분위수 r=0.0642 가 0.4191 로 더 높다. 그런데 r 의
  출처가 `count ≥ t*=3`(경계 규칙이 없던 시절의 임시 작동점)이고 **t\*=3 은 폐기됐다**(D-B33 부칙 1)
  → 정의가 사라진 숫자를 유지하는 것은 순환이다. 채택 기준은 *"평가셋 밖에 출처가 있고 온라인인
  규칙 중 최고"* 이며 c=1.0 은 **DTS 공식 test 36,109 전이**가 골랐다(dev-3 는 오히려 반대 증거).
- 🔴 **약 2.4배 과분절한다** — dev-3 예측 160개 vs gold 66개. oracle 개수를 주면 0.4519 가 나오므로
  **순위력은 있고 개수 교정이 안 된 것**이다. 경계 개수 제어는 **미해결**(D-B37/38).
- 🔴 D-B29 배치에서 **용량·학습HP·예측잔차특징·ψₑ 전부 무효**였다 — 병목은 그 중에 없다.
- 🔴 D-B34: **M5−M4 이득의 실질 전부가 입력경로 분리에서 온다**(상태·dynamics 분리는 단독 기여 없음).
  M5 vs M6(dynamics 공유)는 3 seed 에서 **구별되지 않는다**.
- **영어 서사 픽션으로만 학습**(D-M8). 대화·한국어·비서사는 **도메인 밖**이다.
  (인코더는 다국어라 동작은 하지만 축 라벨은 영어 서사에서 배운 것.)
- 앵커 라벨 자체 노이즈: 자기일치도 **S_self 0.659** = 성능 상한 요인.
- SEM 계승은 **표상 수준**이다. sticky-CRP·우도비·local MAP 등 추론 층은 구현·검정 후
  **기각**됐다(D-B28). 이 분절기는 **지도 분류기 + 고정분위수 임계**다.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from hi_ontop.factored_slot import AXES, FactoredSlot
from hi_ontop.hi_ontop_v2 import adaptive_boundaries

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/m5_default.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _load(device: str, checkpoint: str | None = None) -> tuple:
    """`checkpoint` 를 주면 config 의 동결 경로 대신 그 파일을 쓴다.

    쓰는 이유: 하류 재검증에서 **여러 세대의 체크포인트를 한 프로세스에서 비교**해야 하는데,
    config 의 단일 경로만 읽으면 `models/m5_default/M5.pt` 가 실행 중 교체될 때 조건이 조용히
    바뀐다(D-E20·D-L15 에서 두 번 발생). 경로를 명시하면 그 사고가 원천 차단된다.
    lru_cache 키에 checkpoint 가 들어가므로 세대별로 따로 캐시된다.
    """
    cfg = load_config()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(cfg["encoder"], device=device)
    ck_path = Path(checkpoint) if checkpoint else REPO / cfg["checkpoint"]
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    a = ck["args"]
    # 🔴 2026-08-02(D-B62): 예전에는 slot_dim·mono·dynamics·d1·shared_dim·pred_space 만 복원해
    # ψ·slot_reset·head_feat·병목 변형 체크포인트는 **로드 자체가 실패**했다. 저장된 args 를
    # 전부 복원한다. 동결본은 이 값들이 전부 기본값이라 **동작이 바뀌지 않는다**(회귀 검증 완료).
    dflt = lambda k, d: d if a.get(k) is None else a[k]                    # noqa: E731
    none = lambda v: None if v in (None, "none", 0) else v                 # noqa: E731
    if none(a.get("tagger")) is not None:
        raise ValueError("태거 변형 체크포인트는 축별 태거 피처가 필요하다 — 이 진입점 밖이다")
    m = FactoredSlot(slot_dim=a["slot_dim"], mono=ck["mono"], pos_weight=ck["pos_weight"],
                     alpha=a.get("_alpha", (1.0, 0.5, 1.0, 0.4, 0.25)),
                     dynamics=ck.get("dynamics"), pred_weight=dflt("pred_weight", 1.0),
                     d1=(True if a.get("input_path") is None
                         else a["input_path"] == "axis_extractor"),
                     shared_dim=none(a.get("shared_proj_dim")),
                     pred_space=dflt("pred_space", "obs"),
                     head_feat=dflt("head_feat", "delta"),
                     psi=none(a.get("psi")), psi_beta=dflt("psi_beta", 0.9),
                     psi_dropout=dflt("psi_dropout", 0.3),
                     slot_reset=none(a.get("slot_reset")),
                     slot_reset_dropout=dflt("slot_reset_dropout", 0.3),
                     bottleneck=dflt("bottleneck", 8),
                     dropout=dflt("dropout", 0.4)).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return enc, m, cfg


@torch.no_grad()
def segment(sentences: list[str], c: float | None = None, mode: str | None = None,
            device: str | None = None, checkpoint: str | None = None) -> dict:
    """문장 리스트 -> 경계. `c`/`mode` 를 주면 동결값을 덮어쓴다(기본은 config 값).

    `checkpoint` 를 주면 config 의 동결 체크포인트 대신 그 파일로 추론한다.
    """
    if len(sentences) < 2:
        return {"boundaries": [], "segments": [(0, len(sentences))],
                "scores": np.zeros(0), "axis_probs": np.zeros((0, 5))}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc, m, cfg = _load(device, checkpoint)
    c = cfg["threshold"].get("c", 1.0) if c is None else c
    mode = cfg["threshold"]["mode"] if mode is None else mode

    E = torch.tensor(np.asarray(enc.encode(sentences, normalize_embeddings=True, batch_size=64),
                                np.float32), device=device).unsqueeze(0)
    n = len(sentences)
    # 🔒 변형별 추론 경로는 학습기(`train_anchor_m4m5.axis_probs`)와 **같은 규약**이다:
    #    slot_reset·ψ 계열은 앵커 경계가 없는 추론에서 **자기 예측 경계로 상태를 리셋하며 전진**
    #    (rollout). 앵커 경계를 넣으면 누출이다. 동결본은 둘 다 None 이라 기존 경로 그대로.
    if getattr(m, "slot_reset", None) is not None:
        P = m.change_probs_slot_rollout(E)[0].cpu().numpy()
    elif getattr(m, "psi", None) is not None:
        P = m.change_probs_rollout(E)[0].cpu().numpy()
    else:
        o = m.forward_batched(E, torch.zeros(1, n - 1, 5, dtype=torch.long, device=device),
                              torch.zeros(1, n, 3, dtype=torch.long, device=device),
                              torch.ones(1, n, device=device),
                              torch.ones(1, n - 1, device=device))
        P = torch.stack([torch.sigmoid(o["change_logits"][a]) for a in AXES], -1)[0].cpu().numpy()
    s = P.sum(1)                                              # [n-1]
    # adaptive_boundaries 규약: 0번 sentinel 을 앞에 붙인 시퀀스를 받는다
    if mode == "fixed_quantile":
        # 🔒 D-B29: 앵커 val 5-fold OOF 가 고른 규칙(200 fold 중 192 승). 문서 내 상위 r 분위수.
        b = (s >= np.quantile(s, 1.0 - cfg["threshold"]["rate"])).astype(np.int8)
    else:
        b = np.asarray(adaptive_boundaries(np.concatenate([[0.0], s]).tolist(), c=c, mode=mode),
                       np.int8)[:n - 1]
    bnd = [int(i) + 1 for i in np.flatnonzero(b)]             # 새 세그먼트를 여는 문장 번호
    cuts = [0] + bnd + [n]
    return {"boundaries": bnd, "segments": list(zip(cuts[:-1], cuts[1:])),
            "scores": s, "axis_probs": P}


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 + 앵커 동결 임계 분절 (기본 진입점)")
    ap.add_argument("input", help="한 줄 = 한 문장인 텍스트 파일")
    ap.add_argument("--c", type=float, default=None)
    ap.add_argument("--mode", default=None, choices=[None, "fixed_quantile", "online_nf", "offline", "otsu"])
    ap.add_argument("--json", action="store_true", help="JSON 으로 출력")
    args = ap.parse_args()
    sents = [x.strip() for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
    r = segment(sents, c=args.c, mode=args.mode)
    if args.json:
        print(json.dumps({"boundaries": r["boundaries"],
                          "segments": [list(x) for x in r["segments"]],
                          "scores": r["scores"].round(4).tolist()}, ensure_ascii=False, indent=2))
        return
    cfg = load_config()
    md = args.mode or cfg["threshold"]["mode"]
    hp = (f"r={cfg['threshold']['rate']}" if md == "fixed_quantile"
          else f"c={args.c if args.c is not None else cfg['threshold'].get('c', 1.0)}")
    print(f"[m5_default] {len(sents)}문장 · 임계 {md} {hp} · 세그먼트 {len(r['segments'])}개\n")
    for si, (a, b) in enumerate(r["segments"]):
        print(f"── 세그먼트 {si} [{a}:{b}]")
        for i in range(a, b):
            sc = f"{r['scores'][i-1]:.2f}" if i > 0 else "----"
            print(f"   [{i:3d}] s={sc}  {sents[i]}")


if __name__ == "__main__":
    main()
