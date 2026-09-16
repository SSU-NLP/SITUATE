#!/usr/bin/env python3
"""본편 §2.2 진단 — 채택본이 superseg·ami 에서 무경계 하한 아래인 **원인**.

## 왜 본편 항목인가
적응 임계는 논문 §2.2 다. 대화 4세트 성적이 **1 통과 / 1 해석 불가 / 2 미달**이고
미달 원인이 미확인이면 그 절 자체가 흔들린다. 프록시 회귀는 "과예측 배율"을 기각했다(R² ≤ 0.291).

## 가설 셋 (기존 출력만으로 갈린다 — 새 데이터·새 학습 없음)
- **H1 입도 불일치.** M5 의 작동 밀도가 그 세트의 gold 밀도와 멀면 점수가 떨어진다.
- **H2 웜업.** `online_nf` 는 cnt<2 에서 발화하지 않는다. 문서가 짧으면 결정 지점이 몇 개 없고
  초반 누적 통계가 불안정하다. (음성 트랙 종료 사유와 같은 계열 — 2026-07-31 GATE)
- **H3 채널.** 위 둘을 통제한 뒤에도 남는 대화 대 서사 격차. D-M7 계열.

## 판정
H1·H2 를 문서 수준 회귀로 넣고, 잔차에 서사/대화 지시변수를 넣어 H3 를 본다.
H3 이면 문어 대화 다운스트림(LongConvQA 등)도 같은 이유로 죽는다 ⇒ 접는다.
H1·H2 면 고칠 수 있는 문제다.

🔴 절대값을 성능으로 인용하지 않는다. 이 스크립트는 **원인 분해**용이다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.factored_slot import FactoredSlot  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402  🔒 동결
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402

CONFIG = REPO / "configs/m5_default.json"
SETS = ("dev3", "dts_tiage", "dts_dialseg711", "dts_superseg", "dts_ami")
NARRATIVE = {"dev3"}
OUT = REPO / "outputs/experiments/2026-07-31_voice_duplex"
WARMUP_N = 2          # online_nf 가 발화를 시작하는 최소 표본 수 (동결 구현)


def frozen_adaptive(s: np.ndarray, c: float, mode: str) -> np.ndarray:
    n = len(s)
    return np.asarray(adaptive_boundaries(np.concatenate([[0.0], s]).tolist(), c=c, mode=mode),
                      np.int8)[:n]


def one_doc(gold: np.ndarray, pred: np.ndarray) -> float:
    y = np.concatenate([gold, [0]]).tolist()
    p = np.concatenate([pred, [0]]).tolist()
    return score_dialogues([y], [p])["score"]


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    c_thr, mode = cfg["threshold"]["c"], cfg["threshold"]["mode"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(REPO / cfg["checkpoint"], map_location=dev, weights_only=False)
    m = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"],
                     pos_weight=ck["pos_weight"], dynamics=ck.get("dynamics"),
                     d1=ck["args"].get("input_path", "axis_extractor") == "axis_extractor",
                     shared_dim=ck["args"].get("shared_proj_dim") or None,
                     pred_space=ck["args"].get("pred_space", "obs")).to(dev)
    m.load_state_dict(ck["state"]); m.eval()

    docs = []
    for name in SETS:
        corp = Corpus(name)
        idx = list(range(corp.n_docs))
        probs = axis_probs(m, corp, idx, dev, bs=8, with_anchor=False)
        for i, pr in zip(idx, probs):
            g = corp.gold[int(corp.to[i]):int(corp.to[i + 1])].astype(np.int8)
            s = pr.sum(1)
            b = frozen_adaptive(s, c_thr, mode)
            n = len(g)
            docs.append({
                "set": name, "doc": corp.doc_ids[i], "narrative": name in NARRATIVE,
                "n_trans": n,
                "gold_density": float(g.mean()),
                "pred_density": float(b.mean()),
                # 웜업 구간이 문서에서 차지하는 비중 — 짧은 문서일수록 크다
                "warmup_frac": min(1.0, WARMUP_N / max(n, 1)),
                "score": one_doc(g, b),
                "floor": one_doc(g, np.zeros(n, np.int8)),
                # 신호 자체의 순위력 — 임계와 무관 (H1/H2 통제 후 H3 판별용)
                "signal_auc": _auc(g, s),
            })
            docs[-1]["delta"] = docs[-1]["score"] - docs[-1]["floor"]
            docs[-1]["density_gap"] = abs(docs[-1]["pred_density"] - docs[-1]["gold_density"])

    _report(docs)


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """경계 vs 비경계 순위력. 한쪽 클래스만 있으면 NaN."""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """절편 포함 최소제곱. (계수, R²) 반환."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    r2 = 1.0 - (resid ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return beta, float(r2)


def _report(docs: list[dict]) -> None:
    d = {k: np.array([x[k] for x in docs], float) for k in
         ("n_trans", "gold_density", "pred_density", "warmup_frac", "score", "floor",
          "delta", "density_gap", "signal_auc")}
    narr = np.array([x["narrative"] for x in docs], float)
    n = len(docs)

    L = ["# 진단 — 대화 세트 무경계 하한 미달의 원인 (본편 §2.2)\n",
         f"문서 {n}편 (dev3 3 · tiage 20 · dialseg711 12 · superseg 15 · ami 2). "
         "채택본 + 동결 적응 임계. 새 데이터·새 학습 없음.\n",
         "## 세트별 구조\n",
         "| 세트 | 문서 | 전이/문서 | gold 밀도 | 예측 밀도 | 웜업 비중 | 하한 대비 Δ | 신호 AUC |",
         "|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        m_ = np.array([x["set"] == s for x in docs])
        L.append(f"| {s} | {m_.sum()} | {d['n_trans'][m_].mean():.0f} | "
                 f"{d['gold_density'][m_].mean():.3f} | {d['pred_density'][m_].mean():.3f} | "
                 f"{d['warmup_frac'][m_].mean():.3f} | {d['delta'][m_].mean():+.4f} | "
                 f"{np.nanmean(d['signal_auc'][m_]):.3f} |")

    L += ["\n## 가설 검정 — 문서 수준 회귀 (종속변수 = 하한 대비 Δ)\n",
          "| 모형 | 설명변수 | R² |", "|---|---|---|"]
    ok = ~np.isnan(d["signal_auc"])          # 한쪽 클래스뿐인 문서 제외
    models = [
        ("H1 입도만", ["density_gap"]),
        ("H2 웜업만", ["warmup_frac"]),
        ("H3 신호품질만", ["signal_auc"]),
        ("H1+H2", ["density_gap", "warmup_frac"]),
        ("H1+H2+H3", ["density_gap", "warmup_frac", "signal_auc"]),
    ]
    r2s = {}
    for label, cols in models:
        X = np.column_stack([d[c][ok] for c in cols])
        _, r2 = _ols(X, d["delta"][ok])
        r2s[label] = r2
        L.append(f"| {label} | {', '.join(cols)} | {r2:+.3f} |")
    # 참고: 원래 쓰려던 서사/대화 이분 더미 — 52편 중 3편만 1이라 사실상 상수다
    _, r2_dummy = _ols(np.column_stack([d["density_gap"][ok], d["warmup_frac"][ok], narr[ok]]),
                       d["delta"][ok])
    L.append(f"| (참고) H1+H2+서사 더미 | density_gap, warmup_frac, 서사 여부 | {r2_dummy:+.3f} |")

    inc_auc = r2s["H1+H2+H3"] - r2s["H1+H2"]
    inc_h12 = r2s["H1+H2+H3"] - r2s["H3 신호품질만"]
    L += [f"\n**H1+H2 통제 후 신호 품질의 증분 R² = {inc_auc:+.3f}**",
          f"**신호 품질 통제 후 H1+H2 의 증분 R² = {inc_h12:+.3f}**\n",
          "🔴 **설계 정정.** H3 를 처음에 *서사/대화 이분 더미*로 넣었는데, 문서 52편 중 서사가 3편뿐이라",
          "사실상 상수였다(증분 R² +0.005). 더미로는 채널 효과를 **잴 수 없었던 것**이지 없었던 것이 아니다.",
          "올바른 조작화는 **신호 AUC**(임계를 걷어낸 순위력)이며 위 표는 그것으로 다시 돌린 결과다.",
          "\n| 어느 쪽이 큰가 | 판정 |", "|---|---|",
          "| 신호 품질 | **신호가 그 데이터에서 순위력이 없다.** 임계를 고쳐도 안 된다 ⇒ H3 방향 |",
          "| H1+H2 | **작동점·웜업 문제.** 고칠 수 있다 ⇒ 4번째 다운스트림 재검토 가능 |",
          "\n> 🔴 문서 52편, 그중 ami 2편·dev3 3편이다. 회귀는 **가설 분해용**이며 결론용이 아니다.",
          "> 절대값을 성능으로 인용하지 않는다.",
          "\n## 신호 AUC — 임계와 무관한 순위력\n",
          "임계 규칙을 걷어낸 신호 자체의 순위력. **하한 미달이 임계 탓인지 신호 탓인지** 가른다.\n",
          "| 세트 | 신호 AUC (문서 평균) |", "|---|---|"]
    for s in SETS:
        m_ = np.array([x["set"] == s for x in docs])
        L.append(f"| {s} | {np.nanmean(d['signal_auc'][m_]):.3f} |")
    L += ["\n> AUC 가 우연(0.5) 근처면 **신호가 없는 것**이고 임계를 고쳐도 안 된다(H3 방향).",
          "> AUC 는 살아 있는데 Δ 가 음수면 **임계·입도 문제**다(H1/H2 방향).",
          "\n## 세트별 판정 — 원인이 세트마다 다르다\n",
          "| 세트 | 신호 AUC | 예측/gold 밀도 | Δ | 진단 |", "|---|---|---|---|---|"]
    for s in SETS:
        m_ = np.array([x["set"] == s for x in docs])
        auc = float(np.nanmean(d["signal_auc"][m_]))
        pd_, gd_ = d["pred_density"][m_].mean(), d["gold_density"][m_].mean()
        dl = d["delta"][m_].mean()
        if auc < 0.5:
            diag = "**신호 역방향** — 임계로 해결 불가"
        elif auc < 0.58:
            diag = "**신호 거의 우연** — 임계로 해결 불가"
        elif dl < 0 and pd_ / max(gd_, 1e-9) >= 2.0:
            diag = f"**작동점 불일치** — 신호는 살아 있고 밀도만 {pd_/gd_:.1f}배"
        elif dl < 0:
            diag = "신호 있음 · 밀도 불일치 경미 — 미설명"
        else:
            diag = "통과"
        L.append(f"| {s} | {auc:.3f} | {pd_:.3f} / {gd_:.3f} | {dl:+.4f} | {diag} |")
    L += ["\n> 단일 통합 회귀는 이 차이를 가린다. **세트마다 다른 이유로 하한 아래에 있다.**"]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "DIAG_dialogue_floor_cause.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(docs, (OUT / "diag_dialogue_floor_cause.json").open("w"),
              ensure_ascii=False, indent=2)
    print("\n".join(L[:40]))
    print("\n->", OUT / "DIAG_dialogue_floor_cause.md")


if __name__ == "__main__":
    main()
