#!/usr/bin/env python3
r"""D-B33 — 입력경로 × dynamics 2×2 분해를 **dev-3** 에서 평가한다.

## 무엇을 재나
`M5 − M4` 는 세 가지가 한꺼번에 바뀐 결과다: ①입력경로 분리(축별 extractor E_k) ②상태 분리
(5슬롯) ③dynamics 분리(축별 f_k). 여기서 ①과 ③을 따로 떼어내 **이득이 어디서 오는지** 가른다.

`d1=False`(=공유 투영) 의 의미: 벡터를 5블록으로 쪼개는 게 **아니라** 공유 투영 `proj(z)` 한
벡터가 5슬롯 전부에 **브로드캐스트**된다. 슬롯은 key 와 자기 상태로만 갈라진다 ⇒ *"표현은 하나의
벡터, 스키마 구조는 슬롯 상태·dynamics 에만"* = SEM 원안에 가장 가까운 대조군.

## 🔒 지표 규약 (D-B33 부칙 1·2 — 전부 결과 보기 **전**에 확정)
- **주지표 = dev-3 deployable Score.** 앵커 val 의 J_anchor 는 **판정에서 제외**(의사-gold 가
  폐기된 t\*=3 규칙에 기대어 순환). 산출 json 에 남은 값은 진단 기록일 뿐이다.
- **임계 선택을 하지 않는다.** t\*=3 도 r=0.0642 도 폐기됐고, dev-3 에서 다시 고르면 oracle 개수
  누출이다. 대신 **전 셀에 동일한 무선택 규칙 3종**(otsu · online μ+1.0σ · offline μ+1.0σ)을
  적용하고, 순위가 규칙에 걸쳐 안정적인지를 결론의 근거로 삼는다.
- 진단 2종 병기: **gold 밀도 matched**(⚠️ oracle 개수 — 성능 주장 금지) · **AUC**(임계 무관).
- **판정 = 3조건 동시 충족**: deployable 규칙 3종 중 ≥2종 방향 일치 · 3 seed 중 ≥2 · 3편 중 ≥2.
  하나라도 어긋나면 **"구별되지 않음"**.
- dev-3 = 3편 연습장(pieman 선택압 9회 누적). **절대 Score 인용 금지 — 셀 간 Δ 와 방향만.**

Score = `hi_ontop.dts_scoring.score_dialogues` 직접 호출(재구현 금지).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from db29_final_eval import load  # noqa: E402  (체크포인트 로더 재사용)
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from train_anchor_m4m5 import EXP, Corpus, axis_probs  # noqa: E402

OUT = REPO / "outputs/experiments/inputpath_dynamics_ablation"
SEEDS = (0, 1, 2)

# (표시명, 체크포인트 tag, 네임스페이스) — tag 는 파일명일 뿐 구조는 ns 가 정한다.
CELLS = [
    ("축별입력 × 축별dyn (M5)",     "M5", "_per_slot_slot"),
    ("축별입력 × 공유dyn (M6)",     "M6", "_shared_slot"),
    ("공유입력40 × 축별dyn",        "M5", "_per_slot_slot_sharedinput40"),
    ("공유입력40 × 공유dyn",        "M6", "_shared_slot_sharedinput40"),
    ("공유입력128 × 축별dyn",       "M5", "_per_slot_slot_sharedinput128"),
    ("단일슬롯 (M4)",               "M4", "_per_slot_slot"),
]
DEPLOYABLE = ["otsu", "online μ+1.0σ", "offline μ+1.0σ"]   # 판정에 쓰는 무선택 규칙
DIAG = ["[진단] gold밀도 matched"]                          # oracle 개수 — 성능 주장 금지


def rules_for(s: np.ndarray, n_gold: int) -> dict[str, np.ndarray]:
    """연속 점수 s [T-1] -> 규칙별 경계 이진열. 전 셀에 **동일하게** 적용된다."""
    seq = np.concatenate([[0.0], s]).tolist()          # adaptive_boundaries 는 첫 원소를 버린다
    out = {
        "otsu": np.asarray(adaptive_boundaries(seq, mode="otsu"), np.int8)[:len(s)],
        "online μ+1.0σ": np.asarray(
            adaptive_boundaries(seq, c=1.0, mode="online_nf"), np.int8)[:len(s)],
        "offline μ+1.0σ": np.asarray(
            adaptive_boundaries(seq, c=1.0, mode="offline"), np.int8)[:len(s)],
    }
    # ⚠️ 진단 전용: 문서의 gold 경계 **개수**를 주고 상위 n_gold 개만 예측 (oracle 누출).
    thr = np.sort(s)[::-1][min(n_gold, len(s)) - 1] if n_gold > 0 else np.inf
    out["[진단] gold밀도 matched"] = (s >= thr).astype(np.int8)
    return out


def eval_cell(m, dev3: Corpus, dev: str) -> dict:
    """한 체크포인트의 dev-3 결과: 규칙별 pooled Score + 문서별 Score + AUC."""
    probs = axis_probs(m, dev3, list(range(dev3.n_docs)), dev, bs=4, with_anchor=False)
    ys: list[list[int]] = []
    preds: dict[str, list[list[int]]] = {}
    per_doc: dict[str, dict] = {}
    auc_y, auc_s = [], []
    for i, pr in enumerate(probs):
        g = dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])].astype(np.int8)
        s = pr.sum(1)
        y = np.concatenate([g, [0]]).tolist()          # 끝-turn 규약: 마지막 문장 라벨 0
        ys.append(y)
        R = rules_for(s, int(g.sum()))
        doc: dict = {"n_gold": int(g.sum()), "n_trans": int(len(g))}
        for name, b in R.items():
            p = np.concatenate([b, [0]]).tolist()
            preds.setdefault(name, []).append(p)
            doc[name] = {"score": score_dialogues([y], [p])["score"], "n_pred": int(b.sum())}
        doc["auc"] = float(roc_auc_score(g, s)) if 0 < g.sum() < len(g) else float("nan")
        per_doc[dev3.doc_ids[i]] = doc
        auc_y.append(g)
        auc_s.append(s)
    res = {"per_doc": per_doc,
           "auc_pooled": float(roc_auc_score(np.concatenate(auc_y), np.concatenate(auc_s)))}
    for name, p in preds.items():
        res[name] = {"pooled_score": score_dialogues(ys, p)["score"],
                     "n_pred_total": int(sum(sum(x) for x in p))}
    return res


def j_anchor(tag: str, ns: str, seed: int) -> float:
    """진단 기록용으로만 회수한다(판정 제외 — 부칙 1)."""
    for cand in (EXP / f"results{ns}_{tag}_seed{seed}.json", EXP / f"results{ns}_seed{seed}.json"):
        if cand.exists():
            d = json.loads(cand.read_text())
            if tag in d:
                return float(d[tag].get("anchor_oof", {}).get("J_anchor", float("nan")))
    return float("nan")


def verdict(cell: dict, ref: dict) -> str:
    """🔒 사전 확정 판정 규칙 — 규칙 3종 중 ≥2 · seed 3 중 ≥2 · 문서 3편 중 ≥2 가 **전부** 같은 방향."""
    def sign(x: float) -> int:
        return 0 if abs(x) < 1e-9 else (1 if x > 0 else -1)
    rule_dir = [sign(np.mean(cell["by_rule"][r]) - np.mean(ref["by_rule"][r])) for r in DEPLOYABLE]
    n = min(len(cell["by_rule"][DEPLOYABLE[0]]), len(ref["by_rule"][DEPLOYABLE[0]]))
    seed_dir = [sign(np.mean([cell["by_rule"][r][i] for r in DEPLOYABLE])
                     - np.mean([ref["by_rule"][r][i] for r in DEPLOYABLE])) for i in range(n)]
    doc_dir = [sign(cell["by_doc"][d] - ref["by_doc"][d]) for d in cell["by_doc"]]
    for want in (1, -1):
        if (sum(x == want for x in rule_dir) >= 2 and sum(x == want for x in seed_dir) >= 2
                and sum(x == want for x in doc_dir) >= 2):
            return "**우세**" if want == 1 else "**열세**"
    return "구별되지 않음"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev3 = Corpus("dev3")
    OUT.mkdir(parents=True, exist_ok=True)

    raw: dict = {}
    agg: dict = {}
    for label, tag, ns in CELLS:
        got = {}
        for s in seeds:
            ckp = EXP / f"{tag}{ns}_seed{s}.pt"
            if not ckp.exists():
                print(f"[skip] {label} seed{s} — 체크포인트 없음 ({ckp.name})", flush=True)
                continue
            m, _ = load(tag, s, dev, ns)
            got[s] = eval_cell(m, dev3, dev)
            got[s]["J_anchor(진단)"] = j_anchor(tag, ns, s)
            print(f"  {label} seed{s} · otsu {got[s]['otsu']['pooled_score']:.4f}"
                  f" · onμ+σ {got[s]['online μ+1.0σ']['pooled_score']:.4f}"
                  f" · AUC {got[s]['auc_pooled']:.4f}", flush=True)
        if not got:
            continue
        raw[label] = got
        docs = list(next(iter(got.values()))["per_doc"])
        agg[label] = {
            "n_seed": len(got),
            "by_rule": {r: [got[s][r]["pooled_score"] for s in got] for r in DEPLOYABLE + DIAG},
            "n_pred": {r: [got[s][r]["n_pred_total"] for s in got] for r in DEPLOYABLE + DIAG},
            "auc": [got[s]["auc_pooled"] for s in got],
            # 문서별은 deployable 3종 평균 (규칙 하나를 고르지 않기 위해)
            "by_doc": {d: float(np.mean([got[s]["per_doc"][d][r]["score"]
                                         for s in got for r in DEPLOYABLE])) for d in docs},
            "J_anchor(진단)": [got[s]["J_anchor(진단)"] for s in got],
        }

    ref_label = CELLS[0][0]                                  # 기준 = M5 (축별입력 × 축별dyn)
    L = ["# D-B33 — 입력경로 × dynamics 2×2 분해 (dev-3)\n",
         "> 🔒 주지표 = dev-3 Score. **임계 선택 없음** — 전 셀 동일 규칙 3종(otsu · online μ+1.0σ ·",
         "> offline μ+1.0σ). t\\*=3 과 r=0.0642 는 폐기(부칙 1·2). J_anchor 는 **판정 제외**(진단 기록).",
         "> ⚠️ dev-3 = 3편 연습장, pieman 선택압 9회 누적 — **절대 Score 인용 금지, 셀 간 Δ·방향만**.",
         "> ⚠️ 앵커 라벨 k=1 · S_self 0.659(상한). 봉인 시험지 미개봉.\n",
         "## 1. 규칙별 pooled Score (mean ± std over seed)\n",
         "| 셀 | n seed | " + " | ".join(DEPLOYABLE) + " | AUC | " + DIAG[0] + " |",
         "|---|---|" + "---|" * (len(DEPLOYABLE) + 2)]
    for label in agg:
        a = agg[label]
        cols = [f"{np.mean(a['by_rule'][r]):.4f} ± {np.std(a['by_rule'][r]):.4f}"
                for r in DEPLOYABLE]
        cols.append(f"{np.mean(a['auc']):.4f}")
        cols.append(f"{np.mean(a['by_rule'][DIAG[0]]):.4f}")
        L.append(f"| {label} | {a['n_seed']} | " + " | ".join(cols) + " |")

    L += ["\n## 2. 기준(M5, 축별입력×축별dyn) 대비 Δ + 🔒 사전 판정\n",
          "| 셀 | " + " | ".join(f"Δ {r}" for r in DEPLOYABLE) + " | Δ AUC | 판정 |",
          "|---|" + "---|" * (len(DEPLOYABLE) + 2)]
    for label in agg:
        if label == ref_label:
            continue
        a, ref = agg[label], agg[ref_label]
        cols = [f"{np.mean(a['by_rule'][r]) - np.mean(ref['by_rule'][r]):+.4f}" for r in DEPLOYABLE]
        cols.append(f"{np.mean(a['auc']) - np.mean(ref['auc']):+.4f}")
        L.append(f"| {label} | " + " | ".join(cols) + f" | {verdict(a, ref)} |")

    L += ["\n## 3. 문서별 (deployable 3종 평균 Score, seed 평균)\n",
          "| 셀 | " + " | ".join(agg[ref_label]["by_doc"]) + " |",
          "|---|" + "---|" * len(agg[ref_label]["by_doc"])]
    for label in agg:
        L.append(f"| {label} | " + " | ".join(
            f"{agg[label]['by_doc'][d]:.4f}" for d in agg[label]["by_doc"]) + " |")

    L += ["\n## 4. 예측 경계 수 (gold 66) — 과/과소분절 진단\n",
          "| 셀 | " + " | ".join(DEPLOYABLE) + " |", "|---|" + "---|" * len(DEPLOYABLE)]
    for label in agg:
        L.append(f"| {label} | " + " | ".join(
            f"{np.mean(agg[label]['n_pred'][r]):.0f}" for r in DEPLOYABLE) + " |")

    L += ["\n## 5. [진단 기록] J_anchor — **판정에 쓰이지 않음**\n",
          "| 셀 | J_anchor (mean) |", "|---|---|"]
    for label in agg:
        v = [x for x in agg[label]["J_anchor(진단)"] if not np.isnan(x)]
        L.append(f"| {label} | {np.mean(v):.4f} |" if v else f"| {label} | — |")

    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps({"agg": agg, "raw": raw}, indent=2,
                                                 ensure_ascii=False, default=float))
    print("\n".join(L))
    print(f"\n-> {OUT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
