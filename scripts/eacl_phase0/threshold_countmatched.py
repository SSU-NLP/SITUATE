#!/usr/bin/env python3
r"""임계 규칙 비교 — **예측 개수를 맞춘 뒤** 비교한다 (작동점 통제).

## 왜 (D-B43 정정)
D-B43 은 c 를 1.0 으로 고정하고 누적 vs EWMA 를 비교해 *"국소성 자체가 해롭다"* 고 해석했다.
**이 해석은 과했다.** 같은 c=1.0 에서 규칙별 예측 경계 수가 dev-3 160 vs 226/184/167 로 **제각각**이라
비교된 것은 상당 부분 *국소성* 이 아니라 **서로 다른 작동점**이었다. 실제로 c=1.5 부터 순서가 뒤집힌다
(cumulative 0.3687 < ewma20 0.3851 < ewma50 0.3911).

⚠️ **AUC 는 네 규칙이 완전히 동일하다**(신호 `s = Σ_k P(reset_k)` 가 같으므로). 즉 규칙이 바꾸는 것은
**문턱을 어디에 긋느냐** 뿐이다. 그래서 고정 c 비교는 D-M12 가 경고한 *"μ+cσ 임계의 작동점 성분"* 함정에
그대로 빠진다.

## 무엇을
규칙마다 c 를 훑어 **예측 개수를 같게 맞춘 뒤** Score 를 비교한다. 그러면 작동점이 상쇄되고
**"같은 개수를 쓸 때 어느 규칙이 경계 자리를 더 잘 고르나"** 만 남는다.

🔴 **이것은 dev-3 에서 c 를 고르는 것이 아니라 작동점을 통제하는 것이다** — 맞추는 대상이 *점수* 가
아니라 *개수* 이므로 selection-on-dev 가 아니다. 기준 개수도 **현행 규칙(누적 c=1.0)이 스스로 낸 개수**라
gold 를 쓰지 않는다.

- **주 비교 = 현행 작동점 매칭**: 목표 개수 = 그 (모델·seed·셋) 에서 `cumulative c=1.0` 이 낸 개수.
  **gold 미사용.**
- **[참고] gold 개수 매칭**: 목표 = gold 개수. 🔴 **개수 누출 — 진단 전용, 성능 주장 금지**(CLAUDE.md).
- **[참고] 상한 포락선**: c 를 훑었을 때의 규칙별 최고 Score. 🔴 **c 를 평가셋에서 고른 값이라 채택 근거가
  될 수 없다.** 규칙의 잠재력 비교용 진단일 뿐이다.

## 실행
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eacl_phase0/threshold_countmatched.py --seeds 0
    .venv/bin/python scripts/eacl_phase0/threshold_countmatched.py --report
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
from db29_final_eval import load  # noqa: E402
from ewma_vs_cumulative_threshold import HDR, NS, OUT, SEEDS, SETS, SPANS  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402

MODELS = ("M5", "M4")
C_GRID = np.round(np.arange(0.0, 3.51, 0.05), 3)      # 작동점 통제용 — 선택용이 아님
RULES = [("cumulative", None)] + [("ewma", sp) for sp in SPANS]


def rule_label(kind: str, span: int | None) -> str:
    return "cumulative" if kind == "cumulative" else f"ewma-span{span}"


def preds(s: np.ndarray, kind: str, span: int | None, c: float) -> np.ndarray:
    seq = np.concatenate([[0.0], s]).tolist()
    mode = "online_nf" if kind == "cumulative" else "ewma"
    kw = {} if span is None else {"warmup": span}
    return np.asarray(adaptive_boundaries(seq, c=c, mode=mode, **kw), np.int8)[:len(s)]


_W: dict = {}


def _init(S, ys):
    _W["S"], _W["ys"] = S, ys


def _one(job):
    """워커: 한 (규칙, c) 의 (예측개수, Score). 병목은 GPU 가 아니라 nltk Pk/WD 채점이다."""
    kind, span, c = job
    ps, n = [], 0
    for s in _W["S"]:
        b = preds(s, kind, span, c)
        ps.append(np.concatenate([b, [0]]).tolist())
        n += int(b.sum())
    return n, score_dialogues(_W["ys"], ps)["score"]


def curves(S: list[np.ndarray], G: list[np.ndarray], workers: int) -> dict:
    """규칙 × c -> (예측개수, Score). 한 (모델·seed·셋) 전체. (규칙,c) 조합을 프로세스 병렬."""
    ys = [np.concatenate([g, [0]]).tolist() for g in G]
    jobs = [(k, sp, float(c)) for k, sp in RULES for c in C_GRID]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(S, ys)) as ex:
            got = list(ex.map(_one, jobs, chunksize=1))
    else:
        _init(S, ys)
        got = [_one(j) for j in jobs]
    out: dict[str, dict] = {}
    for i, (k, sp) in enumerate(RULES):
        part = got[i * len(C_GRID):(i + 1) * len(C_GRID)]
        out[rule_label(k, sp)] = {"c": C_GRID.tolist(),
                                  "n_pred": [x[0] for x in part],
                                  "score": [x[1] for x in part]}
    return out


def pick(cv: dict, target: int) -> dict:
    """목표 개수에 가장 가까운 c 의 (c, 개수, Score). 점수로 고르지 않는다."""
    i = int(np.argmin(np.abs(np.asarray(cv["n_pred"]) - target)))
    return {"c": cv["c"][i], "n_pred": cv["n_pred"][i], "score": cv["score"][i]}


def envelope(cv: dict) -> dict:
    """[진단] c 를 훑었을 때 최고 Score. 🔴 평가셋에서 고른 값 — 채택 근거 아님."""
    i = int(np.argmax(cv["score"]))
    return {"c": cv["c"][i], "n_pred": cv["n_pred"][i], "score": cv["score"][i]}


def run(seeds, sets_, bs: int, models, workers: int) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    OUT.mkdir(parents=True, exist_ok=True)
    corp = {s: Corpus(s) for s in sets_}
    for seed in seeds:
        res: dict = {"seed": seed, "c_grid": C_GRID.tolist(), "sets": {}}
        for tag in models:
            m = load(tag, seed, dev, ns=NS)[0]
            with torch.no_grad():
                for sname in sets_:
                    c_ = corp[sname]
                    P = axis_probs(m, c_, list(range(c_.n_docs)), dev, bs=bs, with_anchor=False)
                    S = [p.sum(1) for p in P]
                    G = [c_.gold[int(c_.to[i]):int(c_.to[i + 1])].astype(np.int8)
                         for i in range(c_.n_docs)]
                    cv = curves(S, G, workers)
                    n_gold = int(sum(int(g.sum()) for g in G))
                    # 🔒 목표 개수 = 현행 규칙이 스스로 낸 개수 (gold 미사용)
                    ci = int(np.argmin(np.abs(np.asarray(C_GRID) - 1.0)))
                    tgt = cv["cumulative"]["n_pred"][ci]
                    res["sets"].setdefault(sname, {})[tag] = {
                        "n_gold": n_gold, "target_n": tgt, "curves": cv,
                        "matched_incumbent": {k: pick(v, tgt) for k, v in cv.items()},
                        "matched_gold": {k: pick(v, n_gold) for k, v in cv.items()},
                        "envelope": {k: envelope(v) for k, v in cv.items()}}
                    mi = res["sets"][sname][tag]["matched_incumbent"]
                    print(f"  [{tag} seed{seed}] {sname:22s} 목표 {tgt:5d} | " + " ".join(
                        f"{k.replace('-mu','')}={mi[k]['score']:.4f}({mi[k]['n_pred']})"
                        for k in cv), flush=True)
            del m
            torch.cuda.empty_cache() if dev == "cuda" else None
        p = OUT / f"countmatched_seed{seed}.json"
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(f"-> {p}", flush=True)


def report() -> None:
    runs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(OUT.glob("countmatched_seed*.json"))]
    if not runs:
        sys.exit(f"산출 없음: {OUT}/countmatched_seed*.json")
    seeds = [r["seed"] for r in runs]
    sets_ = [s for s in SETS if s in runs[0]["sets"]]
    labels = [rule_label(k, sp) for k, sp in RULES]
    models = [m for m in MODELS if m in runs[0]["sets"][sets_[0]]]

    def g(sname, tag, block, rule, field):
        return [float(r["sets"][sname][tag][block][rule][field]) for r in runs]

    def ms(v):
        return f"{np.mean(v):.4f}" + (f" ± {np.std(v, ddof=1):.4f}" if len(v) > 1 else "")

    L = ["# 임계 규칙 — **예측 개수를 맞춘** 비교 (작동점 통제)\n",
         f"> seed {seeds} · `{NS}` · c 를 {C_GRID[0]}~{C_GRID[-1]} (step {C_GRID[1]-C_GRID[0]:.2f}) 로 훑어"
         " 목표 개수에 맞춤 · 신호·모델 불변\n",
         "## 왜 이 표가 필요한가 — D-B43 해석 정정\n",
         "D-B43 은 **c=1.0 고정**으로 비교하고 *\"국소성 자체가 해롭다\"* 고 해석했다. **그 해석은 과했다.**",
         "- 같은 c=1.0 에서 규칙별 예측 개수가 **제각각**이라(dev-3 160 vs 226/184/167) 비교된 것은"
         " 상당 부분 *국소성* 이 아니라 **서로 다른 작동점**이었다.",
         "- 실제로 c=1.5 부터 순서가 **뒤집힌다**(cumulative 0.3687 < ewma20 0.3851 < ewma50 0.3911).",
         "- ⚠️ **AUC 는 네 규칙이 완전히 동일하다** — 신호가 같기 때문이다. 규칙이 바꾸는 것은"
         " **문턱을 어디에 긋느냐** 뿐이므로, 고정 c 비교는 D-M12 가 경고한"
         " **\"μ+cσ 임계의 작동점 성분\"** 함정에 그대로 빠진다.\n",
         "🔴 **이 실험은 dev-3 에서 c 를 고르는 것이 아니다.** 맞추는 대상이 *점수* 가 아니라 *개수* 이고,"
         " 목표 개수도 **현행 규칙(누적 c=1.0)이 스스로 낸 값**이라 gold 를 쓰지 않는다 ⇒ 오염 아님.\n",
         "\n## 1. 🚦 주 결과 — 현행 작동점에 개수를 맞춘 Score (gold 미사용)\n"]
    for tag in models:
        L += [f"\n### {tag}\n",
              "| 규칙 | " + " | ".join(f"{HDR[s]}<br>Score (개수 / c)" for s in sets_) + " |",
              "|---|" + "---|" * len(sets_)]
        for rl in labels:
            row = []
            for sname in sets_:
                sc = g(sname, tag, "matched_incumbent", rl, "score")
                n = g(sname, tag, "matched_incumbent", rl, "n_pred")
                c = g(sname, tag, "matched_incumbent", rl, "c")
                row.append(f"{ms(sc)}<br>({np.mean(n):.0f} / c={np.mean(c):.2f})")
            L.append(f"| {rl}{' 🔒' if rl == 'cumulative' else ''} | " + " | ".join(row) + " |")
        L.append("| *목표 개수* | " + " | ".join(
            f"*{np.mean([r['sets'][s][tag]['target_n'] for r in runs]):.0f}*" for s in sets_) + " |")
        L.append("| *gold 개수* | " + " | ".join(
            f"*{runs[0]['sets'][s][tag]['n_gold']}*" for s in sets_) + " |")

    L += ["\n## 2. 🚦 판정 — 개수를 맞추면 국소성 효과가 남는가 (dev-3, M5)\n",
          "| 규칙 | " + " | ".join(f"seed{s} Δ" for s in seeds) + " | 평균 Δ | 판정 |",
          "|---|" + "---|" * (len(seeds) + 2)]
    verdict = {}
    for rl in labels[1:]:
        d = [a - b for a, b in zip(g("dev3", "M5", "matched_incumbent", rl, "score"),
                                   g("dev3", "M5", "matched_incumbent", "cumulative", "score"))]
        v = ("✅ 이김(전 seed +)" if all(x > 0 for x in d)
             else "❌ 짐(전 seed −)" if all(x < 0 for x in d) else "⚠️ 판정 불가(부호 혼재)")
        verdict[rl] = v
        L.append(f"| {rl} | " + " | ".join(f"{x:+.4f}" for x in d)
                 + f" | {np.mean(d):+.4f} | {v} |")
    L += ["", "> 개수가 같으므로 이 Δ 에는 작동점 성분이 없다 — **순수하게 '경계 자리를 어디로 고르나' 차이**다."]

    L += ["\n## 3. [참고] gold 개수 매칭 — 🔴 개수 누출, 진단 전용\n",
          "> gold 경계 수를 방법에 먹인다(matched@gold). **성능 주장에 쓰지 않는다**(CLAUDE.md).\n",
          "| 규칙 | " + " | ".join(f"{HDR[s]} Score (개수)" for s in sets_) + " |",
          "|---|" + "---|" * len(sets_)]
    for rl in labels:
        L.append(f"| {rl} | " + " | ".join(
            f"{ms(g(s, 'M5', 'matched_gold', rl, 'score'))} "
            f"({np.mean(g(s, 'M5', 'matched_gold', rl, 'n_pred')):.0f})" for s in sets_) + " |")

    L += ["\n## 4. [참고] 상한 포락선 — 🔴 c 를 평가셋에서 고른 값, 채택 근거 아님\n",
          "> 각 규칙이 c 를 자유롭게 골랐을 때의 최고 Score. **규칙의 잠재력 비교용 진단**이며,"
          " 이 c 를 동결하면 selection-on-dev 오염이다.\n",
          "| 규칙 | " + " | ".join(f"{HDR[s]} 최고 Score (개수 / c)" for s in sets_) + " |",
          "|---|" + "---|" * len(sets_)]
    for rl in labels:
        row = []
        for sname in sets_:
            sc = g(sname, "M5", "envelope", rl, "score")
            n = g(sname, "M5", "envelope", rl, "n_pred")
            c = g(sname, "M5", "envelope", rl, "c")
            row.append(f"{ms(sc)} ({np.mean(n):.0f} / c={np.mean(c):.2f})")
        L.append(f"| {rl} | " + " | ".join(row) + " |")

    L += ["\n## 5. 한계 (제거 금지)\n",
          "- 개수 매칭은 **총합** 기준이다. 문서별 개수까지 맞춘 것은 아니다.",
          "- c 격자 0.05 라 목표 개수와 정확히 일치하지 않을 수 있다(표의 실제 개수 병기).",
          "- dev-3 = 연습장 3편. 절대값은 성능이 아니다 — Δ·방향만.",
          "- 상한 포락선(§4)은 **평가셋에서 c 를 고른 값**이라 어떤 채택 근거도 되지 못한다.",
          f"- seed {seeds}."]
    (OUT / "REPORT_COUNTMATCHED.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {OUT / 'REPORT_COUNTMATCHED.md'}")
    for k, v in verdict.items():
        print(f"[판정] {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="임계 규칙 개수 매칭 비교 (작동점 통제)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--models", default="M5", help="기본 M5 만 (M4 는 이 질문에 불필요)")
    ap.add_argument("--workers", type=int, default=24, help="(규칙,c) 채점 프로세스 수")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    run([int(x) for x in a.seeds.split(",")], a.sets.split(","), a.bs,
        a.models.split(","), a.workers)


if __name__ == "__main__":
    main()
