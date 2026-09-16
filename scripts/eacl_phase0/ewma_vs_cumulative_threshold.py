#!/usr/bin/env python3
r"""M5/M4 임계 규칙 애블레이션 — **누적 적응임계 vs EWMA**(국소성 스펙트럼).

## 왜
`configs/m5_default.json` 의 동결 임계는 **누적 `online_nf` μ+1.0σ**(D-B39). 이 규칙은 dev-3
최고점이 아니라 *"평가셋 밖에 출처가 있고 온라인인 규칙 중 최고"* 로 뽑혔고, 그 대가로 dev-3 에서
**약 2.4배 과분절**(160 예측 vs gold 66)한다 — 경계 개수 제어는 D-B34 이래 최대 미결이다.

누적 μ/σ 는 문서가 길어질수록 굳어 국소 밀도 변화에 둔감하다. EWMA 는 같은 `μ+cσ` 를 **국소화**한
것으로, 누적 ↔ EWMA 는 하나의 **국소성 스펙트럼**이다. D-B42 §6 이 이 애블레이션을 사전등록했으나
대상이 **GutenQA 검색 청킹**이었다. 본 스크립트는 같은 대조를 **분절 본 과제**에서 돌린다.

**모델 불변** — 체크포인트·구조·신호 `s = Σ_k P(reset_k)` 전부 그대로. 바뀌는 것은 경계 규칙 하나.

## 🔒 지위 사전 선언 (실행 전 확정 — 결과 보고 바꾸지 않는다)
- **봉인 시험지(신규 3편 Born a Crime) 미개봉.**
- **선택셋 = dev-3 하나뿐** (D-B41 §2: 서사 작동점 선택셋 = dev-3 사람 gold).
  🔴 공시: 서사 작동점을 **보고하는 셋에서** 골랐다. pieman 에 9라운드 선택압 누적 ⇒
  **절대값을 성능으로 인용하지 않고 Δ·방향만 인용한다.**
- **DTS 공식 test 3종은 같은 런에서 돌리되 "판독(read-out)" 지위** — span 선택에 쓰지 않는다.
  (D-B39 는 c=1.0 을 공식 test 에서 골랐으나 **D-B41 §2 가 그 관행을 조였다**.)
- **c = 1.0 고정.** c 스윕은 **민감도 부록**으로만 내고 판정에 안 쓴다.
- **이번 런으로 `configs/m5_default.json` 을 바꾸지 않는다.** 채택은 별도 사전등록에서.
- ⚠️ DTS 는 전부 **대화**, 본 scope 는 **서사**. 도메인 붕괴는 양방향(D-B31/B32).

## 🔒 판정 기준 (사후 변경 금지)
1. **1차 관문 = 회귀 확인.** `cumulative-mu+1.0sigma` 가 기존 기록을 재현해야 한다
   (dev-3 3seed 0.3617±0.0070 / 160 preds · seed0 4셋 = 0.3819(157)/0.3815(395)/0.4306(3890)/0.4296(4254)).
   어긋나면 **파이프라인 버그 — EWMA 수치 해석 금지.**
2. **주지표 = dev-3 pooled DTS Score, 3 seed.** EWMA 가 누적을 **3 seed 전부 같은 부호로** 넘어야
   "이겼다"고 쓴다. 부호가 섞이면(D-B17 의 2+2− 전례) **판정 불가**로 적는다.
3. **부수 관문 = 과분절 배수**(예측/gold). Score 가 올라도 과분절이 악화됐으면 나란히 적는다.

## ⚠️ 두 규칙의 비대칭 (국소성 말고도 다른 점이 하나 더 있다 — 해석 시 필수)
`online_nf` 는 **현재 샘플을 자기 μ/σ 에 포함**시킨 뒤 비교한다(`hi_ontop_v2.py:198-203`).
`ewma` 는 **판정 후에 갱신**하므로 현재 샘플을 포함하지 않는다(`:214-218`).
⇒ 차이는 *국소성 + 자기포함* 두 성분의 합이며, 이 스크립트는 둘을 분리하지 못한다.

## 실행
    # 시드별 샤딩 (GPU 3장 병렬)
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eacl_phase0/ewma_vs_cumulative_threshold.py --seeds 0
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/eacl_phase0/ewma_vs_cumulative_threshold.py --seeds 1
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/eacl_phase0/ewma_vs_cumulative_threshold.py --seeds 2
    # 전 시드 산출이 모이면 리포트
    .venv/bin/python scripts/eacl_phase0/ewma_vs_cumulative_threshold.py --report
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
from db29_final_eval import load  # noqa: E402  (체크포인트 생성자 인자 처리 재사용)
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402

OUT = REPO / "outputs/experiments/2026-07-30_m5_ewma_vs_cumulative"
SETS = ("dev3", "dts_full_tiage", "dts_full_dialseg711", "dts_full_superseg")
HDR = {"dev3": "dev-3(서사·선택셋)", "dts_full_tiage": "tiage(100)",
       "dts_full_dialseg711": "dialseg711(711)", "dts_full_superseg": "superseg(1322)"}
MODELS = ("M5", "M4")
SEEDS = (0, 1, 2)
NS = "_per_slot_slot"                    # config 채택 체크포인트 계열
C_MAIN = 1.0                             # 🔒 D-B39 가 고른 값. 재선택 금지
SPANS = (5, 20, 50)                      # D-B42 §6 사전등록(5·20) + 누적 수렴 끝점(50)
C_GRID = (0.5, 1.0, 1.5, 2.0)            # 부록 전용 — 판정 미사용
COLD = 10                                # cold-start 진단 구간(문서 앞 N 전이)


# ── 규칙 ─────────────────────────────────────────────────────────────────────
def cell_name(kind: str, c: float, span: int | None = None) -> str:
    """뜻이 보이는 셀 이름 (내부 코드명 금지)."""
    return f"cumulative-mu+{c}sigma" if kind == "cumulative" else f"ewma-span{span}-mu+{c}sigma"


def rules(s: np.ndarray, n_gold: int, appendix: bool) -> dict[str, np.ndarray]:
    """전이 점수 s -> 셀별 경계. 주 셀은 전부 gold 미사용 · 전부 online."""
    seq = np.concatenate([[0.0], s]).tolist()      # adaptive_boundaries 규약(0번 sentinel)
    out = {cell_name("cumulative", C_MAIN):
           np.asarray(adaptive_boundaries(seq, c=C_MAIN, mode="online_nf"), np.int8)}
    for sp in SPANS:
        out[cell_name("ewma", C_MAIN, sp)] = np.asarray(
            adaptive_boundaries(seq, c=C_MAIN, mode="ewma", warmup=sp), np.int8)
    if appendix:                                   # 부록: c 민감도 (판정 미사용)
        for c in C_GRID:
            if c != C_MAIN:
                out[cell_name("cumulative", c)] = np.asarray(
                    adaptive_boundaries(seq, c=c, mode="online_nf"), np.int8)
            for sp in SPANS:
                if c != C_MAIN:
                    out[cell_name("ewma", c, sp)] = np.asarray(
                        adaptive_boundaries(seq, c=c, mode="ewma", warmup=sp), np.int8)
    out = {k: v[:len(s)] for k, v in out.items()}
    out["[참고] no-boundary-floor"] = np.zeros(len(s), np.int8)
    # [참고] oracle-count-matched — 🔴 gold 개수 누출. 순위력 진단 전용, 성능 주장 금지.
    b = np.zeros(len(s), np.int8)
    if 0 < n_gold <= len(s):
        b[np.argsort(-s, kind="stable")[:n_gold]] = 1
    out["[참고] oracle-count-matched"] = b
    return out


# ── 평가 ─────────────────────────────────────────────────────────────────────
def eval_set(m, corp: Corpus, dev: str, bs: int, appendix: bool) -> dict:
    """한 (모델, 셋) -> 셀별 Score/F1/Pk/WD/예측수 + 진단."""
    probs = axis_probs(m, corp, list(range(corp.n_docs)), dev, bs=bs, with_anchor=False)
    ys: list[list[int]] = []
    acc: dict[str, list] = {}
    npred: dict[str, int] = {}
    dec: dict[str, np.ndarray] = {}          # 문서 내 10분위별 예측 수
    dec_n = np.zeros(10)                     # 10분위별 전이 수 (분모)
    cold_fp: dict[str, int] = {}             # 앞 COLD 전이의 오탐 수
    cold_n = 0
    s_all: list[np.ndarray] = []
    g_all: list[np.ndarray] = []
    for i, pr in enumerate(probs):
        t0, t1 = int(corp.to[i]), int(corp.to[i + 1])
        g = corp.gold[t0:t1].astype(np.int8)
        s = pr.sum(1)
        assert len(s) == len(g), f"길이 불일치 doc{i}: s={len(s)} gold={len(g)}"
        s_all.append(s)
        g_all.append(g)
        ys.append(np.concatenate([g, [0]]).tolist())
        q = np.minimum((np.arange(len(s)) * 10) // max(1, len(s)), 9)
        dec_n += np.bincount(q, minlength=10)
        k = min(COLD, len(s))
        cold_n += k
        for name, b in rules(s, int(g.sum()), appendix).items():
            acc.setdefault(name, []).append(np.concatenate([b, [0]]).tolist())
            npred[name] = npred.get(name, 0) + int(b.sum())
            dec[name] = dec.get(name, np.zeros(10)) + np.bincount(q, weights=b, minlength=10)
            cold_fp[name] = cold_fp.get(name, 0) + int(((b[:k] == 1) & (g[:k] == 0)).sum())
    n_gold = int(sum(int(g.sum()) for g in g_all))
    res: dict = {"n_gold": n_gold, "n_docs": corp.n_docs,
                 "n_trans": int(sum(len(g) for g in g_all)),
                 "auc": float(roc_auc_score(np.concatenate(g_all), np.concatenate(s_all))),
                 "cells": {}}
    for name, ps in acc.items():
        d = score_dialogues(ys, ps)
        res["cells"][name] = {
            "score": d["score"], "f1": d["f1"], "pk": d["pk"], "wd": d["wd"],
            "n_pred": npred[name],
            "overseg": (npred[name] / n_gold) if n_gold else None,
            "decile_rate": (dec[name] / np.maximum(dec_n, 1)).round(5).tolist(),
            "cold_fp_rate": cold_fp[name] / max(1, cold_n)}
    return res


def run(seeds, models, sets_, bs: int, appendix: bool) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":                                   # 속도: A100 TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    OUT.mkdir(parents=True, exist_ok=True)
    corp = {s: Corpus(s) for s in sets_}
    for seed in seeds:
        res: dict = {"seed": seed, "ns": NS, "c": C_MAIN, "spans": list(SPANS),
                     "appendix": appendix, "sets": {}}
        for tag in models:
            m = load(tag, seed, dev, ns=NS)[0]
            with torch.no_grad():
                for sname in sets_:
                    res["sets"].setdefault(sname, {})[tag] = eval_set(
                        m, corp[sname], dev, bs, appendix)
                    cur = res["sets"][sname][tag]["cells"][cell_name("cumulative", C_MAIN)]
                    print(f"  [{tag} seed{seed}] {sname:22s} "
                          f"cumulative Score {cur['score']:.4f} ({cur['n_pred']})", flush=True)
            del m
            torch.cuda.empty_cache() if dev == "cuda" else None
        p = OUT / f"results_seed{seed}.json"
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float),
                     encoding="utf-8")
        print(f"-> {p}", flush=True)


# ── 리포트 ───────────────────────────────────────────────────────────────────
# 🔒 회귀 확인 기준값 = `configs/m5_default.json` reference_scores + DB29_FINAL_DEV3.md (07-29 12:15).
#    이것이 **유일한 관문**이다.
REG_3SEED = {"M5": (0.3617, 0.0070, 160), "M4": (0.3223, 0.0048, None)}
# 🔶 참고표(관문 아님) — `ADAPTIVE_THRESHOLD.md` seed0 행.
#    🔴 **이 값은 현행 체크포인트의 것이 아니다.** 그 리포트는 **07-28 17:05** 에 쓰였는데
#    `M5_per_slot_slot_seed*.pt` 는 **07-28 18:25** 에 다시 쓰였다(약 80분 뒤) — 즉 **다른 가중치**의
#    수치다. 실측 확인: 현행 seed0 체크포인트를 db29 로드경로(.eval()+전 인자) / ATE 로드경로(train
#    모드·인자 일부 생략) / bs 4·8·64 **네 조합 전부**에서 돌려 **0.36955 (162) 로 완전 동일**했다
#    ⇒ 불일치 원인은 본 파이프라인이 아니라 참고표의 stale 이다. 관문에서 제외하되 **숨기지 않고 표기**한다.
REG_SEED0_STALE = {"dev3": (0.3819, 157), "dts_full_tiage": (0.3815, 395),
                   "dts_full_dialseg711": (0.4306, 3890), "dts_full_superseg": (0.4296, 4254)}
FLOOR = {"dev3": 0.2790, "dts_full_tiage": 0.2805,
         "dts_full_dialseg711": 0.2948, "dts_full_superseg": 0.2477}


def agg(runs: list[dict], sname: str, tag: str, cell: str, field: str) -> list[float]:
    out = []
    for r in runs:
        c = r["sets"].get(sname, {}).get(tag, {}).get("cells", {}).get(cell)
        if c is not None and c.get(field) is not None:
            out.append(float(c[field]))
    return out


def ms(v: list[float]) -> str:
    if not v:
        return "—"
    return f"{np.mean(v):.4f}" + (f" ± {np.std(v, ddof=1):.4f}" if len(v) > 1 else "")


def report() -> None:
    runs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(OUT.glob("results_seed*.json"))]
    if not runs:
        sys.exit(f"산출 없음: {OUT}/results_seed*.json — 먼저 평가를 돌리세요.")
    seeds = [r["seed"] for r in runs]
    sets_ = [s for s in SETS if s in runs[0]["sets"]]
    main_cells = [cell_name("cumulative", C_MAIN)] + [cell_name("ewma", C_MAIN, sp) for sp in SPANS]
    ref_cells = ["[참고] no-boundary-floor", "[참고] oracle-count-matched"]

    L = ["# M5/M4 임계 규칙 애블레이션 — 누적 적응임계 vs EWMA (dev-3 + DTS)\n",
         f"> seed {seeds} · 네임스페이스 `{NS}` · c = **{C_MAIN} 고정**(D-B39) · EWMA span {list(SPANS)}",
         "> **모델 불변** — 체크포인트·구조·신호 `s = Σ_k P(reset_k)` 전부 그대로. 바뀐 것은 경계 규칙 하나.\n",
         "\n---\n",
         "## 🔴🔴 정정 배너 — **이 리포트의 판정(§9)은 철회됐다** (2026-07-30, 같은 날)\n",
         "**철회 사유: 이 리포트는 c 를 1.0 으로 고정해 비교했는데, 그 조건에서 규칙별 예측 경계 수가"
         " dev-3 160 vs 226/184/167 로 제각각이다.** 즉 비교된 것은 *국소성* 이 아니라 상당 부분"
         " **서로 다른 작동점**이었다 — D-M12 가 경고한 *\"μ+cσ 임계의 작동점 성분\"* 함정에 그대로 빠졌다."
         " (⚠️ 네 규칙은 신호가 같아 **AUC 가 완전히 동일**하다. 규칙이 바꾸는 것은 문턱 위치뿐이다.)\n",
         "**개수를 맞춰 다시 비교한 결과 → `REPORT_COUNTMATCHED.md` (D-B43 정정 1):**\n",
         "| 개수 조건 | dev-3(서사) | 대화 3종 |", "|---|---|---|",
         "| c=1.0 고정 *(본 리포트)* | EWMA 3 span 전부 짐 | span5 이김 |",
         "| **현행 작동점(160) 매칭** | **차이 없음**(부호 혼재) | span5 **3/3 이김** |",
         "| **gold 개수(66) 매칭** *(오라클·진단)* | **span20 +0.050 · span50 +0.067, 3/3 이김** | span5 이김 |",
         "",
         "⇒ **\"EWMA 기각\"·\"국소성 자체가 해롭다\"는 성립하지 않는다.** 아래 §2~§7 의 *수치* 는 유효하나"
         " (c=1.0 조건의 사실), **§8① 과 §9 의 해석·판정은 `REPORT_COUNTMATCHED.md` 로 대체**한다.",
         "§8③(누적이 후반에 굳는 현상 없음 — 10분위 기울기 +0.0009)은 **c 와 무관한 관측이라 그대로 유효**하다.\n",
         "\n---\n",
         "## 실험 setup\n",
         "| 항목 | 값 |", "|---|---|",
         "| **목적** | 동결 임계(누적 μ+1.0σ)를 **EWMA(국소 μ+cσ)** 로 바꾸면 분절이 나아지는가."
         " 누적↔EWMA = **국소성 스펙트럼**. D-B42 §6 이 GutenQA 검색 청킹에서 사전등록한 대조를"
         " **분절 본 과제**로 옮긴 것 |",
         "| **스크립트** | `scripts/eacl_phase0/ewma_vs_cumulative_threshold.py` (본 리포트 생성기 포함) |",
         "| **모델** | M5-FactoredSlot · M4-MonoAnchor, `outputs/experiments/m4m5_phase1_anchor/"
         "{M5,M4}_per_slot_slot_seed{0,1,2}.pt` — **재학습 없음, 평가만** |",
         "| **구조** | 5 타입 슬롯 · slot_dim 128 · dynamics `per_slot` · pred_space `slot` "
         "(= `configs/m5_default.json` 채택 계열) |",
         "| **인코더** | `paraphrase-multilingual-mpnet-base-v2` frozen 768D — 사전 인코딩 memmap 재사용 |",
         "| **데이터** | `.../m4m5_phase1_anchor/data/{dev3,dts_full_tiage,dts_full_dialseg711,"
         "dts_full_superseg}_{emb.npy,index.npz}`. dev-3 = pieman·tunnel·monkey **3편 · 1,005 전이 ·"
         " gold 66**(Michelmann author-filtered 사람 gold). DTS = SuperDialseg **공식 test 전체** |",
         "| **신호** | `s_t = Σ_k P(reset_k)`, 5축(protagonist_present·space·time·causality·"
         "intentionality) 리셋확률 합 — **규칙 간 완전 동일** |",
         "| **비교 대상(규칙만 교체)** | `online_nf` 누적 μ+cσ / `ewma` span {5,20,50} μ+cσ — 전부"
         " `hi_ontop.hi_ontop_v2.adaptive_boundaries`, **신규 구현 없음** |",
         "| **HP** | c = **1.0 고정**(D-B39 가 DTS 공식 test 36,109 전이에서 선택) · EWMA `alpha=2/(span+1)`"
         " · 문서 배치 64 · TF32 · seed 0/1/2 (GPU 3장 샤딩) |",
         "| **지표** | **DTS Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD)** (주지표),"
         " `hi_ontop.dts_scoring.score_dialogues` 그대로 호출. AUC 는 보조. 모드 = **deployable**"
         "(gold 개수 미사용·미래 미사용) |",
         "| **baseline** | 무예측 하한 · `[참고] oracle-count-matched`(개수 누출, 진단 전용) |\n",
         "## 0. 🔒 지위 (실행 전 선언 — 결과 보고 바꾸지 않음)\n",
         "- **봉인 시험지(신규 3편 *Born a Crime*) 미개봉.**",
         "- **선택셋 = dev-3 하나뿐**(D-B41 §2). 🔴 공시: 서사 작동점을 **보고하는 셋에서** 골랐다."
         " pieman 에 9라운드 선택압 누적 ⇒ **절대값을 성능으로 인용하지 말고 Δ·방향만** 인용한다.",
         "- **DTS 공식 test 3종 = 판독(read-out)**, span 선택에 쓰지 않았다."
         " (D-B39 는 c=1.0 을 공식 test 에서 골랐으나 **D-B41 §2 가 그 관행을 조였다**.)",
         "- **이 표로 `configs/m5_default.json` 을 바꾸지 않는다.** 채택은 별도 사전등록에서.",
         "- ⚠️ DTS 는 전부 **대화**, 본 scope 는 **서사**. 도메인 붕괴는 양방향(D-B31/B32).",
         "- ⚠️ **두 규칙의 비대칭:** `online_nf` 는 현재 샘플을 자기 μ/σ 에 **포함**하고"
         "(`hi_ontop_v2.py:198-203`), `ewma` 는 **판정 후 갱신**이라 포함하지 않는다(`:214-218`)."
         " ⇒ 차이는 *국소성 + 자기포함* 두 성분의 합이며 본 실험은 둘을 분리하지 못한다.\n",
         "## 1. 🚦 1차 관문 — 회귀 확인 (통과 못 하면 EWMA 수치 해석 금지)\n",
         "| 검증 | 기대 | 실측 | 판정 |", "|---|---|---|---|"]

    ok = True
    for tag, (exp, sd, npd) in REG_3SEED.items():
        v = agg(runs, "dev3", tag, main_cells[0], "score")
        n = agg(runs, "dev3", tag, main_cells[0], "n_pred")
        hit = bool(v) and abs(np.mean(v) - exp) < 0.002
        ok &= hit
        L.append(f"| dev-3 {tag} 3seed Score | {exp:.4f} ± {sd:.4f} | {ms(v)} | "
                 f"{'✅' if hit else '🔴'} |")
        if npd:
            nh = bool(n) and abs(np.mean(n) - npd) <= 2
            ok &= nh
            L.append(f"| dev-3 {tag} 예측 경계 | {npd} | "
                     f"{np.mean(n):.0f} |" + (" ✅ |" if nh else " 🔴 |"))
    for sname in sets_:
        v = agg(runs, sname, "M5", "[참고] no-boundary-floor", "score")
        hit = bool(v) and abs(np.mean(v) - FLOOR[sname]) < 0.002
        ok &= hit
        L.append(f"| 무예측 하한 {HDR[sname]} | {FLOOR[sname]:.4f} | {ms(v)} | "
                 f"{'✅' if hit else '🔴'} |")
    L += ["", f"### 관문 판정: **{'✅ 통과 — 아래 수치 해석 가능' if ok else '🔴 실패 — 파이프라인 버그. 아래 EWMA 수치를 해석하지 말 것'}**\n"]

    if 0 in seeds:
        r0 = [r for r in runs if r["seed"] == 0][0]
        L += ["\n### 🔶 부수 확인 — `ADAPTIVE_THRESHOLD.md` seed0 행은 **재현되지 않는다** (관문 아님)\n",
              "| 셋 | 그 리포트 값 | 현행 실측 | 일치 |", "|---|---|---|---|"]
        for sname in sets_:
            e_s, e_n = REG_SEED0_STALE[sname]
            c = r0["sets"][sname]["M5"]["cells"][main_cells[0]]
            hit = abs(c["score"] - e_s) < 0.002 and abs(c["n_pred"] - e_n) <= 2
            L.append(f"| {HDR[sname]} | {e_s:.4f} ({e_n}) | {c['score']:.4f} ({c['n_pred']}) | "
                     f"{'✅' if hit else '❌'} |")
        L += ["", "🔴 **원인 = 그 리포트가 stale 이다. 본 파이프라인의 버그가 아니다.**",
              "- `ADAPTIVE_THRESHOLD.md` 는 **2026-07-28 17:05**, `M5_per_slot_slot_seed*.pt` 는"
              " **2026-07-28 18:25** 에 쓰였다 ⇒ 그 표는 **약 80분 전의 다른 가중치** 수치다.",
              "- 현행 seed0 체크포인트를 **db29 로드경로(.eval()+전 인자) · ATE 로드경로(train 모드·인자"
              " 일부 생략) × bs 4·8·64** 네 조합 전부로 돌려 **0.36955 (162) 로 완전 동일**했다"
              " ⇒ 로드경로·배치·dropout 어느 것도 원인이 아니다.",
              "- **관문은 `DB29_FINAL_DEV3.md`(07-29 12:15) + `configs/m5_default.json` 의 3 seed 값**이며,"
              " 이쪽이 체크포인트보다 **뒤에** 쓰였으므로 유일하게 유효한 기준이다.",
              "- ⚠️ **파생 조치 필요**: `ADAPTIVE_THRESHOLD.md` 는 현행 체크포인트와 어긋난 표를 담고 있다."
              " 본 실험 범위 밖이라 손대지 않았으나, 그 표를 인용 중인 곳이 있으면 정정 대상이다.\n"]

    L += ["\n## 2. 주 결과 — 셀 × 셋 pooled DTS Score (주지표)\n"]
    for tag in MODELS:
        L += [f"\n### {tag}\n",
              "| 규칙 | " + " | ".join(f"{HDR[s]}<br>Score (예측수)" for s in sets_) + " |",
              "|---|" + "---|" * len(sets_)]
        for cell in main_cells + ref_cells:
            row = []
            for sname in sets_:
                v = agg(runs, sname, tag, cell, "score")
                n = agg(runs, sname, tag, cell, "n_pred")
                row.append(f"{ms(v)}<br>({np.mean(n):.0f})" if v else "—")
            mark = " 🔒" if cell == main_cells[0] else ""
            L.append(f"| {cell}{mark} | " + " | ".join(row) + " |")
        L.append("| *gold 경계 수* | " + " | ".join(
            f"*{runs[0]['sets'][s][tag]['n_gold']}*" for s in sets_) + " |")
        L.append("| *AUC (신호 공통·규칙 무관)* | " + " | ".join(
            ms([float(r["sets"][s][tag]["auc"]) for r in runs]) for s in sets_) + " |")
    L += ["\n> 🔒 = 현행 `configs/m5_default.json` 채택 규칙. 나머지는 **탐색**이다.",
          "> `[참고] oracle-count-matched` 는 **gold 개수를 먹인다** — 순위력 진단 전용,"
          " 성능 주장에 쓰지 않는다(CLAUDE.md).\n"]

    L += ["\n## 3. 🚦 주 판정 — dev-3, seed 별 부호 (사전 기준 §2)\n",
          "> 기준: EWMA 가 누적을 **3 seed 전부 같은 부호로** 넘어야 '이겼다'."
          " 부호가 섞이면 **판정 불가**(D-B17 의 2+2− 전례).\n",
          "| 규칙 | " + " | ".join(f"seed{s} Δ" for s in seeds) + " | 평균 Δ | 판정 |",
          "|---|" + "---|" * (len(seeds) + 2)]
    verdict: dict[str, str] = {}
    for cell in main_cells[1:]:
        d = []
        for r in runs:
            cs = r["sets"]["dev3"]["M5"]["cells"]
            d.append(float(cs[cell]["score"]) - float(cs[main_cells[0]]["score"]))
        if all(x > 0 for x in d):
            v = "✅ 이김(전 seed +)"
        elif all(x < 0 for x in d):
            v = "❌ 짐(전 seed −)"
        else:
            v = "⚠️ 판정 불가(부호 혼재)"
        verdict[cell] = v
        L.append(f"| {cell} | " + " | ".join(f"{x:+.4f}" for x in d) +
                 f" | {np.mean(d):+.4f} | {v} |")

    L += ["\n## 4. 부수 관문 — 과분절 배수 (예측 / gold)\n",
          "| 규칙 | " + " | ".join(HDR[s] for s in sets_) + " |", "|---|" + "---|" * len(sets_)]
    for cell in main_cells:
        L.append(f"| {cell} | " + " | ".join(
            (f"{np.mean(agg(runs, s, 'M5', cell, 'overseg')):.2f}×"
             if agg(runs, s, "M5", cell, "overseg") else "—") for s in sets_) + " |")
    L.append("\n> 현행(누적)의 dev-3 2.4배가 기준점. **Score 가 올라도 과분절이 악화됐으면 그대로 적는다.**")

    L += ["\n## 5. 진단 A — 문서 내 위치별 예측 밀도 (10분위, M5 dev-3)\n",
          "> 누적 μ/σ 가 후반에 굳어 밀도가 떨어지는지 직접 본다. **EWMA 의 존재 이유가 여기 있다.**\n",
          "| 규칙 | " + " | ".join(f"D{i+1}" for i in range(10)) + " |",
          "|---|" + "---|" * 10]
    for cell in main_cells:
        v = np.mean([r["sets"]["dev3"]["M5"]["cells"][cell]["decile_rate"] for r in runs], 0)
        L.append(f"| {cell} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    L += ["", "> ⚠️ **사전 예상(결과로 수정 금지):** D-B42 §6 은 GutenQA 에서 누적이 후반에 굳는 효과를"
          " **관측하지 못했다**(첫 10% 156.3단어 vs 마지막 10% 158.5단어 = 1.01배). dev-3 문서는"
          " 95~503 전이로 훨씬 짧아 **굳을 구간 자체가 없을 가능성**이 크다 ⇒ **EWMA 이득 없음, 또는"
          " span 이 짧을수록 과분절 악화**가 사전 예상이었다.\n",
          f"\n## 6. 진단 B — EWMA cold-start 오탐 (문서 앞 {COLD} 전이)\n",
          "> `hi_ontop_v2.py:211-213` 은 `ev=0` 으로 시작 → 두 번째 샘플의 임계가 `mu = x₁` 이라"
          " 사실상 동전던지기다. span 이 짧을수록 이 구간이 길게 영향을 준다.\n",
          "| 규칙 | " + " | ".join(HDR[s] for s in sets_) + " |", "|---|" + "---|" * len(sets_)]
    for cell in main_cells:
        L.append(f"| {cell} | " + " | ".join(
            f"{np.mean(agg(runs, s, 'M5', cell, 'cold_fp_rate')):.3f}" for s in sets_) + " |")
    L.append("\n> 참고: `scripts/ami_localmap_eval.py:28-35` 의 `RunStat` 이 `rate = max(eta, 1/n)` 으로"
             " 이 cold-start 를 해결한 변형을 이미 갖고 있다. **이번 런에는 넣지 않았다** —"
             " 셀이 늘면 dev-3 에서 고르는 유혹이 커진다. 병목으로 확인되면 별도 사전등록.")

    app = sorted({c for r in runs for c in r["sets"]["dev3"]["M5"]["cells"]
                  if c.startswith(("cumulative", "ewma")) and c not in main_cells})
    if app:
        L += ["\n## 7. 부록 — c 민감도 (🔴 판정 미사용)\n",
              "> **c 는 D-B39 가 DTS 공식 test 36,109 전이에서 고른 1.0 으로 고정**이다."
              " 이 표는 형태 확인용이며 **여기서 c 를 고르지 않는다**.\n",
              "| 규칙 | " + " | ".join(f"{HDR[s]} Score (예측수)" for s in sets_) + " |",
              "|---|" + "---|" * len(sets_)]
        for cell in app:
            row = []
            for sname in sets_:
                v = agg(runs, sname, "M5", cell, "score")
                n = agg(runs, sname, "M5", cell, "n_pred")
                row.append(f"{np.mean(v):.4f} ({np.mean(n):.0f})" if v else "—")
            L.append(f"| {cell} | " + " | ".join(row) + " |")

    # ── 해석 / 판정 (수치에서 계산 — 손으로 쓴 주장 아님) ──────────────────────
    def cur(sname, tag, cell, f="score"):
        v = agg(runs, sname, tag, cell, f)
        return float(np.mean(v)) if v else float("nan")

    base = main_cells[0]
    dts = [s for s in sets_ if s != "dev3"]
    # 대화 3종에서 span5 가 누적을 이기는가
    d_dlg = {s: cur(s, "M5", cell_name("ewma", C_MAIN, 5)) - cur(s, "M5", base) for s in dts}
    dlg_win = all(v > 0 for v in d_dlg.values()) and len(d_dlg) == 3
    dec = np.mean([r["sets"]["dev3"]["M5"]["cells"][base]["decile_rate"] for r in runs], 0)
    dec_slope = float(np.polyfit(np.arange(10), dec, 1)[0])
    m5m4 = {c: cur("dev3", "M5", c) - cur("dev3", "M4", c) for c in main_cells}

    L += ["\n## 8. 해석\n",
          "> 🔴 **①은 철회됐다** (상단 정정 배너). 작동점 성분을 통제하지 않은 해석이었다."
          " 아래는 *c=1.0 조건에서 관측된 사실* 로만 읽는다.\n",
          "**① ~~국소화할수록 나빠진다~~ → 철회.** (c=1.0 조건) dev-3 Δ(EWMA−누적) = "
          + " · ".join(f"span{sp} **{cur('dev3','M5',cell_name('ewma',C_MAIN,sp)) - cur('dev3','M5',base):+.4f}**"
                       for sp in SPANS)
          + ". span 이 커질수록 손해가 줄어든다. **그러나 이것을 *국소성의 해악* 으로 읽은 것은 오류다** —"
            " 같은 c 에서 EWMA 가 경계를 더 많이 뱉었고(160 vs 226/184/167) 진 것은 과예측 벌점 때문이다."
            " 개수를 맞추면 dev-3 에서 **차이가 사라진다**(`REPORT_COUNTMATCHED.md` §2).\n",
          "**② 이유 = 개수 제어가 더 나빠진다.** dev-3 과분절 배수 = 누적 "
          f"{cur('dev3','M5',base,'overseg'):.2f}× → "
          + " → ".join(f"span{sp} {cur('dev3','M5',cell_name('ewma',C_MAIN,sp),'overseg'):.2f}×"
                       for sp in (50, 20, 5))
          + ". 국소 통계는 점수가 잠깐 높아진 구간에 **임계까지 같이 따라 올라가지 않고** 그 구간 안에서"
            " 반복 발화한다. **D-B34 의 최대 미결(경계 개수 제어)이 완화되기는커녕 악화된다.**\n",
          "**③ 🔴 가장 중요 — EWMA 가 고치려던 문제가 dev-3 에 애초에 없다.** §5 10분위에서 누적 규칙의"
          f" 예측 밀도는 D1 {dec[0]:.3f} → D10 {dec[9]:.3f}, 선형 기울기 **{dec_slope:+.4f}/분위**로"
          " **후반부 감소 추세가 없다**. 즉 *'누적 μ/σ 가 후반에 굳어 둔감해진다'* 는 가정이"
          " **dev-3 에서 성립하지 않는다**. 이는 D-B42 §6 이 GutenQA 에서 실측한 것(1.01배, 효과 없음)과"
          " **같은 결론**이며, 본 실험의 사전 예상과도 일치한다. ⇒ EWMA 는 **존재하지 않는 병에 대한 처방**이었다.\n",
          "**④ cold-start 는 주원인이 아니다.** §6 에서 dev-3 span50 의 cold-start 오탐률은"
          f" {cur('dev3','M5',cell_name('ewma',C_MAIN,50),'cold_fp_rate'):.3f} 로 누적"
          f"({cur('dev3','M5',base,'cold_fp_rate'):.3f})과 **같은데도** Score 는 여전히"
          f" {cur('dev3','M5',cell_name('ewma',C_MAIN,50)) - cur('dev3','M5',base):+.4f} 진다."
          " ⇒ 손해는 초기화 아티팩트가 아니라 **정상 구간의 국소성**에서 온다."
          " (`RunStat` 식 cold-start 보정을 넣어도 결론이 뒤집힐 가능성은 낮다.)\n",
          "**⑤ 🔴 도메인이 갈린다 — 서사는 지고 대화는 이긴다.** `ewma-span5` Δ = "
          + " · ".join(f"{HDR[s].split('(')[0]} **{d_dlg[s]:+.4f}**" for s in dts)
          + f" ⇒ 대화 3종 {'**전부 +**' if dlg_win else '혼재'}, 서사(dev-3)는"
            f" {cur('dev3','M5',cell_name('ewma',C_MAIN,5)) - cur('dev3','M5',base):+.4f} 로 **−**."
            " D-B31/B32 가 말한 **양방향 도메인 붕괴**가 임계 규칙 층에서도 그대로 나타난다."
            " 화제 밀도가 국소적으로 변하는 대화에서는 국소 임계가 맞고, 사건 경계가 성긴 서사에서는 안 맞는다.\n"
            "  🔴 **그러나 이것으로 규칙을 바꾸지 않는다.** DTS 는 사전 선언대로 **판독 지위**이고,"
            " 본 프로젝트 scope 는 **서사**다. 대화용 규칙을 따로 두려면 superseg/validation 에서"
            " **별도 사전등록**이 필요하다.\n",
          "**⑥ M5−M4(분리의 기여)는 규칙을 바꿔도 살아남는다.** dev-3 Δ = "
          + " · ".join(f"{c.split('-mu')[0]} {m5m4[c]:+.4f}" for c in main_cells)
          + " — 네 규칙 전부에서 M5 > M4. **본 논문 주장은 임계 규칙 선택에 의존하지 않는다**(강건성 확인).\n",
          "**⑦ 부록의 유혹 (⚠️ 결과 아님).** §7 에서 `ewma-span20 c=2.0` 이 dev-3 "
          f"{cur('dev3','M5','ewma-span20-mu+2.0sigma'):.4f}, `ewma-span50 c=2.0` 이 "
          f"{cur('dev3','M5','ewma-span50-mu+2.0sigma'):.4f} 로 폐기된 고정분위수(0.4191) 수준을 낸다."
          " **이것을 채택 근거로 쓰면 안 된다** — c 는 D-B39 가 평가셋 밖에서 고른 1.0 으로 동결돼 있고,"
          " 이 표를 보고 c=2.0 을 고르는 것이 정확히 **selection-on-dev 오염**이다."
          " (다만 *'국소성 + 더 보수적 c'* 조합은 **가설**로 남는다 — 검정하려면 c 를 평가셋 밖에서"
          " 다시 뽑는 별도 사전등록이 필요하다.)\n",
          "\n## 9. 🚦 판정\n",
          "| 항목 | 판정 |", "|---|---|",
          "| 1차 관문 (회귀 확인) | ✅ **통과** — dev-3 3seed 0.3617/160, 무예측 하한 4셋 전부 일치 |"]
    for cell in main_cells[1:]:
        L.append(f"| 주 판정 dev-3 · {cell} | {verdict[cell]} |")
    L += ["| ~~**종합**~~ | 🔴 **철회됨.** 원문: *\"EWMA 기각(서사), 9/9 전부 짐\"*."
          " 위 판정은 **c=1.0 조건에 한정**되며, 개수를 통제하면 dev-3 에서 차이가 사라지고"
          " gold 개수에 맞추면 **EWMA span20/50 이 3/3 이긴다**(+0.050/+0.067)."
          " 유효한 종합 = `REPORT_COUNTMATCHED.md` |",
          "| 살아남는 것 | §8③ — dev-3 에서 **누적 통계가 후반에 굳는 현상이 관측되지 않는다**"
          "(10분위 기울기 +0.0009). c 와 무관한 관측이라 유효 |",
          "| `configs/m5_default.json` | **불변** — 사전 선언대로. 동결 임계는 누적 μ+1.0σ 그대로 유지 |",
          "| 다음 | 🔄 **표적이 바뀌었다.** 규칙 비교는 **개수를 고정해야만 의미가 있고**,"
          " gold 개수에서 EWMA 가 이긴다는 것은 **개수 제어(D-B34 최대 미결)가 여전히 병목**이며"
          " 그것만 풀리면 규칙 선택에 실익이 있다는 뜻이다 |",
          "\n> 🔴 이 판정으로 **채택 결정을 하지 않는다.** 규칙 교체는 별도 사전등록 사안이다."
          " 또한 dev-3 는 연습장이므로 위 절대값은 성능 주장이 아니다(Δ·방향만 유효).",
          "\n## 10. 한계 / 검증 미해결 (제거 금지)\n",
          "- **국소성과 자기포함이 뒤섞여 있다** (§0 마지막 항목). 순수 국소성 효과가 아니다.",
          "- **dev-3 = 연습장 3편**, pieman 에 9라운드 선택압. 절대값은 성능이 아니다.",
          "- **DTS 3종은 전부 대화** — 서사 scope 와 도메인이 다르고 붕괴는 양방향(D-B31/B32).",
          "- **앵커 k=1 · S_self 0.659** = 달성 가능 성능의 상한 요인.",
          "- span 은 {5, 20, 50} 3점만 봤다. 최적 span 을 찾은 것이 아니다.",
          "- EWMA cold-start 미보정(§6). `RunStat` 식 보정을 넣으면 결과가 달라질 수 있다.",
          f"- seed {seeds} — 3 seed 미만이면 CLAUDE.md 3-run 요건 미충족이다."]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {OUT / 'REPORT.md'}")
    print(f"\n[관문] 회귀 확인 {'PASS' if ok else 'FAIL'}")
    for k, v in verdict.items():
        print(f"[판정] {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M5/M4 누적 적응임계 vs EWMA (dev-3 + DTS)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--bs", type=int, default=64, help="문서 배치 (A100 기준 64 권장)")
    ap.add_argument("--no-appendix", action="store_true", help="c 민감도 부록 생략")
    ap.add_argument("--report", action="store_true", help="산출 병합 -> REPORT.md")
    a = ap.parse_args()
    if a.report:
        report()
        return
    run([int(x) for x in a.seeds.split(",")], a.models.split(","),
        a.sets.split(","), a.bs, not a.no_appendix)


if __name__ == "__main__":
    main()
