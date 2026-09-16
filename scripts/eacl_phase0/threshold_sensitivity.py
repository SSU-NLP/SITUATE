#!/usr/bin/env python3
"""작동점(적응임계) 민감도 — 같은 점수에 임계 규칙만 갈아 끼운다.

**이것은 작동점 선택이 아니다.** 채택 작동점은 `online_nf c=1.0`(D-B39) 그대로이고,
본 스크립트는 "모델 순위가 작동점에 얼마나 의존하는가"를 재는 **민감도 분석**이다.
결과를 보고 채택 작동점을 바꾸지 않는다(`no-postresult-criterion-change`).

비교하는 작동점 4종 — 모델·체크포인트·점수는 전부 동일하고 이진화 규칙만 다르다:

| 이름 | 규칙 | 출처 | 온라인 |
|---|---|---|---|
| `adopted_c1.0` | `s_t >= μ_<t + 1.0·σ_<t` (전체 과거) | D-B39. c=1.0 은 DTS 공식 test 36,109 전이가 선택 | ✅ |
| `emllm_gamma1.1_fullpast` | `s_t > μ_<t + 1.1·σ_<t` (전체 과거) | EM-LLM 계수만 이식. γ 효과와 창 효과를 분리하려는 대조군 | ✅ |
| `emllm_sm_win4096` | `s_t > μ(win) + 1.1·σ(win)`, win = 직전 4096 **토큰** | EM-LLM 원 규칙 그대로 (`em_llm.py:19`, `llama3.yaml:19`) | ✅ |
| `quantile_0.0660` | 문서 내 상위 6.60% 분위수 | I2 앵커 경계율. **개수를 고정**해 모델 간 BOR 을 같게 만든다 | ❌ 문서 전체 필요 |

🔴 `quantile` 은 문서 전체 분포가 있어야 임계가 정해지므로 **오프라인**이다. 온라인 주장에는
쓸 수 없고, 여기서는 "개수를 통제하면 모델이 구분되는가"를 보는 **진단용**으로만 넣는다.

γ·n_local 은 EM-LLM 발표값 고정 — 우리 데이터로 튜닝하지 않는다(`baselines-stay-as-published`).

산출: `outputs/experiments/<name>/{results.json, REPORT.md}`

    .venv/bin/python scripts/eacl_phase0/threshold_sensitivity.py --name 2026-08-04_threshold_sensitivity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
sys.path.insert(0, str(REPO / "EACL/methods"))

from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

GAMMA = 1.1        # EM-LLM 원본 기본값 (em_llm.py:19)
N_LOCAL = 4096     # EM-LLM 원본 기본값 (config/llama3.yaml:19), 토큰 단위
WARMUP = 3         # EM-LLM 원본 warmup
RATE = 0.0660      # I2 앵커 경계율


# ── 작동점 ───────────────────────────────────────────────────────────────────
def bnd_running(s: np.ndarray, c: float) -> np.ndarray:
    """`adaptive_boundaries(mode='online_nf')` 규약 — 0번 sentinel 을 앞에 붙인다."""
    return np.asarray(adaptive_boundaries(np.concatenate([[0.0], s]).tolist(),
                                          c=c, mode="online_nf"), np.int8)[:len(s)]


def bnd_emllm_window(s: np.ndarray, tok_len: np.ndarray, gamma: float,
                     n_local: int, warmup: int) -> np.ndarray:
    """EM-LLM SM 규칙. `EACL/methods/emllm/run_sm_dev3.py:sm_boundaries` 와 동일한 로직.

    차이는 sentinel 처리 하나뿐 — 우리 점수열은 전이 단위(길이 n_sent-1)라 앞에 0 을 붙여
    문장 단위로 맞추는데, 그 sentinel(index 0)은 창 통계에서 제외한다(원본 surprise 에는
    sentinel 이 없으므로 포함하면 μ 가 낮아져 규칙이 달라진다).
    `--validate` 로 원본 surprise 에 대해 이 함수가 기록된 EM-LLM 행을 재현하는지 검증한다.
    """
    s_sent = np.concatenate([[0.0], s])              # index 0 = sentinel
    n = len(s_sent)
    cum = np.concatenate([[0], np.cumsum(tok_len[:n])])
    out = np.zeros(len(s), np.int8)
    for i in range(1, n):
        lo_tok = max(0, cum[i] - n_local)
        st = int(np.searchsorted(cum, lo_tok, side="right") - 1)
        st = max(st, 1)                              # sentinel 제외
        past = s_sent[st:i]
        if len(past) < warmup:
            continue
        if s_sent[i] > past.mean() + gamma * past.std():
            out[i - 1] = 1
    return out


def bnd_quantile(s: np.ndarray, rate: float) -> np.ndarray:
    return (s >= np.quantile(s, 1.0 - rate)).astype(np.int8)


OPS = ["adopted_c1.0", "emllm_gamma1.1_fullpast", "emllm_sm_win4096", "quantile_0.0660"]


def apply_op(op: str, s: np.ndarray, tok_len: np.ndarray | None) -> np.ndarray:
    if op == "adopted_c1.0":
        return bnd_running(s, 1.0)
    if op == "emllm_gamma1.1_fullpast":
        return bnd_running(s, GAMMA)
    if op == "emllm_sm_win4096":
        if tok_len is None:
            raise RuntimeError("토큰 길이 없이 EM-LLM 창 규칙을 쓸 수 없다")
        return bnd_emllm_window(s, tok_len, GAMMA, N_LOCAL, WARMUP)
    if op == "quantile_0.0660":
        return bnd_quantile(s, RATE)
    raise ValueError(op)


# ── 지표 ─────────────────────────────────────────────────────────────────────
def metrics(g: np.ndarray, s: np.ndarray, b: np.ndarray) -> dict:
    y = np.concatenate([g, [0]]).tolist()
    p = np.concatenate([b, [0]]).tolist()
    ng, npd, hit = int(g.sum()), int(b.sum()), int((g & b).sum())
    dens = ng / len(g)
    prec = hit / npd if npd else float("nan")
    return {"score": score_dialogues([y], [p])["score"],
            "floor": score_dialogues([y], [[0] * len(y)])["score"],
            "bor": npd / ng if ng else float("nan"),
            "lift": prec / dens if (npd and dens > 0) else float("nan"),
            "auc": float(roc_auc_score(g, s)) if 0 < ng < len(g) else float("nan"),
            "gold_rate": dens, "n_pred": npd, "n_gold": ng}


# ── 점수 추출 ────────────────────────────────────────────────────────────────
def raw_scores(device: str) -> dict[str, dict[str, list[np.ndarray]]]:
    """모델별·문서별 **원 점수열**. 모델 구성은 채택 평가 스크립트를 그대로 재사용한다.

    `eval_all_models_alldocs` 의 `metrics` 를 원 배열 반환으로 임시 치환한다 — 모델 생성
    인자(수십 개)를 복제하면 원본과 어긋날 위험이 있어, 구성 코드는 손대지 않고 재사용한다.
    """
    import eval_all_models_alldocs as E
    import glob

    E.metrics = lambda g, s, rate: {"g": np.asarray(g, np.int8),
                                    "s": np.asarray(s, float)}
    import train_anchor_m4m5 as T
    corpora = {"dev-3": T.Corpus("dev3"), "test3": T.Corpus("test3")}
    EXP = REPO / "outputs/experiments/m4m5_phase1_anchor"
    M8D = REPO / "outputs/experiments/2026-07-31_m8_direct_boundary__personkey_i2_e100"
    models = {
        "M5": (sorted(glob.glob(str(EXP / "M5_phase1__personkey_i2_per_slot_gsauthor_ep100_i2e100_seed*.pt"))), E.m5_scores),
        "M4": (sorted(glob.glob(str(EXP / "M4_phase1__personkey_i2_per_slot_gsauthor_ep100_i2e100_seed*.pt"))), E.m5_scores),
        "M8": (sorted(glob.glob(str(M8D / "M8_pair_seed*.pt"))), E.m8_scores),
    }
    out: dict[str, dict[str, list]] = {}
    for name, (paths, fn) in models.items():
        if not paths:
            raise RuntimeError(f"{name}: 체크포인트 없음")
        acc: dict[str, list] = {}
        for p in paths:
            for k, v in fn(p, corpora, device, RATE).items():
                acc.setdefault(k, []).append(v)
        out[name] = acc
        print(f"  {name}: seed {len(paths)} · 문서 {len(acc)}")
    return out


def token_lengths(tokenizer: str) -> dict[str, np.ndarray]:
    from transformers import AutoTokenizer
    from baseline_compare import load_dataset
    tok = AutoTokenizer.from_pretrained(tokenizer)
    tl = {}
    for ds, tag in (("dev3", "dev-3"), ("test3", "test3")):
        for doc, sents, _gold in load_dataset(ds):
            tl[f"{tag}/{doc}"] = np.array(
                [len(tok.encode(x, add_special_tokens=False)) for x in sents])
    return tl


def validate_emllm(tok_len: dict[str, np.ndarray]) -> dict:
    """원본 surprise 에 이 구현을 적용해 기록된 EM-LLM-SM dev-3 Score(0.2699)를 재현한다."""
    from baseline_compare import load_dataset
    CACHE = REPO / "outputs/experiments/2026-07-31_m9_attn_entity/cache"
    ys, ps, per = [], [], {}
    for doc, sents, gold in load_dataset("dev3"):
        f = CACHE / f"{doc}__emllm.npz"
        if not f.exists():
            return {"ran": False, "why": f"캐시 없음 {doc}"}
        sur = np.asarray(np.load(f, allow_pickle=True)["surprise"], float)
        m = min(len(sents), len(sur))
        tl = tok_len[f"dev-3/{doc}"][:m]
        # 원본 규약: surprise 는 문장 단위이고 sentinel 이 없다 → s_sent = sur 자체.
        n = len(sur[:m])
        cum = np.concatenate([[0], np.cumsum(tl)])
        vec = np.zeros(len(gold), np.int8)
        for i in range(1, n):
            st = int(np.searchsorted(cum, max(0, cum[i] - N_LOCAL), side="right") - 1)
            past = sur[st:i]
            if len(past) < WARMUP:
                continue
            if sur[i] > past.mean() + GAMMA * past.std() and 1 <= i <= len(vec):
                vec[i - 1] = 1
        ys.append(np.concatenate([np.asarray(gold, np.int8), [0]]).tolist())
        ps.append(np.concatenate([vec, [0]]).tolist())
        per[doc] = {"n_pred": int(vec.sum()), "n_gold": int(np.sum(gold))}
    sc = score_dialogues(ys, ps)
    return {"ran": True, "score": sc["score"], "expected": 0.2699, "per_doc": per}


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-08-04_threshold_sensitivity")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    a = ap.parse_args()

    out = REPO / "outputs/experiments" / a.name
    out.mkdir(parents=True, exist_ok=True)

    print("■ 토큰 길이")
    tl = token_lengths(a.tokenizer)
    print("■ EM-LLM 창 구현 검증 (원본 surprise 로 기록값 재현)")
    val = validate_emllm(tl)
    print(f"   {val}")
    print("■ 점수 추출")
    raw = raw_scores(a.device)

    res: dict = {"config": {"gamma": GAMMA, "n_local": N_LOCAL, "warmup": WARMUP,
                            "rate": RATE, "ops": OPS},
                 "validate_emllm": val, "rows": {}}
    for op in OPS:
        res["rows"][op] = {}
        for mdl, docs in raw.items():
            res["rows"][op][mdl] = {}
            for doc, seeds in docs.items():
                per_seed = [metrics(d["g"], d["s"], apply_op(op, d["s"], tl.get(doc)))
                            for d in seeds]
                res["rows"][op][mdl][doc] = {
                    k: float(np.mean([p[k] for p in per_seed]))
                    for k in ("score", "bor", "lift", "auc", "floor", "n_pred", "n_gold")}
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"■ 저장 {out/'results.json'}")

    # 콘솔 요약
    docs = list(next(iter(raw.values())).keys())
    for m in ("score", "bor"):
        print(f"\n=== {m.upper()} (4 seed 평균)")
        print(f"{'작동점':<26}{'모델':<5}" + "".join(f"{d.split('/')[1][:9]:>10}" for d in docs) + f"{'평균':>10}")
        for op in OPS:
            for mdl in raw:
                v = [res["rows"][op][mdl][d][m] for d in docs]
                print(f"{op:<26}{mdl:<5}" + "".join(f"{x:10.3f}" for x in v)
                      + f"{np.mean(v):10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
