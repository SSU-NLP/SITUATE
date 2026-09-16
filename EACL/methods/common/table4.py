#!/usr/bin/env python3
"""서사 6편 · **4지표** 통합표 (Pk · WindowDiff · F1 · Score).

평가셋 = dev3(pieman·tunnel·monkey) + test3(run·gohitler·mymotherslife).
채점 = `hi_ontop.dts_scoring.score_dialogues`, **deployable**(경계 수 자가결정).

기존 `preds_table.py` 는 6지표(+BOR·purity·coverage·F(p,c)·L1)를 내므로 건드리지 않고,
논문용 4지표만 뽑는 별도 러너를 둔다.

**온라인 등급을 이름에서 읽는다**: `-online`(zero-lookahead) · `-online-stream`/`-online-d{d}`(bounded).
그 표기가 없는 행은 오프라인이거나 등급 미확인이다.

    .venv/bin/python EACL/methods/common/table4.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))

PRED = REPO / "outputs/experiments/2026-07-29_baseline_compare/preds"
OUT = REPO / "outputs/experiments/2026-08-03_narrative_online"

# 온라인 등급 주석. 미기재 = 미확인(표에 '?' 로 나온다)
# 🔒 **방법당 원본 / online 두 행만** 남긴다 (사용자 지시 2026-08-04).
# 여러 변종(체크포인트·모델·프롬프트 계열)은 대표 하나로 접고, 나머지는 부록으로 뺀다.
# `-online` = **zero-lookahead**(미래 0). bounded 판은 `-online-bounded` 로 부록에만 둔다.
MAIN = {
    # 우리 방법
    "M5": "Hi-OnTop (online)",
    # NES 베이스라인 — 원본 / online 쌍
    "TextTiling": "TextTiling",
    "TextTiling-online": "TextTiling (online)",
    "EM-LLM-SM": "EM-LLM (online)",
    "EmbeddingDelta-online-d25": "Embedding Delta (online, d=25)",
    "EmbeddingDelta-online-zero": "Embedding Delta (online, zero)",
    "B3event-online-r10_gpt-4.1": "EventRecall (online)",
    "BayesSurprise": "BayesSurprise (online)",
    "RoBERTa": "RoBERTa (online)",
    "CSM": "CSM (online)",
    "GraphSeg": "GraphSeg (online)",
    "GreedySeg": "GreedySeg (online)",
    "Membox-TopicLoom": "Membox-TopicLoom (online)",
    # DTS 무대 방법을 서사에 zero-shot — 대칭 실험(appendix)
    "DialSTART-online-d2": "DialSTART (online, DTS→NES)",
    # 하한
    "FixedInterval": "Fixed interval (density-matched)",
    "RandomRate": "Random (density-matched)",
}
# 부록으로 빼는 변종 (표에서 제외하되 파일은 남긴다)
APPENDIX = {
    "TextTiling-online-bounded", "M5-adaptive", "M4", "M4-adaptive",
    "M1-CosDrift", "M1-CosDrift-adaptive", "CSM-devbest", "CSM-offverified",
    "B3event-online-r10_gpt-5", "B3event-online-r10_gpt-oss-120b",
    "B3event-binary-stateful_gpt-5", "SeCom-binary_gpt-4o",
    "SeCom-binary-stateful_gpt-4o", "SeCom-binary-stateful_gpt-5-mini",
    # D-07 재계산 변종 — 주 표는 Qwen 판 하나만 두고 LM 대조는 부록으로
    "EM-LLM-SM-llama31base", "EM-LLM-SM-llama31i",
}
# 🔒 D-10 (2026-08-04) — '?' 로 남아 있던 8행의 등급을 **실측**으로 확정했다.
# 산출 = `EACL/methods/common/audit_lookahead.py` →
#        `outputs/experiments/2026-08-04_online_baselines_close/lookahead_audit.json`
#
# 측정 규약: push 기반 분절기는 구조적으로 미래를 못 본다(아직 push 되지 않았으니까).
# 실제 쟁점은 **지연**이다 — `push(u_t)` 가 경계 b 를 방출하면 전이 b-1 판정에 u_b..u_t 를
# 썼다는 뜻이므로 `delay = t - b`. 6편 전체에서의 최대값을 d 로 박는다.
#
# 🔴 이름과 실제가 달랐던 2건 (D-03 의 TextTiling 사건과 같은 종류):
#   - GreedySeg: 클래스명이 `GreedySegOnlineDelay2`(delay=2 암시)인데 **실측 d=7**.
#     `_try_finalize_segment` 가 후보를 누적하다 한꺼번에 확정하기 때문이다.
#   - GraphSeg: `window_d=10` 이지만 실측 **d=6** (창이 차기 전에는 방출하지 않아 상한보다 작다).
# 그리고 `TextTiling-online-bounded` 의 d 는 **22** 다 — zero(0.2647) 대비 bounded(0.3212)
# 의 dev3 격차가 큰 이유가 이것이므로 표에 d 를 반드시 적는다.
GRADE = {
    "M5": "online(zero)",
    "EM-LLM-SM": "online(zero)",
    "TextTiling": "offline(원본)",
    "TextTiling-online": "online(zero)",
    "TextTiling-online-bounded": "online(bounded d=22)",
    "EmbeddingDelta-online-d25": "online(bounded d=25)",
    "EmbeddingDelta-online-zero": "online(zero)",
    "B3event-online-r10_gpt-4.1": "online(zero)",
    "BayesSurprise": "online(zero)",
    "FixedInterval": "online(zero)",
    "RandomRate": "online(zero)",
    "GraphSeg": "online(bounded d=6)",
    "GreedySeg": "online(bounded d=7)",
    "CSM": "online(bounded d=2)",
    "CSM-devbest": "online(bounded d=2)",
    "CSM-offverified": "online(bounded d=2)",
    "RoBERTa": "online(zero)",
    "Membox-TopicLoom": "online(zero)",
    "DialSTART-online-d2": "online(bounded d=2)",
    "SeCom-binary_gpt-4o": "online(zero)",
    "SeCom-binary-stateful_gpt-4o": "online(zero)",
    # D-07 닫힘 — surprise LM 을 바꿔도 결론 불변 (셋 다 무경계 하한 아래)
    "EM-LLM-SM-llama31base": "online(zero)",
    "EM-LLM-SM-llama31i": "online(zero)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dev3,test3")
    ap.add_argument("--all", dest="main_only", action="store_false",
                    help="부록 변종까지 전부 표시 (기본은 주 표 행만)")
    ap.set_defaults(main_only=True)
    args = ap.parse_args()

    from baseline_compare import load_dataset            # noqa: PLC0415
    from hi_ontop.dts_scoring import score_dialogues     # noqa: PLC0415

    def sc(P, G):
        ys = [np.concatenate([np.asarray(g, np.int8), [0]]).tolist() for g in G]
        ps = [np.concatenate([np.asarray(p[:len(g)], np.int8), [0]]).tolist()
              for p, g in zip(P, G)]
        s = score_dialogues(ys, ps)
        return {"pk": s["pk"], "wd": s["wd"], "f1": s["f1"], "score": s["score"]}

    allr = {}
    for ds in args.datasets.split(","):
        data = load_dataset(ds)
        docs = [d for d, _s, _g in data]
        G = [g for _d, _s, g in data]
        ngold = int(sum(int(g.sum()) for g in G))
        rows = []
        for f in sorted(PRED.glob(f"*_{ds}.json")):
            name = f.name[: -(len(ds) + 6)]
            if args.main_only and name not in MAIN:
                continue
            d = json.loads(f.read_text())
            if any(x not in d for x in docs):
                continue
            P = [np.asarray(d[x], np.int8) for x in docs]
            r = sc(P, G)
            r["method"] = name
            r["grade"] = GRADE.get(name, "?")
            r["n_pred"] = int(sum(int(np.asarray(p[:len(g)]).sum()) for p, g in zip(P, G)))
            rows.append(r)
        floors = {"무경계 하한": sc([np.zeros(len(g), np.int8) for g in G], G),
                  "전부경계 하한": sc([np.ones(len(g), np.int8) for g in G], G)}
        rows.sort(key=lambda r: -r["score"])
        allr[ds] = {"n_gold": ngold, "floors": floors, "rows": rows}

        print(f"\n[{ds}] gold {ngold} · 무경계 하한 Score {floors['무경계 하한']['score']:.4f}"
              f" · 전부경계 하한 {floors['전부경계 하한']['score']:.4f}")
        print(f"{'방법':<34}{'등급':<20}{'Pk↓':>8}{'WD↓':>8}{'F1↑':>8}{'Score↑':>9}{'예측':>7}")
        for r in rows:
            print(f"{MAIN.get(r['method'], r['method']):<34}{r['grade']:<20}{r['pk']:>8.4f}{r['wd']:>8.4f}"
                  f"{r['f1']:>8.4f}{r['score']:>9.4f}{r['n_pred']:>7}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table4_6docs.json").write_text(json.dumps(allr, ensure_ascii=False, indent=2,
                                                      default=float))
    print(f"\n저장: {OUT / 'table4_6docs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
