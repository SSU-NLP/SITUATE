#!/usr/bin/env python3
"""각 온라인 베이스라인의 **실제 lookahead 등급**을 6편(dev3+test3)에서 측정한다.

## 왜 필요한가
D-03 에서 `TextTiling` 행이 실은 bounded 였던 사건이 있었다. 이름만 보고 "온라인" 이라
가정하면 틀린다. 표에 `?` 로 남은 행들(GraphSeg · GreedySeg · CSM · RoBERTa ·
Membox-TopicLoom · EventRecall · SeCom)의 등급을 **논문 표 확정 전에** 못 박아야 한다.

## 측정 방법 — 방출 지연(emission delay)
push 기반 분절기는 구조적으로 미래를 볼 수 없다(아직 push 되지 않았으니까). 그러므로
`verify_causality.py` 의 미래-교란 검사는 이들에게 자명하게 통과한다 — 쓸모가 없다.
실제 쟁점은 **지연**이다:

    push(u_t) 가 경계 b 를 방출  ⇒  전이 b-1 의 판정에 u_b..u_t 를 썼다
    delay(b) = t - b

- `delay == 0` 전 구간 → **zero-lookahead** (경계 문장이 도착한 즉시 확정)
- `max delay == d > 0` → **bounded-lookahead d**
- `flush()` 로만 나오는 경계 → 문서 끝을 본 뒤 확정 = offline 꼬리

🔴 `baseline_compare.preds_baseline` 은 `flush()` 를 호출하지 않는다. 여기서도 부르지 않아
채점 경로와 조건을 맞춘다. 다만 flush 로 몇 개가 더 나왔을지는 참고로 함께 보고한다.

## 하이퍼
`baseline_compare.FROZEN["narrative"]` 동결값을 그대로 쓴다(재탐색 금지). CSM 체크포인트도
같은 선택 프로토콜(`csm_ckpt(None)` = 마지막)을 쓴다.

    env -C /tmp CUDA_VISIBLE_DEVICES=0 /root/Hi-OnTop/.venv/bin/python \\
        /root/Hi-OnTop/EACL/methods/common/audit_lookahead.py

🔴 중립 cwd 에서 실행할 것 — 이 환경의 `nltk` 는 `/root/Hi-OnTop` 을 cwd 로 두면
`regex` import 를 차단한다(`nltk/inisec.py`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))

OUT = REPO / "outputs/experiments/2026-08-04_online_baselines_close"


def streaming_factories(device: str):
    """push 기반 분절기 팩토리 — 동결 HP 는 baseline_compare 에서 그대로 가져온다."""
    from baseline_compare import FROZEN, CSM_BACKBONE, csm_ckpt  # noqa: PLC0415

    hp = FROZEN["narrative"]

    def _tt():
        from hi_ontop.baselines.texttiling_streaming import StreamingTextTiling as C
        return C(**hp["TextTiling"])

    def _gs():
        from hi_ontop.baselines.graphseg_window import GraphSegWindowD as C
        # 🔴 `glove_path` 기본값이 **상대경로**(benchmarks/glove/...)라 중립 cwd 에서 죽는다.
        # nltk 트랩 때문에 /root/Hi-OnTop 을 cwd 로 쓸 수 없으므로 절대경로로 넘긴다.
        return C(glove_path=str(REPO / "benchmarks/glove/glove.6B.300d.txt"), **hp["GraphSeg"])

    def _gd():
        from hi_ontop.baselines.greedyseg_delay2 import GreedySegOnlineDelay2 as C
        return C(device=device, **hp["GreedySeg"])

    def _csm():
        from hi_ontop.baselines.csm_online import CSMOnlineDelay2 as C
        return C(device=device, backbone=CSM_BACKBONE, ckpt_path=csm_ckpt(None), **hp["CSM"])

    return {
        "TextTiling-online-stream": _tt,
        "GraphSeg": _gs,
        "GreedySeg": _gd,
        "CSM": _csm,
    }


def audit_streaming(name, factory, data) -> dict:
    """문서별로 push 를 돌며 (방출 스텝 t, 경계 b) 를 모아 지연 분포를 낸다."""
    delays: list[int] = []
    per_doc = {}
    n_flush = 0
    t0 = time.perf_counter()
    for did, sents, _gold in data:
        seg = factory()
        d_doc: list[int] = []
        for t, u in enumerate(sents, start=1):
            for b in seg.push(u):
                d_doc.append(t - int(b))
        # flush 는 채점 경로에서 쓰이지 않는다 — 참고 수치로만 센다
        try:
            n_flush += len(seg.flush() or [])
        except Exception:  # noqa: BLE001
            pass
        delays += d_doc
        per_doc[did] = {"n_emit": len(d_doc),
                        "max_delay": max(d_doc) if d_doc else None,
                        "mean_delay": (sum(d_doc) / len(d_doc)) if d_doc else None}
    mx = max(delays) if delays else None
    grade = ("online(zero)" if mx == 0 else
             f"online(bounded d={mx})" if mx is not None else "판정불가(경계 0개)")
    return {"method": name, "kind": "streaming-push", "n_emit": len(delays),
            "max_delay": mx, "grade": grade, "n_flush_extra": n_flush,
            "per_doc": per_doc, "sec": round(time.perf_counter() - t0, 1)}


def audit_roberta(data) -> dict:
    """RoBERTa 는 push API 가 아니다 — 윈도우 상한을 소스에서 직접 감사한다.

    `methods/RoBERTa/online/segment.py::predict_online` 본문::

        for t in range(1, n):          # 경계 (t-1, t), turn t 에서 결정
            ws = max(0, t - W + 1)     # causal 윈도우 시작
            wu = utts[ws:t + 1]        # u_{ws..t} — 미래(>t) 없음

    슬라이스 상한이 `t+1` 이므로 윈도우가 담는 최대 인덱스는 정확히 `t` 다. 전이 `t-1` 의
    판정에 `u_{>t}` 가 들어가지 않는다 ⇒ delay 0. 소스 문자열을 실제로 읽어 이 상한이
    유지되고 있는지 확인한다(리팩터링으로 조용히 바뀌는 것을 잡기 위해).
    """
    src = (REPO / "methods/RoBERTa/online/segment.py").read_text(encoding="utf-8")
    ok_lo = "ws = max(0, t - W + 1)" in src
    ok_hi = "wu = utts[ws:t + 1]" in src
    n_tr = sum(len(g) for _d, _s, g in data)
    return {"method": "RoBERTa", "kind": "sliding-window(causal)", "window": 20,
            "n_transitions_covered": n_tr,
            "src_check": {"causal_start": ok_lo, "upper_bound_is_t": ok_hi},
            "grade": "online(zero)" if (ok_lo and ok_hi) else "🔴 소스가 바뀜 — 재확인 필요",
            "note": "윈도우 슬라이스 상한 t+1 ⇒ 최대 인덱스 t. 전이 t-1 판정에 미래 미사용"}


# LLM 프롬프팅 계열 — 코드 감사로 확정 (호출 비용 0)
LLM_AUDIT = {
    "Membox-TopicLoom": {
        "grade": "online(zero)",
        "evidence": "baseline_compare.py preds_membox.run — 문장 i 에서 ask(cur[-2:], s) 로 "
                    "전이 i-1 을 판정. 창 = 직전 2문장 + 현재. u_{>i} 미사용.",
    },
    "B3event-online-r10_gpt-4.1": {
        "grade": "online(zero)",
        "evidence": "llm_baseline_binary.py build(sents, cur=j+1, past=0) → range(lo, cur+1) "
                    "= 문장 0..j+1. 전이 j 판정에 도착 문장 j+1 까지만 사용. "
                    "docstring L9 'full-past = 미래 미사용 → 온라인 peer'.",
    },
    "SeCom-binary_gpt-4o": {
        "grade": "online(zero)",
        "evidence": "같은 러너(llm_baseline_binary.py), --stateful 없음. 창 규약 동일.",
    },
    "SeCom-binary-stateful_gpt-4o": {
        "grade": "online(zero)",
        "evidence": "같은 러너 + build_stateful(range(lo, cur)) — 자기 과거 경계만 상태로 "
                    "전달. 미래 미사용.",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dev3,test3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default="", help="쉼표구분 방법명 (기본 전부)")
    args = ap.parse_args()

    from baseline_compare import load_dataset  # noqa: PLC0415

    data = []
    for ds in args.datasets.split(","):
        data += load_dataset(ds)
    print(f"문서 {len(data)}편 · 전이 {sum(len(g) for _d, _s, g in data)}\n", flush=True)

    only = {x for x in args.only.split(",") if x}
    rows = []
    for name, fac in streaming_factories(args.device).items():
        if only and name not in only:
            continue
        r = audit_streaming(name, fac, data)
        rows.append(r)
        print(f"  {r['method']:<26} {r['grade']:<24} 방출 {r['n_emit']:>4} "
              f"· max delay {r['max_delay']} · {r['sec']}s", flush=True)

    if not only or "RoBERTa" in only:
        r = audit_roberta(data)
        rows.append(r)
        print(f"  {r['method']:<26} {r['grade']:<24} {r['note']}", flush=True)

    for name, info in LLM_AUDIT.items():
        if only and name not in only:
            continue
        rows.append({"method": name, "kind": "llm-prompting", **info})
        print(f"  {name:<26} {info['grade']:<24} (코드 감사)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "lookahead_audit.json"
    p.write_text(json.dumps({"config": vars(args), "rows": rows},
                            ensure_ascii=False, indent=2))
    print(f"\n→ {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
