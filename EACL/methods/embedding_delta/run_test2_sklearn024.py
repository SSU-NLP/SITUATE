#!/usr/bin/env python3
"""Embedding Delta 재실행 — **sklearn 0.24.2 기본값 복원판**.

## 왜 이 조건이 필요한가
`run_test2.py`(현행 sklearn 1.9 기본값)로 5회 돌린 결과가 저자 공식 제출의 **절반**에서
안정적으로 나왔다(F1 0.0181±0.0034 vs 0.0355). 편차가 작으니 난수 탓이 아니라 **계통 차이**다.

용의자 1번은 라이브러리 기본값 드리프트다. 저자 `requirements.txt` 는 버전을 고정하지
않았지만 논문은 2021년이고, 같이 배포된 SVM 피클은 **sklearn 0.24.2** 산이다. 그런데
`MiniBatchKMeans` 기본값이 그 사이 바뀌었다:

    batch_size : 100 (0.24.2)  →  1024 (1.x)
    n_init     : 3   (0.24.2)  →  'auto' (1.x)

저자 코드는 `MiniBatchKMeans(n_clusters=..., max_iter=..., verbose=False)` 만 넘기고
나머지는 기본값에 맡기므로, 이 드리프트가 코드북을 통째로 바꾼다.

## 이건 튜닝이 아니다
발표 당시 기본값으로 되돌리는 것은 **원 레시피 복원**이지 하이퍼 탐색이 아니다.
([[baselines-stay-as-published]] 의 "소실된 baseline 은 원 레시피대로 재현" 조항.)
값을 바꿔가며 최적점을 찾지 않고, 0.24.2 기본값 **한 조합만** 고정해 돌린다.

산출 파일은 조건을 이름에 박는다([[no-script-edit-during-run]]): `test2_runs_sklearn024.json`

    .venv-stss/bin/python EACL/methods/embedding_delta/run_test2_sklearn024.py --seeds 5
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import warnings
from pathlib import Path

METHODS = Path(__file__).resolve().parents[1]
UPSTREAM_ED = METHODS / "_upstream/embedding-delta/testing/stss-2021-eval"
sys.path.insert(0, str(METHODS))
sys.path.insert(0, str(UPSTREAM_ED))

# 🔴 upstream 모듈을 import 하기 **전에** 패치해야 한다 (mycode 가 클래스를 직접 참조).
import sklearn.cluster as _skc  # noqa: E402

_ORIG_MBK = _skc.MiniBatchKMeans


def MiniBatchKMeans024(*args, **kw):  # noqa: N802 — 원 클래스 이름을 그대로 대체한다
    """sklearn 0.24.2 기본값을 채워 원 클래스를 생성하는 팩토리.

    서브클래스로 만들면 sklearn 이 `__init__` 시그니처에 `*args` 가 있다며 거부한다
    (estimator 규약). 그래서 클래스가 아니라 **호출 가능한 팩토리**로 갈아끼운다 —
    `mycode` 는 `MiniBatchKMeans(...)` 로 호출만 하므로 동작이 같다.
    """
    kw.setdefault("batch_size", 100)
    kw.setdefault("n_init", 3)
    return _ORIG_MBK(*args, **kw)


_skc.MiniBatchKMeans = MiniBatchKMeans024

from common.stss_data import DUMP_TO_DOC, load_doc  # noqa: E402
from common.stss_eval import Boundaries, track_prf  # noqa: E402

import code.mycode as _mycode  # noqa: E402  (upstream)

_mycode.MiniBatchKMeans = MiniBatchKMeans024

OUT_DIR = METHODS / "embedding_delta/results"
REF = json.loads((METHODS / "common/test2_reference_points.json").read_text())


def to_boundaries(pairs, n_sent):
    return Boundaries([i for i, _ in pairs], [t for _, t in pairs], text_length=n_sent + 1)


def predict_doc(doc, svc):
    payload = {
        "text": doc.text,
        "sentences": [{"begin": a, "end": b} for a, b in zip(doc.sent_starts, doc.sent_ends)],
        "scenes": [{"begin": 0, "end": len(doc.text), "type": "Scene"}],
    }
    out = _mycode.add_scenes_to_json(payload, svc)
    scenes = out["scenes"]
    boundaries = [(doc.sentence_index(int(scenes[i]["begin"])),
                   f"{scenes[i-1]['type']}-to-{scenes[i]['type']}")
                  for i in range(1, len(scenes))]
    return boundaries, [s["type"] for s in scenes]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    import pickle

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svc = pickle.load((UPSTREAM_ED / "code/svm.pickle").open("rb"))

    docs = [load_doc(n) for n in DUMP_TO_DOC.values()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []

    for run in range(args.seeds):
        print(f"\n########## 실행 {run + 1}/{args.seeds} (sklearn 0.24 기본값)")
        pairs25, pairs21, preds = [], [], {}
        for doc in docs:
            b, types = predict_doc(doc, svc)
            print(f"  {doc.name}: 예측 경계 {len(b)}개 · Scene {sum(t=='Scene' for t in types)}/{len(types)}")
            preds[doc.name] = {"n_boundaries": len(b)}
            n = len(doc.sent_starts)
            pairs25.append((to_boundaries(doc.gold2025, n), to_boundaries(b, n)))
            pairs21.append((to_boundaries(doc.gold2021, n), to_boundaries(b, n)))
        s25, s21 = track_prf(pairs25), track_prf(pairs21)
        print(f"  → 2025 gold F1 {s25[2]:.4f} · 2021 gold F1 {s21[2]:.4f}")
        runs.append({"run": run, "gold2025": s25, "gold2021": s21, "preds": preds})

    print("\n========== 요약 (sklearn 0.24 기본값 복원) ==========")
    for key, ref_key, label in (("gold2025", "2025", "기준점 C (2025 gold)"),
                                ("gold2021", "2021", "기준점 B (2021 gold)")):
        f1s = [r[key][2] for r in runs]
        ref = REF["reference_points"][ref_key]["f1"]
        mean = st.mean(f1s)
        sd = st.stdev(f1s) if len(f1s) > 1 else 0.0
        print(f"  {label}: 우리 {mean:.4f} ± {sd:.4f} · 저자 {ref:.4f} · 차이 {mean-ref:+.4f}")

    (OUT_DIR / "test2_runs_sklearn024.json").write_text(
        json.dumps(runs, ensure_ascii=False, indent=2))
    print(f"\n  저장: {OUT_DIR / 'test2_runs_sklearn024.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
