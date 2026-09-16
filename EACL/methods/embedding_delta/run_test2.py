#!/usr/bin/env python3
"""Embedding Delta (Schneider et al. 2021) 를 STSS-Test-2 에서 **원 코드 그대로** 돌린다.

## 무엇을 하는가
저자 레포의 추론 함수 `add_scenes_to_json` 을 **한 줄도 고치지 않고** 호출한다
(`_upstream/embedding-delta/testing/stss-2021-eval/code/mycode.py`). 학습된 SVM
(`code/svm.pickle`, 저작권 학습셋으로 저자가 학습한 것)도 동봉본을 그대로 쓴다.
우리가 만드는 것은 입력 JSON 조립과 채점뿐이다. [[baselines-stay-as-published]]

## 방법 요약 (원 논문)
spaCy `de_core_news_lg` 토큰 벡터 → MiniBatchKMeans(k=1000) 로 코드북 →
문장을 클러스터 히스토그램으로 표현 → 좌/우 창(각 25문장)의 **정규화 코사인 거리**
= embedding delta 신호 → 이동평균(10) 평활 → `argrelmax(order=20)` 로 피크 =
장면 경계 → 각 구간의 4개 자질(문장 수·평균 문장길이·표준편차·구간 길이)로
SVM 이 Scene/Nonscene 판정.

## 🔴 재현 주의 3가지
1. **비결정적이다.** `MiniBatchKMeans` 에 `random_state` 가 없다. 실행마다 코드북이
   달라져 경계가 흔들린다. 그래서 `--seeds` 로 여러 번 돌려 분포를 보고한다.
2. **spaCy 버전이 다르다.** 저자 `requirements.txt` 는 버전을 고정하지 않았다.
   우리는 `de_core_news_lg==3.8.0`. 2021 당시는 3.0~3.1 계열이었을 것이다.
3. **SVM 피클이 sklearn 0.24.2 산(産)이고 우리는 1.9.0** 이다. 언피클은 되지만
   경고가 뜬다. 예측이 퇴화(전부 한 클래스)하지 않는지 확인해 로그에 남긴다.

## 비교 대상 (`common/verify_grid.py` 가 만든 기준점)
발표값(F1 0.04)은 2021 gold·문자 오프셋 조건이고 우리 예측은 2025 문장 격자에 찍힌다.
직접 비교가 성립하지 않으므로 **저자의 공식 제출 예측을 같은 격자에 올린 값**과 비교한다.

    기준점 C (2025 gold · 문장 격자) : P 0.0509 / R 0.0273 / F1 0.0355  ← 주 비교 대상
    기준점 B (2021 gold · 문장 격자) : P 0.0815 / R 0.0435 / F1 0.0567

    .venv-stss/bin/python EACL/methods/embedding_delta/run_test2.py --seeds 3
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

METHODS = Path(__file__).resolve().parents[1]
UPSTREAM_ED = METHODS / "_upstream/embedding-delta/testing/stss-2021-eval"
sys.path.insert(0, str(METHODS))
sys.path.insert(0, str(UPSTREAM_ED))

from common.stss_data import DUMP_TO_DOC, load_doc  # noqa: E402
from common.stss_eval import Boundaries, track_prf  # noqa: E402

OUT_DIR = METHODS / "embedding_delta/results"
REF = json.loads((METHODS / "common/test2_reference_points.json").read_text())


def to_boundaries(pairs, n_sent):
    return Boundaries([i for i, _ in pairs], [t for _, t in pairs], text_length=n_sent + 1)


def predict_doc(doc, svc):
    """저자 코드 `add_scenes_to_json` 을 그대로 호출하고 경계를 문장 인덱스로 환산."""
    from code.mycode import add_scenes_to_json  # noqa: PLC0415  (upstream 모듈)

    payload = {
        "text": doc.text,
        "sentences": [{"begin": a, "end": b} for a, b in zip(doc.sent_starts, doc.sent_ends)],
        # 저자 코드는 예측 전에 `scenes` 를 통째로 덮어쓴다. 자질 계산에만 쓰이므로
        # 여기 넣는 값은 결과에 영향을 주지 않는다(누출 없음).
        "scenes": [{"begin": 0, "end": len(doc.text), "type": "Scene"}],
    }
    out = add_scenes_to_json(payload, svc)
    scenes = out["scenes"]
    boundaries = []
    for i in range(1, len(scenes)):
        boundaries.append((doc.sentence_index(int(scenes[i]["begin"])),
                           f"{scenes[i-1]['type']}-to-{scenes[i]['type']}"))
    types = [s["type"] for s in scenes]
    return boundaries, types


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3,
                    help="MiniBatchKMeans 가 비결정적이라 반복 실행 횟수를 받는다")
    args = ap.parse_args()

    import pickle

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svc = pickle.load((UPSTREAM_ED / "code/svm.pickle").open("rb"))

    docs = [load_doc(n) for n in DUMP_TO_DOC.values()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []

    for run in range(args.seeds):
        print(f"\n########## 실행 {run + 1}/{args.seeds}")
        preds, pairs2025, pairs2021 = {}, [], []
        for doc in docs:
            print(f"  -- {doc.name} (문장 {len(doc.sent_starts)})")
            b, types = predict_doc(doc, svc)
            n_scene = sum(t == "Scene" for t in types)
            print(f"     예측 경계 {len(b)}개 · 구간 타입 Scene {n_scene}/{len(types)}")
            preds[doc.name] = {"boundaries": b, "n_scene": n_scene, "n_segments": len(types)}
            n = len(doc.sent_starts)
            pairs2025.append((to_boundaries(doc.gold2025, n), to_boundaries(b, n)))
            pairs2021.append((to_boundaries(doc.gold2021, n), to_boundaries(b, n)))

        s25 = track_prf(pairs2025)
        s21 = track_prf(pairs2021)
        print(f"  → 2025 gold · 문장격자 : P {s25[0]:.4f}  R {s25[1]:.4f}  F1 {s25[2]:.4f}")
        print(f"  → 2021 gold · 문장격자 : P {s21[0]:.4f}  R {s21[1]:.4f}  F1 {s21[2]:.4f}")
        runs.append({"run": run, "gold2025": s25, "gold2021": s21, "preds": preds})

    import statistics as st

    print("\n========== 요약 ==========")
    for key, ref_key, label in (("gold2025", "2025", "기준점 C (2025 gold)"),
                                ("gold2021", "2021", "기준점 B (2021 gold)")):
        f1s = [r[key][2] for r in runs]
        ref = REF["reference_points"][ref_key]["f1"]
        mean = st.mean(f1s)
        sd = st.stdev(f1s) if len(f1s) > 1 else 0.0
        print(f"  {label}")
        print(f"    우리 재실행 F1 : {mean:.4f} ± {sd:.4f}  (n={len(f1s)}, 값 {[f'{x:.4f}' for x in f1s]})")
        print(f"    저자 제출 F1   : {ref:.4f}")
        print(f"    차이           : {mean - ref:+.4f}")

    (OUT_DIR / "test2_runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2))
    print(f"\n  저장: {OUT_DIR / 'test2_runs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
