#!/usr/bin/env python3
"""B1-BayesSurprise 재현 2단계 — Kumar et al. (2023) 의 trailing-window 회귀를 재현한다.

## 대조할 발표 수치 (논문 Results, drift = 0.05)
| 창 크기 | 20 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|
| **transient Bayesian surprise** | 0.07 | 0.07 | 0.10 | 0.11 | 0.11 |
| surprisal | 0.01 | −0.06 | −0.04 | −0.03 | — |
| entropy | −0.01 | −0.05 | 0.01 | 0.01 | — |

지표 = 남겨둔 이야기(held-out story)에서의 **Pearson r**. 논문 핵심 주장은 절대값이 아니라
**"Bayesian surprise 만 양의 상관, surprisal·entropy 는 아니다"** 는 대조다. 우리 판정은
① 각 창의 r 이 발표값 근처인가 ② 세 신호의 **순서·부호가 재현되는가** 둘 다로 한다.

## 저자 절차 (`analysis/regression_window_size_on_events.py` + `surprise_helper.py`)
1. 신호 = **transient 정규화된 KL** — Reynolds et al. (2007) 방식:
       avg[0] = 전체 평균 ; avg[i] = avg[i-1] + drift·(x[i] − avg[i-1])
       avg ← roll(avg, 1) ; avg[0] = avg[1]      # t−1 시점 평균으로 나누기 위함
       transient = x / avg
2. `build_signal_mat(signal, nfeat)` — 각 시점의 **직전 nfeat 개** 값을 자질로. 앞 nfeat 행은 버린다.
3. 타깃 = `*_button_density.csv` 를 단어 onset 시각으로 샘플링한 뒤 **최댓값으로 나눔**.
   onset 스케일이 이야기마다 다르다: monkey = onset/512×1000, tunnel = round(onset×10),
   pieman = round(onset×1000). (monkey 만 512Hz 샘플 인덱스라 초로 바꿔야 한다.)
4. 세 이야기를 이어붙이고 `PredefinedSplit(story_id)` 로 **leave-one-story-out**,
   안쪽에서 `GridSearchCV` 로 Lasso alpha 를 `logspace(-16, -5, 10)` 에서 고른다
   (max_iter=5000, tol=1e-2). 남겨둔 이야기의 예측과 실제 버튼 반응의 Pearson r 을 기록.

## 🔴 논문에 안 적혀 판단이 필요했던 것
저자 코드의 `pad_df`(앞뒤 200개 패딩)는 `create_analyses_df.py` 경로에만 있고, 회귀
스크립트는 datum 에서 onset 을 다시 붙인다. 패딩이 회귀 입력에 들어갔는지 코드만으로는
확정되지 않아 **패딩 없이** 돌린다. 이 선택은 결과에 명시한다.

    uv run python EACL/methods/bayes_surprise/reproduce_kumar_regression.py --drift 5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SIGNALS = HERE / "signals"
BUTTON = DATA / "button_press_proportion"
RESULTS = DATA / "extract-embeddings-data/results"

STORIES = ("monkey", "pieman", "tunnel")   # 저자 코드의 순서 그대로
WINDOWS = (20, 50, 100, 150, 200)
PAPER = {   # 논문 Results, drift 0.05
    "kl":        {20: 0.07, 50: 0.07, 100: 0.10, 150: 0.11, 200: 0.11},
    "surprisal": {20: 0.01, 50: -0.06, 100: -0.04, 150: -0.03},
    "entropy":   {20: -0.01, 50: -0.05, 100: 0.01, 150: 0.01},
}


def read_datum_onsets(story: str) -> np.ndarray:
    """datum.csv 의 onset 열. 토큰에 쉼표가 든 행이 있어 pandas 로는 못 읽는다.

    (예: tunnel 17행의 `predicted_word` 가 `,` 하나다.) 앞 열들의 위치는 고정이므로
    csv 모듈로 읽어 인덱스로 뽑는다 — 데이터 행은 절대 버리지 않는다(정렬이 깨진다).

    🔴 정렬 규약: 저자는 `df_entr_sur['onsets'] = datum_data['onset']` 로 **행 순서 그대로**
    붙인다(pandas 인덱스 정렬). surp_entr 의 `Unnamed: 0` 열은 다른 프레임의 인덱스라
    (pieman 은 최댓값이 1281 인데 datum 행은 854개뿐) **정렬에 쓰면 안 된다.**
    빈 줄만 제거해 pandas 의 `skip_blank_lines` 기본 동작을 맞춘 뒤 위에서부터 자른다.
    """
    path = RESULTS / story / "datum.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r]   # 빈 줄 제거 = pandas 의 skip_blank_lines 기본값
    header = rows[0]
    col = header.index("onset")
    return np.array([float(r[col]) for r in rows[1:]])


def onset_to_index(story: str, onsets: np.ndarray) -> np.ndarray:
    """저자 `get_onset_time` 이식 — 이야기마다 스케일이 다르다."""
    if story == "monkey":
        return (onsets / 512 * 1000).astype(int)     # 512Hz 샘플 → ms
    if story == "tunnel":
        return np.round(onsets * 10).astype(int)     # 초 → 0.1초 격자
    if story == "pieman":
        return np.round(onsets * 1000).astype(int)   # 초 → ms
    raise ValueError(story)


def read_button_density(story: str) -> np.ndarray:
    """`*_button_density.csv` 의 값 열(저자 코드는 `[:,1]` 을 쓴다)."""
    with (BUTTON / f"{story}_button_density.csv").open(newline="") as f:
        rows = list(csv.reader(f))
    return np.array([float(r[1]) for r in rows[1:]])


def reynolds_transient(x: np.ndarray, drift: float) -> np.ndarray:
    """저자 `average_surp_reynolds` 이식 (t−1 평균으로 나누기 위한 roll 포함)."""
    avg = np.empty(len(x))
    avg[0] = np.nanmean(x)
    for i in range(1, len(x)):
        avg[i] = avg[i - 1] + drift * (x[i] - avg[i - 1])
    avg = np.roll(avg, 1)
    avg[0] = avg[1]
    return x / avg


def build_signal_mat(signal: np.ndarray, nfeat: int) -> np.ndarray:
    """저자 `build_signal_mat` 이식 — 시점 i 의 자질은 직전 nfeat 개 값."""
    mat = np.empty((len(signal), nfeat))
    for i in range(nfeat, len(signal)):
        mat[i] = signal[i - nfeat:i]
    return mat


def run(signal_name: str, drift: float, verbose: bool = True) -> dict[int, float]:
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GridSearchCV, PredefinedSplit

    per_window = {}
    for nfeat in WINDOWS:
        X_parts, y_parts, sid_parts = [], [], []
        for i, story in enumerate(STORIES):
            npz = np.load(SIGNALS / f"{story}_signals.npz", allow_pickle=True)
            raw = npz[signal_name].astype(float)
            signal = reynolds_transient(raw, drift) if signal_name == "kl" else raw

            onsets = read_datum_onsets(story)
            n = min(len(signal), len(onsets))
            signal, onsets = signal[:n], onsets[:n]

            finite = np.isfinite(signal)
            num_nan = int((~finite).sum())
            signal = signal[finite]
            onsets = onsets[finite]

            mat = build_signal_mat(signal, nfeat)[nfeat:]
            idx = onset_to_index(story, onsets)[nfeat:]

            button = read_button_density(story)
            idx = np.clip(idx, 0, len(button) - 1)
            y = button[idx]
            y = y / y.max()

            m = min(len(mat), len(y))
            X_parts.append(mat[:m])
            y_parts.append(y[:m])
            sid_parts.append(np.full(m, i))
            if verbose and nfeat == WINDOWS[0]:
                print(f"    {story}: 단어 {n} · NaN {num_nan} · 회귀표본 {m}")

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        sid = np.concatenate(sid_parts)

        alphas = np.logspace(-16, -5, num=10, endpoint=True)
        outer = PredefinedSplit(sid)
        rs = []
        for tr, te in outer.split():
            search = GridSearchCV(Lasso(max_iter=5000, tol=1e-2),
                                  param_grid={"alpha": alphas}, n_jobs=4,
                                  cv=PredefinedSplit(sid[tr]))
            search.fit(X[tr], y[tr])
            pred = search.best_estimator_.predict(X[te])
            rs.append(float(np.corrcoef(y[te], pred)[0, 1]))
        per_window[nfeat] = float(np.mean(rs))
        if verbose:
            print(f"  창 {nfeat:>3}: r = {per_window[nfeat]:+.4f} "
                  f"(이야기별 {[f'{v:+.3f}' for v in rs]})")
    return per_window


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", type=int, default=5, help="저자 인자와 동일: 퍼센트 정수(5 = 0.05)")
    ap.add_argument("--signals", nargs="+", default=["kl", "surprisal", "entropy"])
    args = ap.parse_args()
    drift = args.drift / 100

    missing = [s for s in STORIES if not (SIGNALS / f"{s}_signals.npz").exists()]
    if missing:
        print(f"🔴 신호 파일 없음: {missing} — compute_kl.py 를 먼저 돌려야 한다")
        return 1

    out = {}
    for name in args.signals:
        print(f"\n### {name} (drift={drift})")
        out[name] = run(name, drift)

    print("\n========== 발표 수치 대조 ==========")
    print(f"{'신호':<12}{'창':>6}{'우리 r':>10}{'논문 r':>10}{'차이':>10}")
    for name, res in out.items():
        for w, r in res.items():
            p = PAPER.get(name, {}).get(w)
            ps = f"{p:+.2f}" if p is not None else "  —"
            ds = f"{r - p:+.3f}" if p is not None else "   —"
            print(f"{name:<12}{w:>6}{r:>+10.4f}{ps:>10}{ds:>10}")

    print("\n부호·순서 재현 검사 (논문의 핵심 주장)")
    for w in WINDOWS:
        vals = {n: out[n][w] for n in out if w in out[n]}
        if "kl" in vals and len(vals) > 1:
            ok = all(vals["kl"] > v for n, v in vals.items() if n != "kl")
            print(f"  창 {w:>3}: KL({vals['kl']:+.3f}) > "
                  + ", ".join(f"{n}({v:+.3f})" for n, v in vals.items() if n != "kl")
                  + f"  {'✅' if ok else '🔴'}")

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / f"results/kumar_regression_drift{args.drift}.json").write_text(
        json.dumps({"drift": drift, "ours": out, "paper": PAPER}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
