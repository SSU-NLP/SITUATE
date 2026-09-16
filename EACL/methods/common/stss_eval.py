#!/usr/bin/env python3
"""STSS 공식 채점기 재구현 (KONVENS 2021 Shared Task on Scene Segmentation).

## 왜 재구현인가
공식 `eval.py`(Zehe et al. 2021a, 세 참가 레포에 동일 md5 로 배포됨)는 gold/pred 를
**전체 텍스트를 담은 JSON 파일 쌍**으로만 받는다. STSS 코퍼스는 저작권 대상이라
우리가 가진 것은 (a) 경계 오프셋만 담긴 standoff 주석과 (b) embedding-delta 레포에
동봉된 공식 제출물 덤프뿐이다. 그래서 **입력을 경계 리스트로 받는 형태**로 옮기되
점수 정의는 원본과 한 글자도 다르지 않게 유지한다.

## 원본과 동일함을 보장하는 지점 (공식 eval.py 대조)
1. 경계 = `scenes[1:]` 의 `begin` 문자 오프셋. **첫 장면의 시작(0)은 경계가 아니다.**
2. 라벨 = `"{이전장면타입}-to-{현재장면타입}"` — Scene-to-Scene / Scene-to-Nonscene /
   Nonscene-to-Scene 3종. 타입이 다르면 **경계 위치가 맞아도 오답**이다.
3. 문자 인덱스 배열(길이 = len(text))에 라벨을 찍고 `sklearn.classification_report`
   를 `labels=[1,2,3]` 으로 호출한다. 배경(0=NOBORDER)은 점수에서 빠진다.
4. 공식 결과표(Table 1, Zehe et al. 2021a)가 쓰는 값 = **micro avg** P/R/F1.
   컨테이너 로그가 출력하는 `macro avg` 와 다르므로 혼동 금지.

## 추가로 제공하는 지표 (원본이 아님 — 반드시 구분해서 보고할 것)
- `relaxed_f1(tolerance=t)` : Zehe et al. 2025(NAACL) §6.1 의 완화 F1. 문장 인덱스
  기준으로 ±t 문장 안에 대응 경계가 있으면 정답. 2025 논문 표는 전부 t=3 이다.
- `iou()` : Schneider et al. 2021 이 자체 보고에 쓴 장면 단위 IoU.

두 지표는 **2021 공식 순위와 비교할 수 없다.** 공식 순위 = exact micro F1 뿐이다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import classification_report

BOUNDARY_TYPES = ("Scene-to-Scene", "Scene-to-Nonscene", "Nonscene-to-Scene")


@dataclass
class Boundaries:
    """한 문서의 경계 목록. offsets = 문자 오프셋, types = 전이 라벨."""

    offsets: list[int]
    types: list[str]
    text_length: int

    def __post_init__(self):
        if len(self.offsets) != len(self.types):
            raise ValueError("offsets 와 types 의 길이가 다르다")


def boundaries_from_scenes(scenes: list[dict], text_length: int) -> Boundaries:
    """공식 JSON 의 `scenes` 리스트에서 경계를 뽑는다 (공식 eval.py 와 동일 규칙).

    scenes = [{"begin": int, "end": int, "type": "Scene"|"Nonscene"}, ...] (문서 순서)
    """
    offsets, types = [], []
    prev = scenes[0]["type"]
    for sc in scenes[1:]:
        offsets.append(int(sc["begin"]))
        types.append(f"{prev}-to-{sc['type']}")
        prev = sc["type"]
    return Boundaries(offsets, types, text_length)


def exact_report(gold: Boundaries, pred: Boundaries) -> dict:
    """공식 exact 채점. 반환값은 sklearn classification_report(output_dict=True).

    공식 표의 값은 `["micro avg"]` 다. 라벨 인덱스 매핑은 공식 코드와 동일하게
    NOBORDER=0 으로 고정하고 나머지를 등장 순서로 부여한다 — labels=[1,2,3] 이
    3종 전이를 모두 덮으므로 매핑 순서는 micro avg 에 영향을 주지 않는다.
    """
    if gold.text_length != pred.text_length:
        raise ValueError("gold/pred 의 텍스트 길이가 다르다 (공식 eval.py 는 여기서 assert 실패)")
    n = gold.text_length
    # 🔴 공식 코드의 라벨 매핑을 그대로 복제한다: defaultdict 에 **pred 를 먼저, gold 를
    # 나중에** 넣으므로 정수 배정이 등장 순서에 달려 있다. 그리고 채점은 labels=[1,2,3]
    # 으로 고정돼 있어, 4번째 전이 타입(Nonscene-to-Nonscene)이 나타나면 그 경계는
    # **점수에서 통째로 빠진다.** 실제 STSS Track 2 gold 에 이 타입이 1개 있다.
    label_to_int: dict[str, int] = {"NOBORDER": 0}

    def code(t: str) -> int:
        if t not in label_to_int:
            label_to_int[t] = len(label_to_int)
        return label_to_int[t]

    y_true = np.zeros(n)
    y_pred = np.zeros(n)
    for off, typ in zip(pred.offsets, pred.types):
        y_pred[off] = code(typ)
    for off, typ in zip(gold.offsets, gold.types):
        y_true[off] = code(typ)

    int_to_label = {v: k for k, v in label_to_int.items()}
    return classification_report(
        y_true=y_true, y_pred=y_pred,
        target_names=[int_to_label.get(i, f"UNUSED_{i}") for i in (1, 2, 3)],
        labels=[1, 2, 3], output_dict=True, zero_division=0,
    )


def exact_prf(gold: Boundaries, pred: Boundaries) -> tuple[float, float, float]:
    """공식 표에 실리는 micro avg (precision, recall, f1)."""
    rep = exact_report(gold, pred)
    key = "micro avg" if "micro avg" in rep else "accuracy"
    m = rep[key]
    return m["precision"], m["recall"], m["f1-score"]


def track_prf(pairs: list[tuple[Boundaries, Boundaries]]) -> tuple[float, float, float]:
    """트랙 단위 점수 = **문서별 micro avg 를 문서 간 산술평균** 한 값.

    🔴 이름 주의. 공식 표(Zehe et al. 2021a, Table 1)의 표제는 "Micro avg." 지만,
    실제로 실린 값은 트랙 전체를 pooled 한 micro 가 아니라 **문서별 micro 의 평균**이다.
    두 방식은 문서 길이·경계 수가 다르면 갈라진다. 우리가 실측으로 확인했다
    (`embedding_delta/check_official_submission.py`, Schneider et al. 행 기준):

        집계 방식              Track 1 P/R/F1        Track 2 P/R/F1
        문서별 micro 의 평균   0.014 / 0.021 / 0.017   0.064 / 0.034 / 0.044  ← 발표값과 일치
        트랙 pooled micro      0.011 / 0.012 / 0.012   0.064 / 0.034 / 0.045
        발표값 (Table 1)       0.01  / 0.02  / 0.02    0.06  / 0.03  / 0.04

    Track 2 는 두 방식이 같은 곳으로 반올림돼 구분이 안 되지만 Track 1 은 갈라지고,
    발표값은 문서별 평균 쪽이다. 공식 `eval_folder` 도 문서별 report 를 모아 평균하는
    구조라 이 해석이 코드와도 맞는다. **재현 비교는 반드시 이 함수로 한다.**
    """
    ps, rs, fs = [], [], []
    for gold, pred in pairs:
        p, r, f = exact_prf(gold, pred)
        ps.append(p)
        rs.append(r)
        fs.append(f)
    return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(fs))


def pooled_exact_prf(pairs: list[tuple[Boundaries, Boundaries]]) -> tuple[float, float, float]:
    """트랙 전체를 pooled 한 micro avg. **공식 표와 비교하지 말 것** (`track_prf` 참조).
    진단용으로만 둔다 — 문서별 평균과 갈라지는 정도를 보여준다.
    """
    tp = fp = fn = 0
    for gold, pred in pairs:
        gpos = dict(zip(gold.offsets, gold.types))
        ppos = dict(zip(pred.offsets, pred.types))
        for o, t in ppos.items():
            if gpos.get(o) == t:
                tp += 1
            else:
                fp += 1
        for o, t in gpos.items():
            if ppos.get(o) != t:
                fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


NOBORDER = "NOBORDER"


def relaxed_f1_zehe2025(true_labels: list[str], pred_labels: list[str],
                        tolerance: int = 3) -> dict[str, float]:
    """Zehe et al. (2025) 완화 F1 — 저자 코드 `ssc/evaluation.py` **그대로 이식**.

    입력은 **문장 단위 라벨 시퀀스**다(길이 = 문서의 문장 수). 경계 문장에는 전이 타입
    문자열이, 나머지에는 `NOBORDER` 가 들어간다. 2025 논문 표는 전부 tolerance=3.

    🔴 직관과 다른 지점 3가지 — 직접 짜면 반드시 어긋난다. 그래서 이식했다.
    1. **창이 비대칭이다.** 저자 코드는 `labels[max(0,i-t) : i+t]` 를 본다. 파이썬 슬라이스라
       오른쪽 끝이 열려 있어 실제 허용 구간은 **[i-t, i+t-1]** — ±t 가 아니다.
    2. **타입까지 같아야 한다.** 창 안에 '경계가 있는가'가 아니라 '**같은 전이 라벨**이
       있는가'를 본다.
    3. **TP 와 FP 를 따로 센다.** TP/FN 은 gold 를 훑으며, FP 는 pred 를 훑으며 집계한다.
       매칭이 아니므로 한 pred 가 여러 gold 의 TP 로 세어질 수 있고, precision 의 분모
       `TP+FP` 는 예측 개수와 일치하지 않는다. 표준 F1 정의가 아니지만 **발표 수치의 정의**다.

    우리 자체 지표(`relaxed_f1_matched`)와 값이 다르다. 논문 대조에는 **이 함수만** 쓴다.
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("라벨 시퀀스 길이가 다르다")
    if tolerance < 0:
        raise ValueError("tolerance 는 0 이상이어야 한다")

    tp = fp = fn = 0
    for i, label in enumerate(true_labels):
        if label != NOBORDER:
            window = (pred_labels[max(0, i - tolerance):i + tolerance]
                      if tolerance > 0 else [pred_labels[i]])
            if label in window:
                tp += 1
            else:
                fn += 1
    for i, label in enumerate(pred_labels):
        if label != NOBORDER:
            window = (true_labels[max(0, i - tolerance):i + tolerance]
                      if tolerance > 0 else [true_labels[i]])
            if label not in window:
                fp += 1

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def iou_zehe2025(true_labels: list[str], pred_labels: list[str]) -> float:
    """Zehe et al. (2025) Table 4 의 IoU — 저자 코드 `evaluate_predictions_iou` 그대로 이식.

    경계가 아닌 문장들의 연속 구간을 장면으로 보고, gold 장면마다 아직 배정되지 않은
    pred 장면 중 겹침이 최대인 것을 골라 겹침 길이를 더한 뒤 전체 문장 수로 나눈다.
    (합집합으로 나누지 않으므로 이름과 달리 IoU 가 아니라 **겹침 비율**이다 — 원 코드 그대로.)
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("라벨 시퀀스 길이가 다르다")

    def spans(labels):
        out, start = [], None
        for i, lab in enumerate(labels):
            if lab != NOBORDER:
                if start is not None:
                    out.append((start, i - 1))
                    start = None
            elif start is None:
                start = i
        if start is not None:
            out.append((start, len(labels) - 1))
        return out

    t_spans, p_spans = spans(true_labels), spans(pred_labels)
    assigned, total = set(), 0
    for ts, te in t_spans:
        best_len, best_i = 0, None
        for pi, (ps, pe) in enumerate(p_spans):
            if pi in assigned:
                continue
            s, e = max(ts, ps), min(te, pe)
            if s <= e and (e - s + 1) > best_len:
                best_len, best_i = e - s + 1, pi
        if best_i is not None:
            assigned.add(best_i)
            total += best_len
    return total / len(true_labels) if true_labels else 0.0


def labels_from_boundaries(boundaries: list[tuple[int, str]], n_sentences: int) -> list[str]:
    """[(문장 인덱스, 전이 타입)] → 길이 n_sentences 의 라벨 시퀀스."""
    labels = [NOBORDER] * n_sentences
    for i, t in boundaries:
        if 0 <= i < n_sentences:
            labels[i] = t
    return labels


def relaxed_f1_matched(gold_idx: list[int], pred_idx: list[int],
                       tolerance: int = 3) -> tuple[float, float, float]:
    """우리 자체 완화 F1 — **논문 대조용이 아니다.** 진단용으로만 쓴다.

    표준적인 1:1 그리디 최근접 매칭(±tolerance 대칭, gold 하나는 한 번만 소비).
    `relaxed_f1_zehe2025` 와 값이 다르며, 그 차이가 곧 저자 지표의 관대함이다.
    """
    remaining = sorted(gold_idx)
    used = [False] * len(remaining)
    tp = 0
    for p in sorted(pred_idx):
        best, best_d = -1, tolerance + 1
        for i, g in enumerate(remaining):
            if used[i]:
                continue
            d = abs(g - p)
            if d <= tolerance and d < best_d:
                best, best_d = i, d
        if best >= 0:
            used[best] = True
            tp += 1
    prec = tp / len(pred_idx) if pred_idx else 0.0
    rec = tp / len(gold_idx) if gold_idx else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def iou(gold_offsets: list[int], pred_offsets: list[int], text_length: int) -> float:
    """Schneider et al. 2021 의 장면 단위 IoU (문자 길이 가중, 전체 길이로 정규화).

    각 gold 장면에 대해 겹침이 가장 큰 pred 장면을 찾아 |∩|/|∪| 를 길이 가중 합산한다.
    """
    def spans(offs):
        bounds = [0] + sorted(offs) + [text_length]
        return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    g_spans, p_spans = spans(gold_offsets), spans(pred_offsets)
    total = 0.0
    for gb, ge in g_spans:
        best = 0.0
        for pb, pe in p_spans:
            inter = max(0, min(ge, pe) - max(gb, pb))
            if inter == 0:
                continue
            union = max(ge, pe) - min(gb, pb)
            best = max(best, inter / union)
        total += best * (ge - gb)
    return total / text_length if text_length else 0.0


# --------------------------------------------------------------------------
# embedding-delta 레포에 동봉된 공식 제출물 덤프 파서
# --------------------------------------------------------------------------
_TUPLE_RE = re.compile(r"\(\s*(\d+)\s*,\s*'([^']+)'\s*\)")


def parse_official_dump(path) -> dict[str, Boundaries]:
    """`embedding-delta/training/stss_results/{t1,t2}/N` 형식을 읽는다.

    파일 구조: `gold` 줄 뒤에 (offset, 'Type-to-Type') 튜플들, `pred` 줄 뒤에 같은 형식.
    텍스트 길이는 파일에 없다 → 두 리스트의 최대 오프셋을 상한으로 쓰고, 정확한
    길이가 필요한 지표(IoU)에는 별도로 주입해야 한다. exact micro F1 은 길이에
    의존하지 않는다(배경 클래스가 점수에서 빠지므로).
    """
    from pathlib import Path

    text = Path(path).read_text()
    i = text.index("pred")
    out = {}
    for key, chunk in (("gold", text[:i]), ("pred", text[i:])):
        offs, typs = [], []
        for m in _TUPLE_RE.finditer(chunk):
            offs.append(int(m.group(1)))
            typs.append(m.group(2))
        out[key] = Boundaries(offs, typs, text_length=0)
    n = max(max(out["gold"].offsets, default=0), max(out["pred"].offsets, default=0)) + 1
    out["gold"].text_length = n
    out["pred"].text_length = n
    return out
