#!/usr/bin/env python3
"""고정간격 · 무작위 하한 베이스라인 — **밀도를 맞춘** 통제군.

## 왜 필요한가
분절 점수는 **경계 밀도**에 크게 좌우된다. 어떤 방법이 좋아 보일 때, 그게
*"경계를 잘 놓아서"* 인지 *"경계 개수가 우연히 맞아서"* 인지 구분되지 않으면 주장이 서지 않는다.
이 두 베이스라인은 **밀도를 비교 대상과 똑같이 맞춘 채 위치만 무의미하게** 놓는다.
⇒ `우리 방법 − 이 하한` 이 곧 **위치 선택의 기여**다.

## 🔴 SuperDialseg 의 동명 클래스와 다르다
`super_dialseg.models.baselines` 에도 `RandomSegmenter`·`EvenSegmenter` 가 있지만,
그쪽은 경계 개수를 **무작위로** 뽑는다(`b = random.randint(0, k-1)`).
밀도가 통제되지 않으므로 *"밀도를 맞춘 통제군"* 으로 쓸 수 없다. 그래서 별도로 둔다.

## 인과성
둘 다 **정의상 온라인**이다 — 턴 t 의 판정이 t 이전 정보(위치·난수)만 쓴다.
따라서 오프라인/온라인 점수가 같아야 정상이며, 그 자체가 하네스의 sanity check 가 된다.

## 밀도를 어디에 맞추나
`rate` = 전이당 경계 확률(= 경계 수 / 전이 수). 비교 대상의 실측 밀도를 넣는다.
- NES 무대: **Hi-OnTop 의 문서별 예측 경계 수**에 맞춘다(사용자 지정).
- DTS 무대: 비교하려는 방법의 실측 밀도, 또는 gold 밀도(🔴 gold 를 쓰면 **oracle 누출**이므로
  성능 주장에 쓰지 말고 참조선으로만).
"""
from __future__ import annotations

import random


class _Base:
    """SuperDialseg `BaseSegmenter` 와 같은 호출 규약만 맞춘 얇은 베이스.

    하네스가 요구하는 것은 `forward(inputs) -> list[int]` 와 `evaluation` 누적기뿐이다.
    """

    def __init__(self):
        from super_dialseg.metrics.segmentation import SegmentationEvaluation

        self.evaluation = SegmentationEvaluation()

    @staticmethod
    def _n(inputs, ignore_index: int = -100) -> int:
        if "utterances" in inputs:
            return len(inputs["utterances"])
        if "labels" in inputs:
            return len([x for x in inputs["labels"] if x != ignore_index])
        raise AttributeError("inputs 에 utterances 나 labels 가 있어야 한다")

    def to(self, device):        # 하네스 호환용 no-op
        return self


class FixedIntervalSegmenter(_Base):
    """`period` 턴마다 경계를 하나 놓는다. 결정적이고 정의상 인과적이다.

    `period` 대신 `rate` 를 주면 `period = round(1/rate)` 로 환산한다.
    """

    def __init__(self, period: int | None = None, rate: float | None = None):
        super().__init__()
        if period is None:
            if not rate or rate <= 0:
                raise ValueError("period 또는 rate 중 하나는 양수여야 한다")
            period = max(1, round(1 / rate))
        self.period = int(period)

    def forward(self, inputs, ignore_index: int = -100) -> list[int]:
        n = self._n(inputs, ignore_index)
        # 마지막 전이는 하네스가 0 으로 강제하므로 여기서 굳이 비우지 않는다
        return [1 if (i + 1) % self.period == 0 else 0 for i in range(n)]


class RandomRateSegmenter(_Base):
    """각 전이를 확률 `rate` 로 독립적으로 경계로 찍는다.

    🔴 **시드를 문서 위치가 아니라 호출 순서에 걸면 안 된다.** 온라인 평가에서는 같은 문서에
    forward 가 n 번 호출되므로, 호출마다 난수를 새로 뽑으면 앞선 턴의 판정이 매번 바뀐다
    (= 인과성 검사 실패). 그래서 **전이 인덱스로 결정되는 난수**를 쓴다 —
    같은 전이는 몇 번을 물어도 같은 답이 나온다.
    """

    def __init__(self, rate: float, seed: int = 20260803):
        super().__init__()
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate 는 [0,1]")
        self.rate = rate
        self.seed = seed
        self._doc = 0          # 문서 구분용 — 하네스가 문서마다 새 인스턴스를 만든다면 0 고정

    def forward(self, inputs, ignore_index: int = -100) -> list[int]:
        n = self._n(inputs, ignore_index)
        out = []
        for i in range(n):
            # random.Random 은 tuple 시드를 안 받으므로 문자열로 합친다.
            r = random.Random(f"{self.seed}|{self._doc}|{i}")   # 전이 i 에 고정된 난수
            out.append(1 if r.random() < self.rate else 0)
        return out

    def new_document(self) -> None:
        """문서가 바뀌었음을 알린다(난수 계열 분리)."""
        self._doc += 1
