#!/usr/bin/env python3
"""기존 스트리밍 온라인 구현을 **하네스 C** 에서 돌리기 위한 어댑터.

## 왜 어댑터인가
`methods/*/online/` 과 `src/hi_ontop/baselines/` 의 스트리밍 구현들은 전부
**하네스 B**(Def-DTS 번들 + 우리 segeval) 전용으로 짜여 있다. 논문과 같은 축에서
`Δ = online − offline` 을 재려면 **하네스 C**(공식 데이터 + 저자 `SegmentationEvaluation`)
에서 돌아야 한다. 알고리즘은 손대지 않고 **호출 규약만** 맞춘다.

## 🔴 "온라인"에는 세 단계가 있다 — 뭉뚱그리면 논증이 무너진다
| 단계 | 미래 사용 | 판정 확정 시점 | 예 |
|---|---|---|---|
| **zero-lookahead** | 전혀 없음 | 턴 t 도착 즉시 | prefix-recompute (`evaluate_online`) |
| **bounded-lookahead** | **k 블록 / delay=d 만큼 있음** | 턴 t+d | `StreamingTextTiling`(k 블록 대기) · `CSMOnlineDelay2`(d=2) |
| offline | 무제한 | 문서 종료 후 | 원 논문 |

스트리밍 구현은 **bounded-lookahead** 다. `StreamingTextTiling.push()` 의 docstring 이
명시한다 — *"gap 의 score 는 right block(k pseudo-sentence)이 모두 닫혀야 평가 가능하므로,
boundary 가 확정되는 시점은 그 boundary 의 utterance 가 들어온 시점보다 늦다."*

⇒ 스트리밍 점수를 "미래를 안 쓴 점수"라고 쓰면 **틀린다.** 쓸 수 있는 문장은
*"제한된 lookahead 로 얻은 점수"* 다. Causal-SSC 논증은 이 세 점을 잇는
**lookahead 곡선**으로 세우는 편이 정확하다(단일 Δ 하나로는 무엇이 변했는지 특정되지 않는다).

## 구현
어댑터는 `forward(inputs) -> list[int]` 와 `evaluation` 만 제공하면 되고,
인과성은 **내부 push 루프가 보장**한다(각 push 는 그 시점까지의 발화만 본다).
따라서 하네스의 **오프라인 `evaluate()`** 를 그대로 써도 되며 O(n²) 재계산이 필요 없다.
"""
from __future__ import annotations


class _AdapterBase:
    def __init__(self):
        from super_dialseg.metrics.segmentation import SegmentationEvaluation

        self.evaluation = SegmentationEvaluation()

    def to(self, device):        # 하네스 호환용 no-op
        return self

    @staticmethod
    def _utts(inputs) -> list[str]:
        if "utterances" not in inputs:
            raise AttributeError("inputs 에 utterances 가 있어야 한다")
        return list(inputs["utterances"])


class StreamingTextTilingAdapter(_AdapterBase):
    """`hi_ontop.baselines.StreamingTextTiling` 을 하네스 C 규약으로 감싼다.

    **bounded-lookahead** — running threshold(Welford) + k 블록 대기.
    알고리즘·상태 갱신은 원 구현 그대로이고, 여기서는 문서마다 인스턴스를 새로 만들고
    `push` 반환값을 발화 플래그로 환산할 뿐이다.

    🔴 하이퍼: 기본을 **원 논문값 w=10 · k=6** 으로 둔다. 기존 러너
    (`methods/texttiling/online/streaming.py`)는 짧은 대화 때문에 w=5·k=3 으로 낮췄는데,
    그 상태로 오프라인(w=10·k=6)과 비교하면 **하이퍼 변경과 온라인화가 섞인다.**
    낮춘 조건이 필요하면 명시적으로 넘기고 결과에 병기한다.
    """

    def __init__(self, w: int = 10, k: int = 6, c: float = 0.5,
                 min_gap: int = 4, warmup_gaps: int = 3):
        super().__init__()
        self.kw = {"w": w, "k": k, "c": c, "min_gap": min_gap, "warmup_gaps": warmup_gaps}

    def forward(self, inputs, ignore_index: int = -100) -> list[int]:
        from hi_ontop.baselines import StreamingTextTiling

        utts = self._utts(inputs)
        seg = StreamingTextTiling(**self.kw)
        flags = [0] * len(utts)
        for u in utts:
            for b in seg.push(u):          # 1-based utterance index (지연 확정 가능)
                if 1 <= b <= len(flags):
                    flags[b - 1] = 1
        return flags


def make(name: str, **kw):
    """이름 → 어댑터 팩토리."""
    if name == "texttiling-streaming":
        return lambda: StreamingTextTilingAdapter(**kw)
    raise ValueError(f"미지원 어댑터: {name}")
