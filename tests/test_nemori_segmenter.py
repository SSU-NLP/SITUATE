"""M5 분절기가 Nemori `BatchSegmenter` 의 **드롭인 대체**인지 검증한다.

LLM·DB·네트워크 없이 돈다 (M5 체크포인트와 인코더만 필요).

    .venv/bin/python -m pytest tests/test_nemori_segmenter.py -v
"""
from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO / "src"), str(REPO / "external/nemori")):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("nemori", reason="external/nemori 가 clone 돼 있어야 한다")

from hi_ontop.nemori_segmenter import (  # noqa: E402
    M5BatchSegmenter, M5SegmenterConfig, install, uninstall,
)


@dataclass
class FakeMessage:
    """`nemori.domain.models.Message` 중 분절기가 실제로 쓰는 부분만."""

    role: str
    content: str
    timestamp: datetime | None = None

    def text_content(self) -> str:
        return self.content


TOPIC_A = [
    "I signed up for the Seattle marathon last week.",
    "The registration fee was higher than I expected this year.",
    "My training plan starts on Monday with easy recovery runs.",
    "I want to break four hours if my knee holds up.",
    "I bought new shoes because the old ones were worn flat.",
]
TOPIC_B = [
    "Did the plumber ever come to look at the kitchen faucet?",
    "He said the cartridge inside the handle was worn out.",
    "The replacement part had to be ordered from the supplier.",
    "He finally installed it yesterday afternoon.",
    "The counter has been completely dry since then.",
]


def _messages(texts):
    return [FakeMessage(role="user" if i % 2 == 0 else "assistant", content=t)
            for i, t in enumerate(texts)]


def test_signature_matches_original():
    """원본과 생성자·메서드 시그니처가 호환돼야 교체가 성립한다."""
    from nemori.llm.generators.segmenter import BatchSegmenter

    original = inspect.signature(BatchSegmenter.__init__).parameters
    ours = inspect.signature(M5BatchSegmenter.__init__).parameters
    assert "orchestrator" in original and "orchestrator" in ours
    # 원본 호출 형태 `BatchSegmenter(orchestrator=...)` 가 그대로 통해야 한다
    M5BatchSegmenter(orchestrator=object())

    assert inspect.iscoroutinefunction(BatchSegmenter.segment)
    assert inspect.iscoroutinefunction(M5BatchSegmenter.segment)


def test_install_replaces_and_uninstall_restores():
    import nemori.core.memory_system as ms
    from nemori.llm.generators.segmenter import BatchSegmenter

    assert ms.BatchSegmenter is BatchSegmenter
    install(M5SegmenterConfig(device="cpu"))
    assert ms.BatchSegmenter is M5BatchSegmenter
    uninstall()
    assert ms.BatchSegmenter is BatchSegmenter


@pytest.mark.asyncio
async def test_returns_contract_shape():
    seg = M5BatchSegmenter(config=M5SegmenterConfig(device="cpu"))
    msgs = _messages(TOPIC_A + TOPIC_B)
    groups = await seg.segment(msgs)

    assert isinstance(groups, list) and groups
    for g in groups:
        assert set(g) == {"messages", "topic"}
        assert isinstance(g["topic"], str) and g["topic"]
        assert g["messages"], "빈 그룹을 내면 안 된다"


@pytest.mark.asyncio
async def test_partition_is_complete_and_ordered():
    """원본과 달리 M5 는 메시지를 **하나도 버리지 않고** 순서를 지킨 연속 분할을 낸다."""
    seg = M5BatchSegmenter(config=M5SegmenterConfig(device="cpu"))
    msgs = _messages(TOPIC_A + TOPIC_B)
    groups = await seg.segment(msgs)

    flat = [m for g in groups for m in g["messages"]]
    assert flat == msgs, "전체를 순서대로 덮어야 한다"


@pytest.mark.asyncio
async def test_short_input_single_group():
    seg = M5BatchSegmenter(config=M5SegmenterConfig(device="cpu"))
    for n in (0, 1):
        groups = await seg.segment(_messages(TOPIC_A[:n]))
        assert len(groups) == 1


@pytest.mark.asyncio
async def test_failure_falls_back_to_single_group(monkeypatch):
    """원본은 예외 시 `[{messages, topic:'conversation'}]` 한 덩어리를 낸다 — 동일해야 한다."""
    import hi_ontop.nemori_segmenter as mod

    def boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "_boundaries", boom)
    seg = M5BatchSegmenter(config=M5SegmenterConfig(device="cpu"))
    msgs = _messages(TOPIC_A)
    groups = await seg.segment(msgs)
    assert len(groups) == 1
    assert groups[0]["messages"] == msgs
    assert groups[0]["topic"] == "conversation"


@pytest.mark.asyncio
async def test_no_llm_calls():
    """분절 경로에서 orchestrator 를 절대 건드리지 않는다."""

    class ExplodingOrchestrator:
        async def execute(self, request):  # pragma: no cover
            raise AssertionError("M5 분절기가 LLM 을 호출했다")

        async def execute_batch(self, requests):  # pragma: no cover
            raise AssertionError("M5 분절기가 LLM 을 호출했다")

    seg = M5BatchSegmenter(orchestrator=ExplodingOrchestrator(),
                           config=M5SegmenterConfig(device="cpu"))
    await seg.segment(_messages(TOPIC_A + TOPIC_B))


def test_frozen_config_matches_registry():
    """동결본(`configs/m5_default.json`)에서 벗어나면 즉시 알아채야 한다."""
    resolved = M5SegmenterConfig().resolved()
    assert resolved["mode"] == "online_nf"
    assert resolved["c"] == 1.0
    assert resolved["checkpoint"] == "models/m5_default/M5.pt"
    assert resolved["unit"] == "message"
    assert resolved["include_role"] is False
