#!/usr/bin/env python3
r"""Nemori 의 LLM 배치 분절기를 **M5-FactoredSlot + 적응임계**로 갈아끼우는 어댑터.

Nemori(`nemori-ai/nemori`, arXiv 2508.03341)의 `nemori.llm.generators.segmenter.BatchSegmenter`
와 **동일한 인터페이스**를 갖는 대체 클래스를 제공한다. 교체 지점은 딱 하나 —
`MemorySystem._process` 안에서 버퍼된 메시지를 에피소드 그룹으로 나누는 호출뿐이다.
에피소드 서술 생성·시맨틱 메모리·병합·검색·QA 는 **원본 그대로** 둔다.

    from hi_ontop.nemori_segmenter import M5SegmenterConfig, install
    install(M5SegmenterConfig())          # nemori import 후 호출
    # 이후 NemoriMemory 를 평소처럼 쓰면 분절만 M5 로 바뀐다

## 인터페이스 계약 (원본 `BatchSegmenter` 와 동일해야 하는 것)
| 항목 | 값 |
|---|---|
| 생성자 | `BatchSegmenter(orchestrator=<LLMOrchestrator>)` — 우리는 orchestrator 를 **안 쓴다** |
| 메서드 | `async def segment(self, messages: list[Message]) -> list[dict]` |
| 반환 | `[{"messages": [Message, ...], "topic": str}, ...]` |
| 실패 시 | 원본과 동일하게 `[{"messages": messages, "topic": "conversation"}]` 한 덩어리 |

## 원본과 다른 점 (전부 의도적, REPORT 에 공시)
1. **LLM 호출 0회.** 원본은 80메시지 청크마다 1콜(temperature 0.2, JSON 모드).
2. **연속 분할만 낸다.** 원본 프롬프트는 비연속 인덱스 묶음(interleaved)을 허용하고,
   어느 그룹에도 안 들어간 메시지를 **버릴 수 있다**. M5 는 전체를 빈틈없이 덮는 연속 분할이다.
3. **80메시지 청킹 없음.** M5 는 온라인이라 컨텍스트 한계가 없다. 적응임계의 누적 평균·표준편차가
   대화 시작부터 이어진다(청크마다 리셋되지 않는다).
4. **topic 문자열을 만들지 않는다.** 기본값 `"conversation"` — 이건 우리가 지어낸 값이 아니라
   **nemori 자신이 분절을 건너뛰거나 실패할 때 쓰는 값**(`memory_system.py:113`,
   `segmenter.py:87`)이라 하류 프롬프트 입장에서 in-distribution 이다.
   에피소드 생성 프롬프트의 `Boundary detection reason` 자리에 들어간다.

## 입력 규약 (LoCoMo 를 보기 전에 동결한 것 — 사후 선택 아님)
- **단위 = 메시지(turn) 1개**. 프로젝트의 기존 대화 평가 규약과 같다
  (`scripts/eacl_phase0/tolerance_bands.py::load_dialogue_val` — SuperDialseg turn 단위).
- **화자 접두사 없음.** 위 규약이 `t["utterance"]` 원문만 인코더에 넣는다. 같은 형식을 따른다.
- **임계 c=1.0 · `online_nf`** 동결. c 의 출처는 DTS 공식 test 36,109 전이(D-B39)이며
  LoCoMo 에서 재탐색하지 않는다.

## 🔴 알아야 할 한계
- M5 는 **영어 서사 픽션 앵커**로 학습됐다(D-M8). 대화는 도메인 밖이다.
  참고 수치: dev-3 서사 Score 0.4053 / superseg validation 60대화 0.3628(둘 다 Δ·방향용).
- dev-3 에서 **약 2.2배 과분절**한다. 대화에서도 과분절하면 에피소드 수가 늘어
  하류 LLM 호출·검색 단위가 함께 늘어난다 — 비용 축을 반드시 같이 보고할 것.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger("nemori")

REPO = Path(__file__).resolve().parents[2]

# 인코더 forward 를 직렬화한다. nemori 는 대화 10개를 동시에 굴리므로 여러 코루틴이 같은
# SentenceTransformer/torch 모듈에 동시에 들어올 수 있다. 재현성(배치 구성 고정)과
# 스레드 안전 둘 다를 위해 한 번에 하나만 통과시킨다.
_GPU_LOCK = threading.Lock()


@dataclass
class M5SegmenterConfig:
    """M5 분절기 동작 설정. 기본값 = `configs/m5_default.json` 동결본."""

    checkpoint: str | None = None
    """None 이면 `configs/m5_default.json` 의 동결 체크포인트(= 현행 디폴트 I2)."""

    c: float | None = None
    """None 이면 config 의 c(=1.0). 적응임계 `s_t >= mean(s_<t) + c*std(s_<t)`."""

    mode: str | None = None
    """None 이면 config 의 mode(=`online_nf`)."""

    device: str = "cpu"
    """기본 CPU. 로컬 GPU 4장은 vLLM 이 쓰고 있고, 5,882 메시지 인코딩은 CPU 로 충분하다."""

    unit: str = "message"
    """`message`(=turn 단위, 동결 규약) 또는 `sentence`(문장 분절 후 메시지로 사영)."""

    include_role: bool = False
    """True 면 `"{role}: {text}"` 를 인코딩. 동결 규약은 False(원문만)."""

    topic: str = "conversation"
    """그룹의 topic 문자열. nemori 자신의 분절-없음 폴백 값과 동일하게 둔다."""

    trace_path: str | None = None
    """지정하면 배치마다 분절 통계를 JSONL 로 append 한다(REPORT 용)."""

    def resolved(self) -> dict[str, Any]:
        from hi_ontop.m5_default import load_config

        cfg = load_config()
        return {
            "checkpoint": self.checkpoint or cfg["checkpoint"],
            "c": cfg["threshold"].get("c", 1.0) if self.c is None else self.c,
            "mode": cfg["threshold"]["mode"] if self.mode is None else self.mode,
            "encoder": cfg["encoder"],
            "device": self.device,
            "unit": self.unit,
            "include_role": self.include_role,
            "topic": self.topic,
        }


def _unit_texts(messages: Sequence[Any], cfg: M5SegmenterConfig) -> list[str]:
    out = []
    for m in messages:
        text = m.text_content() if hasattr(m, "text_content") else str(m)
        text = " ".join(str(text).split()) or "."
        out.append(f"{m.role}: {text}" if cfg.include_role else text)
    return out


def _sentence_split(texts: Sequence[str]) -> tuple[list[str], list[int]]:
    """메시지별 텍스트를 문장으로 쪼개고, 각 문장이 어느 메시지에서 왔는지 반환."""
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:  # pragma: no cover - 최초 1회
        nltk.download("punkt_tab", quiet=True)

    sents: list[str] = []
    owner: list[int] = []
    for i, t in enumerate(texts):
        parts = [s.strip() for s in nltk.sent_tokenize(t) if s.strip()] or [t]
        sents.extend(parts)
        owner.extend([i] * len(parts))
    return sents, owner


def _boundaries(texts: list[str], cfg: M5SegmenterConfig) -> tuple[list[int], dict[str, Any]]:
    """메시지 인덱스 기준 경계(새 그룹을 여는 메시지 번호) 리스트를 돌려준다."""
    from hi_ontop.m5_default import segment as m5_segment

    t0 = time.perf_counter()
    if cfg.unit == "sentence":
        sents, owner = _sentence_split(texts)
        with _GPU_LOCK:
            r = m5_segment(sents, c=cfg.c, mode=cfg.mode, device=cfg.device,
                           checkpoint=cfg.checkpoint)
        # 문장 경계 -> 그 문장을 담은 메시지가 새 그룹을 연다(메시지 내부 분할은 불가).
        bnd = sorted({owner[s] for s in r["boundaries"] if owner[s] > 0})
        n_units = len(sents)
    else:
        with _GPU_LOCK:
            r = m5_segment(texts, c=cfg.c, mode=cfg.mode, device=cfg.device,
                           checkpoint=cfg.checkpoint)
        bnd = list(r["boundaries"])
        n_units = len(texts)

    scores = r["scores"]
    stats = {
        "n_messages": len(texts),
        "n_units": n_units,
        "n_boundaries": len(bnd),
        "n_groups": len(bnd) + 1,
        "mean_signal": float(scores.mean()) if len(scores) else 0.0,
        "seg_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return bnd, stats


class M5BatchSegmenter:
    """`nemori.llm.generators.segmenter.BatchSegmenter` 의 드롭인 대체.

    `orchestrator` 인자는 시그니처 호환을 위해 받기만 하고 **쓰지 않는다**(LLM 호출 0회).
    """

    def __init__(self, orchestrator: Any = None, config: M5SegmenterConfig | None = None) -> None:
        self._orchestrator = orchestrator
        self._cfg = config or _INSTALLED_CONFIG or M5SegmenterConfig()

    async def segment(self, messages: list[Any]) -> list[dict[str, Any]]:
        if len(messages) < 2:
            return [{"messages": list(messages), "topic": self._cfg.topic}]

        cfg = self._cfg
        try:
            texts = _unit_texts(messages, cfg)
            bnd, stats = await asyncio.to_thread(_boundaries, texts, cfg)
        except Exception as e:  # 원본 BatchSegmenter 와 동일한 폴백
            logger.warning("M5 segmentation failed, returning single group: %s", e)
            return [{"messages": list(messages), "topic": self._cfg.topic}]

        cuts = [0] + bnd + [len(messages)]
        groups = [
            {"messages": messages[a:b], "topic": cfg.topic}
            for a, b in zip(cuts[:-1], cuts[1:])
            if b > a
        ]
        if cfg.trace_path:
            _append_trace(cfg.trace_path, stats | {
                "group_sizes": [len(g["messages"]) for g in groups],
            })
        return groups or [{"messages": list(messages), "topic": cfg.topic}]


_TRACE_LOCK = threading.Lock()


def _append_trace(path: str, record: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _TRACE_LOCK:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


_INSTALLED_CONFIG: M5SegmenterConfig | None = None


def install(config: M5SegmenterConfig | None = None) -> dict[str, Any]:
    """`nemori.core.memory_system.BatchSegmenter` 를 M5 판으로 교체한다.

    원본 저장소 파일은 **한 글자도 고치지 않는다** — 모듈 전역 이름만 바꿔 끼운다.
    `MemorySystem._process` 가 `BatchSegmenter(orchestrator=...)` 로 호출하므로
    시그니처만 맞으면 된다.
    """
    global _INSTALLED_CONFIG
    _INSTALLED_CONFIG = config or M5SegmenterConfig()

    import nemori.core.memory_system as ms

    if getattr(ms.BatchSegmenter, "__name__", "") != "M5BatchSegmenter":
        ms._ORIGINAL_BatchSegmenter = ms.BatchSegmenter  # 되돌릴 수 있게 보관
    ms.BatchSegmenter = M5BatchSegmenter

    resolved = _INSTALLED_CONFIG.resolved()
    logger.info("M5 segmenter installed: %s", resolved)
    return resolved


def uninstall() -> None:
    import nemori.core.memory_system as ms

    original = getattr(ms, "_ORIGINAL_BatchSegmenter", None)
    if original is not None:
        ms.BatchSegmenter = original


def selftest(device: str = "cpu") -> dict[str, Any]:
    """의존성·체크포인트·경계 산출이 실제로 도는지 확인하는 최소 스모크."""
    cfg = M5SegmenterConfig(device=device)
    texts = [
        "I finally signed up for the Seattle marathon last week.",
        "My training plan starts on Monday with easy runs.",
        "I am aiming for a sub four hour finish.",
        "By the way, did you ever fix the leaking kitchen faucet?",
        "The plumber said the cartridge was worn out.",
        "He replaced it yesterday afternoon and it is dry now.",
    ]
    bnd, stats = _boundaries(texts, cfg)
    return {"config": cfg.resolved(), "boundaries": bnd, **stats}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(selftest(), ensure_ascii=False, indent=2))
