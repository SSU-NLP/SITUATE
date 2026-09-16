#!/usr/bin/env python3
"""LitSeg (Zhang et al. 2026) — **분절 단계만** 재구현.

## 범위 (사용자 결정, 2026-08-03)
논문의 주 기여는 문학 RAG 의 검색·QA 성능이고 분절은 그 전처리다. 우리가 필요한 것은
분절기이므로 **§3.1 의 3단계 분절 파이프라인만** 구현한다. 논문의 검색 지표
(GutenQA CR 0.938 / MRR 0.683 / H@5 0.813)는 **재현 대상이 아니다** — 대조하지 않는다.
LitSeg-Lite(SFT+GRPO 증류 학생 모델)도 범위 밖이다.

## 구현 근거
저자 코드가 공개되지 않았다. 사양은 논문 §3.1.1(이론 프레임)·§3.1.2(구현)·
Appendix D.1(전체 프롬프트)뿐이며, 프롬프트는 `prompts.py` 에 문구 그대로 옮겼다.

## 파이프라인 (§3.1.2)
1. **전처리** — 최상위 서사 단위(장/막/절)로 나눈다. 메타데이터가 없거나 단위가
   n 단어를 넘으면 문단→문장 순으로 재귀 분할해 n 단어 제약을 그리디하게 만족시킨다.
   각 후보 청크의 문장에 1부터 순차 인덱스를 붙인다.
2. **3단계 생성** — 전 단계가 같은 시스템 프롬프트를 공유하고, 사용자 프롬프트에만
   단계별 지시와 이전 단계 출력이 들어간다. 모델은 본문이 아니라 **문장 인덱스**를 낸다.
3. **검증·폴백** — 커버리지·순서·경계 규칙을 검사하고, 위반하면 재시도한 뒤
   그래도 실패하면 해당 청크를 통째로 한 세그먼트로 둔다(§3.1.2 "validation with
   fallback mechanisms").

## 출력
`from_idx` 가 2번째 세그먼트부터 경계다. 0-based 문장 인덱스 리스트로 돌려준다.
`context_idx`(비인접 문맥 문장)는 검색용 부가 정보라 경계에 영향을 주지 않는다.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.llm_client import chat, extract_json, make_client  # noqa: E402
from litseg import prompts  # noqa: E402

DEFAULT_MODEL = "openai/gpt-4.1"
DEFAULT_MAX_WORDS = 8000       # §3.1.2 의 n — 모델 컨텍스트 예산에 맞춘 상한
CHAPTER_RE = re.compile(
    r"^\s*(chapter|kapitel|book|part|canto|act|scene)\b.{0,60}$|^\s*(?:[IVXLC]+|\d{1,3})\.?\s*$",
    re.IGNORECASE)


@dataclass
class Chunk:
    """후보 청크 = LLM 한 번에 들어가는 단위. sent_ids 는 문서 전체 기준 0-based 인덱스."""

    sent_ids: list[int]
    sentences: list[str]


@dataclass
class Result:
    boundaries: list[int] = field(default_factory=list)   # 문서 전체 기준 0-based
    segments: list[dict] = field(default_factory=list)
    fallbacks: int = 0
    calls: int = 0


def split_into_chunks(sentences: list[str], max_words: int = DEFAULT_MAX_WORDS,
                      chapter_starts: list[int] | None = None) -> list[Chunk]:
    """§3.1.2 전처리 — 구조 단위 우선, 초과분은 단어 수 제약으로 그리디 재귀 분할."""
    if chapter_starts is None:
        chapter_starts = [i for i, s in enumerate(sentences) if CHAPTER_RE.match(s.strip())]
    units: list[list[int]] = []
    if chapter_starts:
        bounds = sorted(set([0] + chapter_starts + [len(sentences)]))
        for a, b in zip(bounds, bounds[1:]):
            if b > a:
                units.append(list(range(a, b)))
    else:
        units.append(list(range(len(sentences))))

    chunks: list[Chunk] = []
    for unit in units:
        cur, cur_words = [], 0
        for i in unit:
            w = len(sentences[i].split())
            if cur and cur_words + w > max_words:
                chunks.append(Chunk(cur, [sentences[j] for j in cur]))
                cur, cur_words = [], 0
            cur.append(i)
            cur_words += w
        if cur:
            chunks.append(Chunk(cur, [sentences[j] for j in cur]))
    return chunks


def render(chunk: Chunk) -> str:
    """문장에 1-based 인덱스를 붙여 렌더링 — 프롬프트가 요구하는 `[1]` 형식."""
    return "\n".join(f"[{k + 1}] {s.strip()}" for k, s in enumerate(chunk.sentences))


def _validate(segments: list[dict], n: int) -> tuple[bool, str]:
    """§ Constraints 5·6 검사: 1 에서 시작, n 에서 끝, 순서, 전 문장 커버."""
    if not segments:
        return False, "세그먼트가 비었다"
    try:
        spans = [(int(s["from_idx"]), int(s["to_idx"])) for s in segments]
    except (KeyError, TypeError, ValueError) as e:
        return False, f"인덱스 파싱 실패: {e}"
    if spans[0][0] != 1:
        return False, f"첫 from_idx 가 1 이 아니다({spans[0][0]})"
    if spans[-1][1] != n:
        return False, f"마지막 to_idx 가 {n} 이 아니다({spans[-1][1]})"
    if any(a > b for a, b in spans):
        return False, "from_idx > to_idx 인 세그먼트가 있다"
    if any(spans[i + 1][0] <= spans[i][0] for i in range(len(spans) - 1)):
        return False, "세그먼트 시작이 단조증가하지 않는다"
    covered = set()
    for a, b in spans:
        covered.update(range(a, b + 1))
    if len(covered) != n:
        return False, f"커버리지 부족: {len(covered)}/{n}"
    return True, ""


def segment_chunk(client, chunk: Chunk, model: str, *, max_retries: int = 2,
                  temperature: float = 0.0) -> tuple[list[dict], int, bool]:
    """한 청크에 3단계 프롬프팅. (세그먼트, 호출 수, 폴백여부) 반환."""
    content = render(chunk)
    n = len(chunk.sentences)
    calls = 0
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            r1 = chat(client, model, prompts.SYSTEM,
                      prompts.STAGE_1.format(chapter_content=content), temperature=temperature)
            calls += 1
            s1 = json.dumps(extract_json(r1), ensure_ascii=False)

            r2 = chat(client, model, prompts.SYSTEM,
                      prompts.STAGE_2.format(chapter_content=content, step_1_result=s1),
                      temperature=temperature)
            calls += 1
            s2 = json.dumps(extract_json(r2), ensure_ascii=False)

            r3 = chat(client, model, prompts.SYSTEM,
                      prompts.STAGE_3.format(chapter_content=content, step_1_result=s1,
                                             step_2_result=s2, last_index=n),
                      temperature=temperature)
            calls += 1
            out = extract_json(r3)
            segs = out.get("segments", []) if isinstance(out, dict) else []
            ok, why = _validate(segs, n)
            if ok:
                return segs, calls, False
            last_err = why
        except (ValueError, RuntimeError) as e:
            last_err = str(e)
        # 재시도 시 온도를 조금 올려 같은 실패를 반복하지 않게 한다
        temperature = min(0.7, temperature + 0.3)

    print(f"    ⚠️ 폴백(청크 통째 1세그먼트): {last_err}")
    return [{"subtitle": "FALLBACK", "from_idx": 1, "to_idx": n, "context_idx": []}], calls, True


def segment_document(sentences: list[str], *, model: str = DEFAULT_MODEL,
                     max_words: int = DEFAULT_MAX_WORDS,
                     chapter_starts: list[int] | None = None,
                     client=None) -> Result:
    """문서 전체를 분절한다. 반환 경계는 문서 기준 0-based 문장 인덱스."""
    client = client or make_client()
    chunks = split_into_chunks(sentences, max_words, chapter_starts)
    res = Result()
    print(f"  후보 청크 {len(chunks)}개 (문장 {len(sentences)})")
    for ci, ch in enumerate(chunks):
        segs, calls, fb = segment_chunk(client, ch, model)
        res.calls += calls
        res.fallbacks += int(fb)
        for si, s in enumerate(segs):
            local = int(s["from_idx"]) - 1
            g = ch.sent_ids[min(local, len(ch.sent_ids) - 1)]
            # 청크의 첫 세그먼트 시작은 청크 경계다. 문서 첫 문장은 경계가 아니다.
            if not (ci == 0 and si == 0):
                res.boundaries.append(g)
            res.segments.append({**s, "chunk": ci, "global_from": g})
    res.boundaries = sorted(set(res.boundaries))
    return res
