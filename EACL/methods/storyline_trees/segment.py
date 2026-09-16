#!/usr/bin/env python3
"""Storyline Trees (Ou & Lapata 2026) — **장면 분절 단계만** 재구현.

## 범위 (사용자 결정, 2026-08-03)
논문의 주 기여는 스토리라인 트리 기반 적응 검색과 서사 QA 정확도(DetectiveQA·NovelQA·
LiteraryQA)다. 장면 분절(§3.1)은 트리 구축의 **기본 단위를 만드는 전처리**다.
우리가 쓰려는 것이 분절기이므로 §3.1 만 구현하고, 논문의 QA 수치는 **대조하지 않는다.**
Top-down/bottom-up 트리 유도(§3.2·3.3)와 적응 검색(§4)은 범위 밖이다.

## 🔴 코드는 공개되지 않았다
논문 각주 1 이 `https://github.com/Leonard907/StoryTree` 를 가리키지만, 2026-08-03 확인
결과 해당 레포에는 **README(한 줄)·LICENSE·.gitignore 만 있고 코드가 없다**
(`_upstream/StoryTree/` 에 클론해 둠). 따라서 이 구현의 사양은 논문 §3.1 본문과
**Appendix Table 14 의 프롬프트 전문**뿐이다.

## 방법 (§3.1)
장면 = 시간·장소·서사 초점이 대체로 일관된 연속 구간. 그중 하나가 바뀌면 경계.
1. 원문을 **장(chapter)** 단위로 파싱한다.
2. 장마다 문장에 `[1] [2] …` 번호를 붙이고 **표준 장면 추출 프롬프트를 그대로** 건다
   (전 작품에 같은 프롬프트 — 책마다 프롬프트를 바꾸지 않는 것이 논문의 설계다).
3. 모델은 장면별 `scene_title` / `ending_sentence` / `synopsis` / `significance` 를 낸다.
   경계 복원의 근거는 **`ending_sentence`(장면의 마지막 문장 번호)** 다.

## 출력
`ending_sentence` 는 장면의 *끝*이므로, 경계(다음 장면의 시작)는 `ending_sentence + 1`
번째 문장이다. 문서 전체 기준 0-based 인덱스로 돌려준다.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.llm_client import chat, extract_json, make_client  # noqa: E402
from storyline_trees.prompts import SCENE_EXTRACTION  # noqa: E402

DEFAULT_MODEL = "openai/gpt-4.1"
DEFAULT_MAX_WORDS = 12000
CHAPTER_RE = re.compile(
    r"^\s*(chapter|kapitel|book|part|canto|act)\b.{0,60}$|^\s*(?:[IVXLC]+|\d{1,3})\.?\s*$",
    re.IGNORECASE)


@dataclass
class Result:
    boundaries: list[int] = field(default_factory=list)
    scenes: list[dict] = field(default_factory=list)
    fallbacks: int = 0
    calls: int = 0


def split_into_chapters(sentences: list[str], max_words: int = DEFAULT_MAX_WORDS,
                        chapter_starts: list[int] | None = None) -> list[list[int]]:
    """장 단위 분할. 장 표지가 없거나 장이 너무 길면 단어 수 상한으로 쪼갠다.

    논문은 "원문을 장으로 파싱한다"까지만 말하고 장이 컨텍스트를 넘을 때를 명시하지
    않는다. 넘치면 호출이 실패하므로 **단어 상한 그리디 분할**을 폴백으로 둔다 — 이건
    논문에 없는 우리 결정이므로 결과 보고에 명시할 것.
    """
    if chapter_starts is None:
        chapter_starts = [i for i, s in enumerate(sentences) if CHAPTER_RE.match(s.strip())]
    bounds = sorted(set([0] + list(chapter_starts) + [len(sentences)]))
    units = [list(range(a, b)) for a, b in zip(bounds, bounds[1:]) if b > a]

    out = []
    for unit in units:
        cur, w = [], 0
        for i in unit:
            n = len(sentences[i].split())
            if cur and w + n > max_words:
                out.append(cur)
                cur, w = [], 0
            cur.append(i)
            w += n
        if cur:
            out.append(cur)
    return out


def render(sentences: list[str], ids: list[int]) -> str:
    return "\n".join(f"[{k + 1}] {sentences[i].strip()}" for k, i in enumerate(ids))


def _validate(scenes, n) -> tuple[bool, str]:
    """§2 Strict Constraints: 오름차순 + 마지막 장면의 ending_sentence == n."""
    if not scenes:
        return False, "장면이 비었다"
    try:
        ends = [int(s["ending_sentence"]) for s in scenes]
    except (KeyError, TypeError, ValueError) as e:
        return False, f"ending_sentence 파싱 실패: {e}"
    if any(e < 1 or e > n for e in ends):
        return False, f"범위를 벗어난 ending_sentence: {ends[:5]}"
    if any(ends[i + 1] <= ends[i] for i in range(len(ends) - 1)):
        return False, "ending_sentence 가 단조증가하지 않는다"
    if ends[-1] != n:
        return False, f"마지막 ending_sentence 가 {n} 이 아니다({ends[-1]})"
    return True, ""


def segment_chapter(client, sentences, ids, model, *, max_retries=2, temperature=0.0):
    n = len(ids)
    prompt = SCENE_EXTRACTION.replace("{{{chapter_text}}}", render(sentences, ids))
    calls, last = 0, ""
    for _ in range(max_retries + 1):
        try:
            r = chat(client, model, "You are a helpful assistant.", prompt,
                     temperature=temperature)
            calls += 1
            scenes = extract_json(r)
            if isinstance(scenes, dict):
                scenes = scenes.get("scenes", [])
            ok, why = _validate(scenes, n)
            if ok:
                return scenes, calls, False
            last = why
        except (ValueError, RuntimeError) as e:
            last = str(e)
        temperature = min(0.7, temperature + 0.3)
    print(f"    ⚠️ 폴백(장 통째 1장면): {last}")
    return [{"scene_title": "FALLBACK", "ending_sentence": n,
             "synopsis": "", "significance": ""}], calls, True


def segment_document(sentences: list[str], *, model: str = DEFAULT_MODEL,
                     max_words: int = DEFAULT_MAX_WORDS,
                     chapter_starts: list[int] | None = None, client=None) -> Result:
    client = client or make_client()
    chapters = split_into_chapters(sentences, max_words, chapter_starts)
    res = Result()
    print(f"  장 단위 {len(chapters)}개 (문장 {len(sentences)})")
    for ci, ids in enumerate(chapters):
        scenes, calls, fb = segment_chapter(client, sentences, ids, model)
        res.calls += calls
        res.fallbacks += int(fb)
        # 장의 시작 자체가 경계다 (문서 첫 문장은 제외)
        if ci > 0:
            res.boundaries.append(ids[0])
        for s in scenes[:-1]:                      # 마지막 장면의 끝 = 장의 끝이라 중복
            local_end = int(s["ending_sentence"])  # 1-based, 장면의 마지막 문장
            nxt = local_end                        # 다음 장면의 첫 문장 = 0-based local_end
            if nxt < len(ids):
                res.boundaries.append(ids[nxt])
        res.scenes.extend({**s, "chapter": ci} for s in scenes)
    res.boundaries = sorted(set(res.boundaries))
    return res
