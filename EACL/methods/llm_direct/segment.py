#!/usr/bin/env python3
"""B2-LLM-Direct (Michelmann et al. 2023/2025) — 저자 절차 `A_1_segmentStory.py` 이식.

## 🔴 우리 기존 구현과 무엇이 다른가 (정정)
저장소는 `scripts/eacl_phase0/build_tier2_gold.py` 의 프롬프트를 "B2" 라고 불러 왔지만,
문자열 대조 결과 그것은 **B3(Panela/EventRecall)의 프롬프트**였다. B2 원문은 다르다:

    B2 (이 파일) : "...segmented into long events. ... whenever one long event ends and another begins."
                   꼬리: "...that is segmented into long events: "
    B3 (그쪽)    : "...segmented into large events. ... whenever one event ends and another begins."
                   꼬리: "...that is segmented into large event units: "

절차도 다르다. B3 는 전문을 **한 번에** 넣지만, B2 는 컨텍스트 창(2048 토큰)에 맞춰
**슬라이딩 창으로 여러 번** 호출하고 각 응답의 **마지막 사건을 버린 뒤** 그 사건의 앞머리를
원문에서 다시 찾아 다음 창의 시작점으로 삼는다. 이 "마지막 사건 폐기 + 재정렬"이 B2 의 핵심이라
한 번에 넣는 방식과는 경계 분포가 달라진다.

## 저자 절차 (A_1_segmentStory.py)
1. 전처리: `...` → `.` (nltk 가 문장 경계로 오인하지 않도록). 큰따옴표는 없다고 가정.
2. `min_check_length` = 원문의 **최장 반복 부분문자열 길이 + 1** — 재정렬 시 유일하게
   찾히는 최소 길이를 보장하는 장치다. 원 코드의 O(n²) DP 는 장문에서 못 쓰므로
   접미사 자동자(suffix automaton) 기반 O(n) 구현으로 대체했다 — **값은 동일**하고
   `verify_lrs.py` 가 무작위 문자열로 두 구현의 일치를 검사한다.
3. 프롬프트 = head + 창 텍스트 + tail + " \n", `text-davinci-002`, temperature=0,
   top_p=1, penalty 0, logprobs=5.
4. 응답을 개행으로 잘라 사건 리스트를 만들고, **아직 안 끝났으면 마지막 사건을 버린다**
   (창 끝에서 잘린 사건이므로).
5. 마지막 사건의 앞 `min_check_length` 글자를 원문에서 `rfind` 해 다음 시작 인덱스로.
   못 찾으면 25자 밀어 재시도하고, 그래도 못 찾으면 **중단**한다(저자 코드의 실패 처리).
6. 사건 리스트를 문장 단위로 다시 세어 문장 사이 경계 벡터를 만든다.

## 🔴 재현 제약 2가지
- **`text-davinci-002` 는 OpenAI 가 폐기했다.** 논문 수치를 그대로 재현하는 것은
  원리적으로 불가능하다. 모델을 바꾸면 그건 재현이 아니라 변형이다.
- 2026-08-03 현재 우리 프록시는 예산 소진(429)이라 어떤 모델로도 실행할 수 없다.

따라서 이 모듈의 지위는 **"절차의 충실한 이식"** 이며, 발표 수치와의 대조는 하지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.llm_client import make_client  # noqa: E402

PAPER_MODEL = "text-davinci-002"     # 폐기됨 — 재현 불가의 근원
CONTEXT_WINDOW = 2048                # 저자 설정 (max 4096 중 절반만 사용)
VERSIONS = ("long ", "")             # 저자의 `versions` — 공백 포함에 주의


def build_prompt(version: str) -> tuple[str, str]:
    """저자 프롬프트 문자열을 그대로 조립한다. version 은 'long ' 또는 ''."""
    head = ("An event is an ongoing coherent situation. The following story needs to be copied "
            "and segmented into " + version + "events. Copy the following story word-for-word "
            "and start a new line whenever one " + version + "event ends and another begins. "
            "This is the story: ")
    tail = ("\n This is a word-for-word copy of the same story that is segmented into "
            + version + "events: ")
    return head, tail


def _has_nonoverlapping_repeat(s: str, L: int) -> bool:
    """길이 L 부분문자열 중 **서로 겹치지 않게** 두 번 나타나는 것이 있는가."""
    if L == 0:
        return True
    if 2 * L > len(s):
        return False
    first: dict[int, int] = {}          # 해시 → 가장 이른 시작 위치
    base, mod = 131, (1 << 61) - 1
    h, power = 0, pow(base, L - 1, mod)
    for i, ch in enumerate(s):
        h = (h * base + ord(ch)) % mod
        if i >= L:
            h = (h - ord(s[i - L]) * base % mod * power % mod) % mod
        if i < L - 1:
            continue
        start = i - L + 1
        prev = first.get(h)
        if prev is None:
            first[h] = start
        elif start - prev >= L and s[prev:prev + L] == s[start:start + L]:
            return True                  # 해시 충돌 방지를 위해 실제 문자열도 대조
    return False


def longest_repeated_substring_len(s: str) -> int:
    """저자 `longestRepeatedSubstring` 이 반환하는 문자열의 **길이**.

    🔴 "가장 긴 반복 부분문자열"이 아니라 **가장 긴 비중첩 반복 부분문자열**이다.
    저자 DP 의 `(j-i) > LCSRe[i-1][j-1]` 조건이 겹침을 배제하기 때문이다
    (예: `aaaa` → 답은 `aa`(2), `aaa`(3) 이 아니다).

    저자의 O(n²) DP 는 수십만 자 원문에서 메모리·시간이 터진다. 길이에 대한 이분탐색 +
    롤링 해시로 같은 값을 O(n log n) 에 구한다. 두 구현의 동치는 `verify_lrs.py` 가
    무작위 문자열로 검사한다.
    """
    if not s:
        return 0
    lo, hi = 0, len(s) // 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _has_nonoverlapping_repeat(s, mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def boundaries_from_events(events: list[str], sentences: list[str],
                           sent_tokenize) -> list[int]:
    """사건 리스트 → 문장 사이 경계 인덱스 (저자 §'find the indices of the sentence bounds').

    각 사건을 문장 단위로 세어 누적하는 방식이라 **단어 수가 아니라 문장 수**로 맞춘다
    (저자 주석: "safer than words").
    """
    idx = -1
    bounds = []
    for ev in events[:-1]:
        n = len(sent_tokenize(ev))
        idx += n
        if 0 <= idx < len(sentences) - 1:
            bounds.append(idx + 1)   # idx+1 번째 문장에서 새 사건이 시작
    return sorted(set(bounds))


def segment_story(text: str, *, version: str = "long ", model: str = PAPER_MODEL,
                  context_window: int = CONTEXT_WINDOW, padding: int = 0,
                  client=None, verbose: bool = True) -> dict:
    """저자 while 루프 이식. 반환 = {events, boundaries, n_calls, stopped_early}."""
    from nltk.tokenize import sent_tokenize
    from transformers import GPT2TokenizerFast

    client = client or make_client()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    all_text = text.replace("...", ".")          # 저자 전처리
    all_sentences = sent_tokenize(all_text)
    min_check_length = longest_repeated_substring_len(all_text) + 1

    head, tail = build_prompt(version)
    n_prompt_tokens = len(tokenizer(head + tail)["input_ids"])
    max_n_story_tokens = (context_window - n_prompt_tokens) // 2
    n_max_token = context_window - (max_n_story_tokens + n_prompt_tokens)

    n_story_tokens_total = len(tokenizer(all_text)["input_ids"])
    max_loop = -(-8 * n_story_tokens_total // n_max_token)   # 저자의 ceil

    start_index, i_loop, finished, stopped_early = 0, 0, False, False
    event_list: list[str] = []

    while not finished:
        i_loop += 1
        if i_loop >= max_loop:
            if verbose:
                print("  ⚠️ too many loops — 저자 코드대로 중단")
            stopped_early = True
            break

        tokens = tokenizer(all_text[start_index:])["input_ids"]
        window = min(max_n_story_tokens, len(tokens))
        new_tokens = tokens[:window]
        text_segment = "".join(tokenizer.decode(x) for x in new_tokens)

        r = client.completions.create(
            model=model, prompt=head + text_segment + tail + " \n",
            temperature=0, max_tokens=n_max_token + padding,
            top_p=1, frequency_penalty=0, presence_penalty=0, logprobs=5,
        )
        response_text = (r.choices[0].text or "").rstrip("\n")
        events = [e for e in response_text.splitlines() if e]

        finished = len(tokens) == len(new_tokens)
        if not finished:
            events = events[:-1]        # 창 끝에서 잘린 마지막 사건은 버린다
        event_list += events

        if finished:
            break

        last_nl = response_text.rfind("\n")
        end_segment = response_text[last_nl + 1:].replace('"', "")
        probe = end_segment[1:min_check_length + 1] if end_segment[:1] == "-" \
            else end_segment[:min_check_length]
        new_start = all_text.rfind(probe)
        if new_start == -1:                                  # 저자의 25자 보정
            off = 26 if end_segment[:1] == "-" else 25
            tmp = all_text.rfind(end_segment[off:min_check_length + off])
            new_start = tmp - 25 if tmp != -1 else -1
        if new_start == -1:
            if verbose:
                print("  ⚠️ 마지막 사건을 원문에서 못 찾음 — 저자 코드대로 중단")
            stopped_early = True
            break
        start_index = new_start

    return {
        "events": event_list,
        "boundaries": boundaries_from_events(event_list, all_sentences, sent_tokenize),
        "n_sentences": len(all_sentences),
        "n_calls": i_loop,
        "stopped_early": stopped_early,
        "min_check_length": min_check_length,
    }
