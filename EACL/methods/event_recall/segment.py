#!/usr/bin/env python3
"""B3-EventRecall (Panela et al. 2025, Communications Psychology) — 분절 파이프라인 이식.

## 범위
논문의 주 기여는 **회상(recall) 자동 채점**이고 분절은 그 전처리다. 우리가 쓰는 것은
분절뿐이므로 `module/segmentation.py` 의 분절 경로만 옮긴다. 임베딩 기반 회상 채점
(`module/recall.py`)은 범위 밖이다. — 이 사실은 결과 보고에 반드시 병기한다.

## 🔴 이 파일이 생긴 이유 (기존 구현의 오분류 정정)
우리 저장소는 `scripts/eacl_phase0/build_tier2_gold.py` 의 `PROMPT_HEAD/TAIL` 을
**"B2 프롬프트"** 라고 부르며 써 왔다. 그런데 원문을 대조하니 그 프롬프트는
**B2(Michelmann)가 아니라 B3(Panela)의 프롬프트**였다. 실측 대조:

    우리 : "...segmented into large events. ... whenever one event ends, and another begins."
    B3   : "...segmented into large events. ... whenever one event ends and another begins."   ← 쉼표만 차이
    B2   : "...segmented into long events.  ... whenever one long event ends and another begins."

꼬리 문장은 우리 것과 B3 가 **완전히 동일**하고("large event units:"), B2 는 다르다("long events:").
즉 우리는 **B3 를 B2 라고 부르며 쉼표 하나를 더한 판**을 돌리고 있었다.
이 모듈은 쉼표 없는 **B3 원문 그대로**를 쓴다.

## 저자 절차 (module/segmentation.py)
1. `clean_text` — 큰따옴표(", ", ")·`...`→`.`·하이픈→공백 치환. **원 코드 그대로 이식.**
2. 프롬프트 = onset + 전문 + offset (슬라이딩 창 없이 **한 번에**).
3. `prompt_gpt` — chat, temperature=0, `max_completion_tokens=4096`, penalty 0.
4. `finish_reason == 'stop'` 인 응답만 남긴다(잘린 출력은 버린다).
5. `parse_llm_output` — 정규식 7단계로 개행을 `\n\n` 로 정규화하고 "Event N:" 류 머리표를 제거.
6. `split_text` — `\n\n` 로 잘라 사건 리스트.

## 🔴 재현 제약
논문은 GPT-4 와 Meta-Llama-3-8B-Instruct 를 쓴다. 2026-08-03 현재 우리 프록시는 예산
소진(429 ExceededBudget)이라 **실행 불가**다. 코드는 예산 복구 즉시 돌아가게 해 둔다.
다른 모델로 바꿔 돌린 값은 이 논문의 재현이 아니다([[baselines-stay-as-published]]).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.llm_client import make_client  # noqa: E402

# 저자 코드 `module/segmentation.py` L77-78 문자열 그대로 (쉼표 없음에 주의)
PROMPT_ONSET = ("An event is an ongoing coherent situation. The following story needs to be copied "
                "and segmented into large events. Copy the following story word-for-word and start "
                "a new line whenever one event ends and another begins. This is the story: ")
PROMPT_OFFSET = ("\n This is a word-for-word copy of the same story that is segmented into "
                 "large event units: ")

PAPER_MODELS = ("gpt-4", "meta-llama/Meta-Llama-3-8B-Instruct")


def clean_text(text: str) -> str:
    """저자 `clean_text` 이식. 🔴 하이픈을 공백으로 바꾸는 것까지 원 코드 그대로다."""
    for a, b in (('"', ""), ("“", ""), ("”", ""), ("...", "."), ("-", " ")):
        text = text.replace(a, b)
    return text


def parse_llm_output(output: str) -> str:
    """저자 `parse_llm_output` 이식 — 정규식 순서까지 동일해야 같은 분절이 나온다."""
    t = re.sub(r"(?<=\n)\s(?=\n)", "\n\n", output)
    t = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", t)
    t = re.sub(r"[0-9]+\.", "", t)
    t = re.sub(r"Event \d+:", "", t)
    t = re.sub(r"Segment \d+: ", "", t)
    t = re.sub(r"(?<=\n\n)\s", "", t)
    t = re.sub(r" (?=\n\n)", "", t)
    t = re.sub(r"^\s", "", t)
    return t.lstrip()


def split_text(parsed: str) -> list[str]:
    """저자 `split_text` 이식 — `\\n\\n` 단위로 사건을 나눈다."""
    return parsed.split("\n\n")


def segment_text_llama(text: str, *, model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                       iters: int = 1, temperature: float = 0.1,
                       max_new_tokens: int = 4096, frequency_penalty: float = 0.0,
                       pipe=None, sample: bool = False) -> tuple[list[list[str]], list[str]]:
    """저자 `llama_segmentation` 이식 — **로컬 실행, API 불필요**.

    저자 코드(`module/segmentation.py:50-58, 91-107`)의 세부를 그대로 따른다:
      - `pipeline("text-generation", model, device_map="auto")`
      - `repetition_penalty = 1 + frequency_penalty`
      - GPT 경로와 달리 **`finish_reason` 필터가 없다** — 잘린 출력도 그대로 채택된다.

    🔴 **저자 코드는 `do_sample` 을 넘기지 않는다** (`module/segmentation.py:54-57`).
    HuggingFace 기본값이 `do_sample=False` 이므로 실제 동작은 **greedy 디코딩**이고
    `temperature` 는 **무시된다**(HF 가 경고만 낸다). 즉 저자 코드의
    `if temperature == 0: temperature = 0.1` 강제는 **실질적으로 아무 효과가 없다** —
    Llama 경로는 애초에 결정적이다. 우리도 `do_sample` 을 넘기지 않아 그 동작을 그대로 따른다.
    (`--sample` 로 굳이 샘플링을 켜면 그건 **저자와 다른 조건**이므로 산출에 표시된다.)

    반환: (반복별 사건 리스트, 반복별 원 응답). 원 응답을 함께 돌려주는 것은 우리 추가다 —
    저자 코드는 버리지만, 잘림 여부를 사후에 판정하려면 필요하다.
    """
    from transformers import pipeline as hf_pipeline

    if temperature == 0:
        temperature = 0.1                      # 저자 코드의 강제 규칙 (greedy 에서는 무시됨)
    pipe = pipe or hf_pipeline("text-generation", model, device_map="auto")

    cleaned = clean_text(text)
    prompt = PROMPT_ONSET + cleaned + PROMPT_OFFSET

    # 저자와 동일: do_sample 을 넘기지 않는다 → HF 기본값 False → greedy.
    # `sample=True` 로 켜면 저자와 다른 조건이 되므로 호출자가 명시해야 한다.
    extra = {"do_sample": True, "temperature": temperature} if sample else {}

    events, raws = [], []
    for _ in range(iters):
        out = pipe([{"role": "user", "content": prompt}],
                   max_new_tokens=max_new_tokens,
                   repetition_penalty=1 + frequency_penalty, **extra)
        gen = out[0]["generated_text"]
        txt = gen[-1]["content"] if isinstance(gen, list) else str(gen)
        raws.append(txt)
        events.append(split_text(parse_llm_output(txt)))
    return events, raws


def segment_text(text: str, *, model: str = "gpt-4", iters: int = 1,
                 temperature: float = 0.0, max_completion_tokens: int = 4096,
                 client=None) -> tuple[list[list[str]], list[dict]]:
    """저자 `gpt_segmentation` 이식. 반환 = (채택된 반복별 사건 리스트, 반복별 원자료).

    `finish_reason != 'stop'` 인 반복은 저자 코드대로 **버린다**(사건 리스트에서 제외).

    🔴 **원 응답을 반드시 남긴다.** 저자 코드는 버리지만, 우리 저장소에는 원 응답을 저장하지 않아
    사후 재분석이 통째로 막힌 전례가 있다 — `main_result.md` 의 GPT-5 "경계 0개" 행이
    퇴화인지 비실행인지 지금도 판정할 수 없다(`REPRO.md` §6). 두 번째 반환값에
    `{raw, finish_reason, served_model, n_words}` 를 담아 그 실수를 반복하지 않는다.
    """
    import re

    client = client or make_client()
    cleaned = clean_text(text)
    prompt = PROMPT_ONSET + cleaned + PROMPT_OFFSET
    word = re.compile(r"\w+")
    n_src = len(word.findall(cleaned))

    kept, meta = [], []
    for _ in range(iters):
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            frequency_penalty=0,
            presence_penalty=0,
        )
        choice = r.choices[0]
        raw = choice.message.content or ""
        meta.append({
            "raw": raw,
            "finish_reason": choice.finish_reason,
            "served_model": getattr(r, "model", "?"),   # 실제 서빙된 모델 (요청명과 다를 수 있다)
            "copy_fidelity": round(len(word.findall(raw)) / max(1, n_src), 3),
        })
        if choice.finish_reason != "stop":
            print(f"  ⚠️ finish_reason={choice.finish_reason} — 저자 절차대로 이 반복은 버린다")
            continue
        kept.append(split_text(parse_llm_output(raw)))
    return kept, meta
