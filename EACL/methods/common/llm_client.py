#!/usr/bin/env python3
"""LLM 프롬프팅 계열 베이스라인이 공유하는 클라이언트.

저장소 관례(`scripts/llm_judge_universal.py`)를 그대로 따른다: OpenAI 호환 프록시 +
Cloudflare Access 서비스 토큰 헤더. 키는 `.env` 에서 읽는다.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def load_env() -> None:
    """저장소 루트 `.env` 를 환경변수로 올린다(이미 있는 값은 덮지 않는다)."""
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def make_client():
    load_env()
    from openai import OpenAI

    hdr = {}
    if os.environ.get("CF_ACCESS_CLIENT_ID"):
        hdr["CF-Access-Client-Id"] = os.environ["CF_ACCESS_CLIENT_ID"]
        hdr["CF-Access-Client-Secret"] = os.environ["CF_ACCESS_CLIENT_SECRET"]
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                  base_url=os.environ["OPENAI_BASE_URL"],
                  default_headers=hdr or None)


def chat(client, model: str, system: str, user: str, *, temperature: float = 0.0,
         max_retries: int = 4, timeout: float = 300.0) -> str:
    """한 번의 chat 호출. 일시 오류는 지수 백오프로 재시도한다."""
    last = None
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature,
                timeout=timeout,
            )
            return r.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 — 프록시가 던지는 예외 종류가 다양하다
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM 호출 {max_retries}회 실패: {last}")


def extract_json(text: str):
    """모델 출력에서 첫 JSON 객체/배열을 꺼낸다. 코드펜스·설명문이 섞여도 견딘다."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    for opener, closer in (("[", "]"), ("{", "}")):
        i = text.find(opener)
        if i < 0:
            continue
        depth, in_str, esc = 0, False, False
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"JSON 을 찾지 못했다: {text[:200]!r}")
