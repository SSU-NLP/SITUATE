#!/usr/bin/env python3
r"""Nemori 를 이 서버 환경에서 **원본 그대로** 돌리기 위한 부트스트랩.

원본 저장소(`external/nemori`)는 한 줄도 고치지 않는다. 대신 여기서 세 가지만 맞춰준다.

1. **게이트웨이 헤더** — 사내 LLM 게이트웨이(`api.ssunlp.co.kr`)는 Cloudflare Access 헤더를
   요구하는데 nemori 의 `AsyncOpenAI(api_key, base_url)` 호출엔 그 자리가 없다.
   `openai.OpenAI` / `openai.AsyncOpenAI` 를 **서브클래스로 갈아끼워** `default_headers` 를 넣는다.
   ⚠️ `nemori.llm.client` 가 `from openai import AsyncOpenAI` 를 **모듈 최상단에서** 하므로
   이 패치는 반드시 `import nemori` **전에** 실행돼야 한다.
2. **환경변수 이름** — 저장소 `.env` 는 `OPENAI_*` 를 쓰고 nemori 는 `LLM_*` / `EMBEDDING_*` 를
   읽는다. 값을 그대로 복사한다(모델·엔드포인트를 바꾸지 않는다).
3. **모델 라우팅 접두사** — 게이트웨이(litellm)는 `openrouter/` 접두사를 요구한다.
   즉 발표 설정의 `openai/gpt-4.1-mini` → `openrouter/openai/gpt-4.1-mini`.
   **같은 모델**이며 경로 표기만 다르다.

로컬 인프라(이 스크립트가 쓰는 것):
- PostgreSQL 16 @ 127.0.0.1:5432 (db `nemori`, user `nemori`)
- Qdrant 1.18.3 @ 127.0.0.1:6333/6334  (`external/_infra/qdrant`)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NEMORI = REPO / "external/nemori"

# 발표 설정(evaluation/locomo/config.json)의 모델 -> 게이트웨이 경로
PUBLISHED_LLM = "openai/gpt-4.1-mini"
PUBLISHED_EMBEDDING = "google/gemini-embedding-001"
GATEWAY_PREFIX = "openrouter/"


def _load_dotenv() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")


def extra_headers() -> dict[str, str]:
    h = {}
    cid, sec = os.getenv("CF_ACCESS_CLIENT_ID"), os.getenv("CF_ACCESS_CLIENT_SECRET")
    if cid and sec:
        h["CF-Access-Client-Id"] = cid
        h["CF-Access-Client-Secret"] = sec
    return h


def _force_float_embeddings(client) -> None:
    """`encoding_format="float"` 를 강제한다.

    openai SDK 는 numpy 가 있으면 임베딩 요청에 `encoding_format="base64"` 를 **기본으로** 붙인다.
    게이트웨이가 gemini-embedding-001 을 Google AI Studio 로 라우팅하면 그쪽은 base64 를 거부한다
    (`Google AI Studio embeddings do not support base64 encoding_format`).
    모델·차원·값은 그대로이고 **전송 인코딩만** 바꾸는 환경 호환 패치다.
    """
    try:
        emb = client.embeddings
    except Exception:  # pragma: no cover
        return
    original = emb.create

    def create(*args, **kwargs):
        kwargs.setdefault("encoding_format", "float")
        return original(*args, **kwargs)

    emb.create = create


def patch_openai_headers() -> dict[str, str]:
    """`openai.OpenAI`/`AsyncOpenAI` 가 항상 CF-Access 헤더를 붙이게 만든다."""
    import openai

    headers = extra_headers()
    if not headers or getattr(openai, "_hiontop_headers_patched", False):
        return headers

    for name in ("OpenAI", "AsyncOpenAI"):
        base = getattr(openai, name)

        class _Patched(base):  # type: ignore[misc,valid-type]
            def __init__(self, *args, **kwargs):
                merged = dict(headers)
                merged.update(kwargs.get("default_headers") or {})
                kwargs["default_headers"] = merged
                super().__init__(*args, **kwargs)
                _force_float_embeddings(self)

        _Patched.__name__ = name
        _Patched.__qualname__ = name
        setattr(openai, name, _Patched)

    openai._hiontop_headers_patched = True  # type: ignore[attr-defined]
    return headers


def patch_qdrant(timeout: int = 60, log_slow_ms: float = 2000.0) -> None:
    """Qdrant REST 클라이언트의 **읽기 타임아웃을 늘리고** 느린/실패 호출을 추적한다.

    `qdrant_client` 는 REST 타임아웃을 지정 안 하면 httpx 기본값 **5초**를 쓴다. nemori 는 동기
    Qdrant 클라이언트를 asyncio 루프 안에서 부르기 때문에, 대화 10개를 동시에 굴리는 동안
    이벤트 루프가 막히면 5초를 넘겨 `ResponseHandlingException: timed out` 으로 대화 하나가
    통째로 실패한다(1대화 스모크에서 실측). **검색 결과·랭킹을 바꾸지 않는 전송 계층 설정**이므로
    두 팔에 동일하게 적용하고 REPORT 에 공시한다.
    """
    import time
    import traceback

    from qdrant_client import QdrantClient

    if getattr(QdrantClient, "_hiontop_patched", False):
        return
    original_init = QdrantClient.__init__

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("timeout", timeout)
        original_init(self, *args, **kwargs)

    QdrantClient.__init__ = __init__
    QdrantClient._hiontop_patched = True

    from qdrant_client.http.api_client import ApiClient

    original_send = ApiClient.send

    def send(self, request, type_):
        t0 = time.perf_counter()
        try:
            return original_send(self, request, type_)
        except Exception:
            logging.getLogger("nemori").error(
                "Qdrant %s %s 실패 (%.0fms)\n%s", request.method, request.url,
                (time.perf_counter() - t0) * 1000, traceback.format_exc())
            raise
        finally:
            dt = (time.perf_counter() - t0) * 1000
            if dt > log_slow_ms:
                logging.getLogger("nemori").warning(
                    "Qdrant 느림 %s %s %.0fms", request.method, request.url, dt)

    ApiClient.send = send


def patch_buffer_store() -> None:
    """🔴 **upstream 버그 2건 수정** — `PgMessageBufferStore.get_unprocessed`.

    둘 다 두 팔에 **똑같이** 영향을 주고, 고치면 **원본(LLM 분절기) 쪽이 더 강해지는** 방향이라
    baseline 조작이 아니다(프로젝트 규약: 명백한 upstream 버그는 고치되 공시한다).

    ### 버그 1 — 턴 순서가 깨진다 (치명적)
    `ORDER BY timestamp ASC` 에 **동점 처리(tiebreaker)가 없다.** LoCoMo 는 한 세션의 모든 턴이
    `session_N_date_time` **하나의 타임스탬프**를 공유하므로, 세션 안 수십~수백 턴이 전부 동점이 되어
    PostgreSQL 이 임의 순서로 돌려준다. 실측: 어떤 에피소드의 원본 턴 위치가
    `[1, 8, 9, 46, 48, 2, 4, 6, 7, 39, ...]` — 완전히 뒤섞였다.
    대화 순서가 뒤섞이면 **"사건 분절"이라는 과제 자체가 성립하지 않는다**(LLM 분절기도, M5 도).
    → `ORDER BY timestamp ASC, id ASC` (id = BIGSERIAL = 삽입 순서).

    ### 버그 2 — JSONB 가 파이썬 객체로 안 온다
    `push` 는 `json.dumps(content)` 를 `::jsonb` 로 넣는데, asyncpg 는 코덱을 등록하지 않으면
    JSONB 를 **원문 문자열 그대로** 돌려준다. 코드 주석("JSONB returns Python objects directly")과
    반대다. 결과: 모든 메시지 본문에 **바깥 따옴표가 한 겹 더 붙는다**
    (`'"Hey Nate! ..."'`). 멀티모달 리스트도 문자열로 와서 `has_images()` 가 항상 False 가 된다.
    → 읽을 때 `json.loads` 로 되돌린다.
    """
    import json as _json

    from nemori.db.buffer_store import PgMessageBufferStore
    from nemori.domain.models import Message

    if getattr(PgMessageBufferStore, "_hiontop_patched", False):
        return

    async def get_unprocessed(self, user_id: str, agent_id: str) -> list:
        rows = await self._db.fetch(
            """SELECT id, role, content, timestamp
               FROM message_buffer
               WHERE user_id = $1 AND agent_id = $2 AND NOT processed
               ORDER BY timestamp ASC, id ASC""",
            user_id, agent_id,
        )
        result = []
        for row in rows:
            content = row["content"]
            if isinstance(content, (str, bytes)):
                try:
                    content = _json.loads(content)
                except (ValueError, TypeError):
                    pass
            result.append(Message(
                role=row["role"], content=content, timestamp=row["timestamp"],
                metadata={"buffer_id": row["id"]},
            ))
        return result

    PgMessageBufferStore.get_unprocessed = get_unprocessed
    PgMessageBufferStore._hiontop_patched = True


def bootstrap(*, judge_model: str | None = None) -> dict[str, str]:
    """dotenv 로드 -> 환경변수 정규화 -> openai 헤더 패치. `import nemori` 전에 부른다."""
    _load_dotenv()

    key = os.getenv("OPENAI_API_KEY", "")
    base = os.getenv("OPENAI_BASE_URL", "")
    os.environ.setdefault("LLM_API_KEY", key)
    os.environ.setdefault("EMBEDDING_API_KEY", key)
    if base:
        os.environ.setdefault("LLM_BASE_URL", base)
        os.environ.setdefault("EMBEDDING_BASE_URL", base)
    os.environ.setdefault(
        "DATABASE_URL", "postgresql://nemori:nemori@127.0.0.1:5432/nemori"
    )
    os.environ.setdefault("QDRANT_URL", "127.0.0.1")
    os.environ.setdefault("QDRANT_PORT", "6333")
    # metrics/llm_judge.py 가 읽는 심판 모델. 발표 기본값과 동일 모델.
    os.environ.setdefault("LLM_JUDGE_MODEL", judge_model or (GATEWAY_PREFIX + PUBLISHED_LLM))

    patch_openai_headers()          # 반드시 nemori import 보다 먼저
    patch_qdrant(timeout=int(os.getenv("QDRANT_TIMEOUT", "60")))
    add_nemori_to_path()
    patch_buffer_store()
    return {
        "base_url": base,
        "llm_model": GATEWAY_PREFIX + PUBLISHED_LLM,
        "embedding_model": GATEWAY_PREFIX + PUBLISHED_EMBEDDING,
        "judge_model": os.environ["LLM_JUDGE_MODEL"],
        "dsn": os.environ["DATABASE_URL"],
    }


def add_nemori_to_path() -> None:
    import sys

    for p in (str(NEMORI), str(NEMORI / "evaluation"), str(NEMORI / "evaluation/locomo"),
              str(REPO / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
