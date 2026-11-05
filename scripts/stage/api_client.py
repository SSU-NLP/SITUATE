"""Authenticated API transport with per-response usage accounting."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import time


def load_environment():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / "EACL/.env", override=True)


class ApiClient:
    def __init__(self, config):
        from openai import AsyncOpenAI
        load_environment()
        self.config = config
        self.client = AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
            default_headers={"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
                             "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"]},
            timeout=180, max_retries=0)

    async def generate_json(self, *, system_prompt, user_prompt, stage):
        from stage_narrative.clients import JsonCall, ModelResponseParseError, _parse_json_object
        from api_concurrency import request_slot, record_error
        started = time.monotonic()
        for attempt in range(6):
            try:
                async with request_slot(self.config["model"]) as slot:
                    sent_at = time.time()
                    response = await self.client.chat.completions.create(
                        model=self.config["model"], temperature=0, seed=1,
                        max_tokens=self.config.get("max_tokens", 3072),
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": user_prompt}],
                        extra_body=self.config.get("extra_body", {}))
                    service_seconds = time.time() - sent_at
                break
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status == 429:
                    body = json.dumps(getattr(exc, "body", {}), ensure_ascii=False)
                    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET"):
                        secret = os.environ.get(name, "")
                        if len(secret) > 5:
                            body = body.replace(secret, "[redacted]")
                    headers = getattr(getattr(exc, "response", None), "headers", {})
                    detail = {"time": time.time(), "model": self.config["model"], "body": body[:2500],
                              "limits": {k: v for k, v in headers.items()
                                         if "ratelimit" in k.lower() or k.lower() == "retry-after"}}
                    with Path(os.environ["STAGE_USAGE_LEDGER"]).with_name("rate_limit_details.jsonl").open("a") as log:
                        log.write(json.dumps(detail, ensure_ascii=False) + "\n")
                record_error(self.config["model"], status, attempt+1, time.monotonic()-started,
                             type(exc).__name__)
                if attempt == 5 or status in (400, 401, 403, 404):
                    raise RuntimeError(f"API transport failed: {type(exc).__name__}, status={status}") from None
                await asyncio.sleep(min(30, 2 ** (attempt + 1)))
        usage = response.usage.model_dump() if response.usage else {}
        details = usage.get("completion_tokens_details") or {}
        metadata = {"stage": stage, "model": response.model,
                    "sent_at": sent_at, "completed_at": time.time(),
                    "service_seconds": round(service_seconds, 3),
                    "transport_attempt": attempt + 1, **slot,
                    "requested_model": self.config["model"], "response_id": response.id,
                    "provider": getattr(response, "provider", None),
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": details.get("reasoning_tokens", 0),
                    "usage": usage, "elapsed_seconds": round(time.monotonic()-started, 3),
                    "prompt_sha256": hashlib.sha256((system_prompt+'\n'+user_prompt).encode()).hexdigest()}
        ledger = Path(os.environ["STAGE_USAGE_LEDGER"])
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        expected = self.config.get("expected_provider")
        if expected and metadata["provider"] != expected:
            raise ValueError(f"Provider changed: expected {expected}, got {metadata['provider']}")
        text = response.choices[0].message.content or ""
        try:
            data, repaired = _parse_json_object(text)
        except ValueError as exc:
            raise ModelResponseParseError(str(exc), raw_text=text, metadata=metadata) from None
        metadata["deterministic_json_repair"] = repaired
        return JsonCall(data=data, metadata=metadata)


def build_api_client(config, **kwargs):
    return ApiClient(config)
